from pathlib import Path
import math
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import train_unet_monitoring_207 as base


# ============================================================
# Paths / settings
# ============================================================

PROJECT = Path.home() / "opm_runs" / "spe11_crossscale_project"
ROOT = PROJECT / "v2_analysis" / "channel_ablation"
ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = base.DEVICE

CONFIGS = {
    "S_dsgas_only": [1],
    "SX_dsgas_xco2": [1, 2],
    "SPX_pressure_dsgas_xco2": [0, 1, 2],
}

CHANNEL_NAMES = {
    0: "delta_pressure",
    1: "delta_sgas",
    2: "xco2_liquid",
    3: "active_mask",
}


# ============================================================
# Dataset
# ============================================================

class AblationDataset(Dataset):

    def __init__(self, case, scenario_ids, index_df, dyn_channels):

        self.case = case
        self.dyn_channels = dyn_channels
        self.samples = []

        for sid in scenario_ids:

            row = index_df[
                (index_df["case"] == case)
                & (index_df["scenario_id"] == sid)
            ].iloc[0]

            npz_path = Path(row["npz_path"])
            z = np.load(npz_path)
            n = len(z["times"])

            # Monitoring-only:
            # exclude zero-plume initial state
            for i in range(1, n - 1):
                for j in range(i + 1, n):

                    self.samples.append({
                        "case": case,
                        "scenario_id": sid,
                        "npz_path": str(npz_path),
                        "i": i,
                        "j": j,
                        "q_mult": float(row["q_mult"]),
                        "kz_rel_mult": float(row["kz_rel_mult"]),
                        "global_perm_mult":
                            float(row["global_perm_mult"]),
                        "well1_rate_mult":
                            float(row["well1_rate_mult"]),
                        "well2_rate_mult":
                            float(row["well2_rate_mult"]),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        s = self.samples[idx]
        z = np.load(s["npz_path"])

        Xall = z["X"].astype(np.float32)
        yall = z["plume_mask"].astype(np.float32)

        i = s["i"]
        j = s["j"]

        # ----------------------------------------------------
        # Select requested dynamic channels
        # ----------------------------------------------------
        dyn_list = []

        for c in self.dyn_channels:

            a = Xall[i, c].copy()

            # Preserve original preprocessing:
            # pressure and XCO2 are scaled by each image max
            if c in [0, 2]:
                vmax = np.nanmax(np.abs(a))
                if vmax > 0:
                    a = a / vmax

            dyn_list.append(a.astype(np.float32))

        Xdyn = np.stack(dyn_list, axis=0)

        H, W = Xdyn.shape[-2:]

        # ----------------------------------------------------
        # Same temporal / scenario conditioning
        # as original model
        # ----------------------------------------------------
        tau_i = i / 5.0
        tau_j = j / 5.0
        dtau = tau_j - tau_i

        q_norm = (s["q_mult"] - 1.0) / 0.35
        kz_norm = math.log(
            float(s["kz_rel_mult"]), 2.0
        )
        perm_norm = (
            s["global_perm_mult"] - 1.0
        ) / 0.25
        w1_norm = (
            s["well1_rate_mult"] - 1.0
        ) / 0.7
        w2_norm = (
            s["well2_rate_mult"] - 1.0
        ) / 0.7

        consts = np.stack([
            np.full((H, W), tau_i, dtype=np.float32),
            np.full((H, W), tau_j, dtype=np.float32),
            np.full((H, W), dtau, dtype=np.float32),

            np.full((H, W), q_norm, dtype=np.float32),
            np.full((H, W), kz_norm, dtype=np.float32),
            np.full((H, W), perm_norm, dtype=np.float32),
            np.full((H, W), w1_norm, dtype=np.float32),
            np.full((H, W), w2_norm, dtype=np.float32),
        ], axis=0)

        X = np.concatenate(
            [Xdyn, consts],
            axis=0
        )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        y = yall[j][None, :, :].astype(np.float32)

        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            s["scenario_id"],
            i,
            j,
        )


# ============================================================
# Run one channel configuration
# ============================================================

def run_config(
    config_name,
    dyn_channels,
    index,
    a_train,
    a_val,
    a_test,
    b_train,
    b_val,
    b_final,
):

    print()
    print("=" * 80)
    print("CONFIG:", config_name)
    print(
        "Dynamic channels:",
        [CHANNEL_NAMES[c] for c in dyn_channels]
    )
    print("=" * 80)

    # Reset random state so configurations are comparable
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out = ROOT / config_name
    out.mkdir(parents=True, exist_ok=True)

    a_train_ds = AblationDataset(
        "spe11a", a_train, index, dyn_channels
    )
    a_val_ds = AblationDataset(
        "spe11a", a_val, index, dyn_channels
    )
    a_test_ds = AblationDataset(
        "spe11a", a_test, index, dyn_channels
    )

    b_train_ds = AblationDataset(
        "spe11b", b_train, index, dyn_channels
    )
    b_val_ds = AblationDataset(
        "spe11b", b_val, index, dyn_channels
    )
    b_final_ds = AblationDataset(
        "spe11b", b_final, index, dyn_channels
    )

    in_ch = len(dyn_channels) + 8

    print("Total input channels:", in_ch)

    # --------------------------------------------------------
    # 1. SPE11A source training
    # --------------------------------------------------------

    source_path = out / "source_best.pt"

    source_model = base.SmallUNet(
        in_ch=in_ch,
        base=16
    )

    hist = base.train_model(
        source_model,
        DataLoader(
            a_train_ds,
            batch_size=4,
            shuffle=True,
            num_workers=0
        ),
        DataLoader(
            a_val_ds,
            batch_size=4,
            shuffle=False,
            num_workers=0
        ),
        epochs=25,
        lr=1e-3,
        save_path=source_path,
        tag=f"{config_name}_A_source"
    )

    hist.to_csv(
        out / "history_A_source.csv",
        index=False
    )

    source_model = base.SmallUNet(
        in_ch=in_ch,
        base=16
    )

    source_model.load_state_dict(
        torch.load(
            source_path,
            map_location=DEVICE
        )
    )

    a_test_result = base.evaluate(
        source_model,
        a_test_ds,
        f"{config_name}_A_test"
    )

    zero_result = base.evaluate(
        source_model,
        b_final_ds,
        f"{config_name}_B_zero"
    )

    # --------------------------------------------------------
    # 2. SPE11A pretrain -> SPE11B fine-tune
    # --------------------------------------------------------

    ft_path = out / "finetuned_best.pt"

    ft_model = base.SmallUNet(
        in_ch=in_ch,
        base=16
    )

    ft_model.load_state_dict(
        torch.load(
            source_path,
            map_location=DEVICE
        )
    )

    hist_ft = base.train_model(
        ft_model,
        DataLoader(
            b_train_ds,
            batch_size=4,
            shuffle=True,
            num_workers=0
        ),
        DataLoader(
            b_val_ds,
            batch_size=4,
            shuffle=False,
            num_workers=0
        ),
        epochs=30,
        lr=1e-3,
        save_path=ft_path,
        tag=f"{config_name}_B_finetune"
    )

    hist_ft.to_csv(
        out / "history_B_finetune.csv",
        index=False
    )

    ft_model = base.SmallUNet(
        in_ch=in_ch,
        base=16
    )

    ft_model.load_state_dict(
        torch.load(
            ft_path,
            map_location=DEVICE
        )
    )

    ft_result = base.evaluate(
        ft_model,
        b_final_ds,
        f"{config_name}_B_finetuned"
    )

    # --------------------------------------------------------
    # 3. SPE11B scratch
    # --------------------------------------------------------

    scratch_path = out / "scratch_best.pt"

    scratch_model = base.SmallUNet(
        in_ch=in_ch,
        base=16
    )

    hist_sc = base.train_model(
        scratch_model,
        DataLoader(
            b_train_ds,
            batch_size=4,
            shuffle=True,
            num_workers=0
        ),
        DataLoader(
            b_val_ds,
            batch_size=4,
            shuffle=False,
            num_workers=0
        ),
        epochs=30,
        lr=1e-3,
        save_path=scratch_path,
        tag=f"{config_name}_B_scratch"
    )

    hist_sc.to_csv(
        out / "history_B_scratch.csv",
        index=False
    )

    scratch_model = base.SmallUNet(
        in_ch=in_ch,
        base=16
    )

    scratch_model.load_state_dict(
        torch.load(
            scratch_path,
            map_location=DEVICE
        )
    )

    scratch_result = base.evaluate(
        scratch_model,
        b_final_ds,
        f"{config_name}_B_scratch"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    metrics = pd.concat([
        a_test_result,
        zero_result,
        scratch_result,
        ft_result,
    ], ignore_index=True)

    metrics.to_csv(
        out / "metrics.csv",
        index=False
    )

    summary = (
        metrics.groupby("eval_tag")[
            ["iou", "dice", "area_rel_error"]
        ]
        .agg(["mean", "std", "median", "count"])
    )

    summary.to_csv(
        out / "summary.csv"
    )

    print()
    print(summary)

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    index = base.prepare_index()

    a_train = index[
        (index["case"] == "spe11a")
        & (index["source_A_split"] == "train")
    ]["scenario_id"].tolist()

    a_val = index[
        (index["case"] == "spe11a")
        & (index["source_A_split"] == "val")
    ]["scenario_id"].tolist()

    a_test = index[
        (index["case"] == "spe11a")
        & (index["source_A_split"] == "test")
    ]["scenario_id"].tolist()

    b_train = index[
        (index["case"] == "spe11b")
        & (
            index["target_B_role"]
            == "finetune_train"
        )
    ]["scenario_id"].tolist()

    b_val = index[
        (index["case"] == "spe11b")
        & (
            index["target_B_role"]
            == "finetune_val"
        )
    ]["scenario_id"].tolist()

    b_final = index[
        (index["case"] == "spe11b")
        & (
            index["target_B_role"]
            == "zero_shot_and_final_test"
        )
    ]["scenario_id"].tolist()

    all_results = []

    for config_name, channels in CONFIGS.items():

        r = run_config(
            config_name,
            channels,
            index,
            a_train,
            a_val,
            a_test,
            b_train,
            b_val,
            b_final,
        )

        r["configuration"] = config_name
        all_results.append(r)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined.to_csv(
        ROOT / "channel_ablation_metrics_all.csv",
        index=False
    )

    print()
    print("=" * 100)
    print("FINAL CHANNEL ABLATION SUMMARY")
    print("=" * 100)

    final = (
        combined.groupby(
            ["configuration", "eval_tag"]
        )[["iou", "dice", "area_rel_error"]]
        .mean()
    )

    print(final.to_string())

    final.to_csv(
        ROOT / "channel_ablation_summary.csv"
    )

    print()
    print("Saved to:")
    print(ROOT)


if __name__ == "__main__":
    main()
