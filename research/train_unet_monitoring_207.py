from pathlib import Path
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT = Path.home() / "opm_runs" / "spe11_crossscale_project"
DATA = PROJECT / "ai_dataset_npz_207"
MATRIX = PROJECT / "scenario_matrix_valid_207.csv"
OUT = PROJECT / "ai_results_unet_monitoring_207"
OUT.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


def prepare_index():
    idx = pd.read_csv(DATA / "dataset_index.csv")
    mat = pd.read_csv(MATRIX)

    keep_cols = [
        "scenario_id",
        "well1_rate_mult",
        "well2_rate_mult",
        "q_mult",
        "kz_rel_mult",
        "global_perm_mult",
        "source_A_split",
        "target_B_role",
    ]

    mat = mat[keep_cols].copy()

    # Avoid duplicated metadata columns from dataset_index
    drop_cols = [
        "well1_rate_mult", "well2_rate_mult",
        "q_mult", "kz_rel_mult", "global_perm_mult",
        "source_A_split", "target_B_role",
    ]
    idx = idx.drop(columns=[c for c in drop_cols if c in idx.columns], errors="ignore")

    idx = idx.merge(mat, on="scenario_id", how="left")

    missing = idx[["well1_rate_mult", "well2_rate_mult"]].isna().sum().sum()
    if missing > 0:
        raise RuntimeError("Missing well-rate multipliers after merge.")

    return idx


class PlumePairDataset(Dataset):
    def __init__(self, case, scenario_ids, index_df):
        self.case = case
        self.samples = []

        for sid in scenario_ids:
            row = index_df[(index_df["case"] == case) & (index_df["scenario_id"] == sid)].iloc[0]
            npz_path = Path(row["npz_path"])
            z = np.load(npz_path)
            n = len(z["times"])

            # Monitoring-only forecasting: exclude t=0 input
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
                        "global_perm_mult": float(row["global_perm_mult"]),
                        "well1_rate_mult": float(row["well1_rate_mult"]),
                        "well2_rate_mult": float(row["well2_rate_mult"]),
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

        Xdyn = Xall[i].copy()  # [4,H,W]

        # Per-sample dynamic scaling
        if np.nanmax(np.abs(Xdyn[0])) > 0:
            Xdyn[0] = Xdyn[0] / np.nanmax(np.abs(Xdyn[0]))

        if np.nanmax(np.abs(Xdyn[2])) > 0:
            Xdyn[2] = Xdyn[2] / np.nanmax(np.abs(Xdyn[2]))

        H, W = Xdyn.shape[-2:]

        tau_i = i / 5.0
        tau_j = j / 5.0
        dtau = tau_j - tau_i

        q_norm = (s["q_mult"] - 1.0) / 0.35
        kz_norm = math.log(float(s["kz_rel_mult"]), 2.0)
        perm_norm = (s["global_perm_mult"] - 1.0) / 0.25
        w1_norm = (s["well1_rate_mult"] - 1.0) / 0.7
        w2_norm = (s["well2_rate_mult"] - 1.0) / 0.7

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

        X = np.concatenate([Xdyn, consts], axis=0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        y = yall[j][None, :, :].astype(np.float32)

        return torch.from_numpy(X), torch.from_numpy(y), s["scenario_id"], i, j


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch=12, base=16):
        super().__init__()

        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        _, _, h, w = x.shape

        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x = nn.functional.pad(x, (0, pw, 0, ph))

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        m = self.mid(self.pool3(d3))

        u3 = self.up3(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        y = self.out(u1)
        return y[:, :, :h, :w]


def loss_fn(logits, y):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y)

    prob = torch.sigmoid(logits)
    inter = (prob * y).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (denom + 1e-6)

    return bce + (1 - dice.mean())


def calc_metrics(logits, y):
    pred = (torch.sigmoid(logits) > 0.5).float()

    inter = (pred * y).sum(dim=(1, 2, 3))
    union = ((pred + y) > 0).float().sum(dim=(1, 2, 3))

    pred_area = pred.sum(dim=(1, 2, 3))
    true_area = y.sum(dim=(1, 2, 3))

    iou = (inter / (union + 1e-6)).detach().cpu().numpy()
    dice = (2 * inter / (pred_area + true_area + 1e-6)).detach().cpu().numpy()
    area_rel_error = (torch.abs(pred_area - true_area) / (true_area + 1e-6)).detach().cpu().numpy()

    return iou, dice, area_rel_error


def train_model(model, train_loader, val_loader, epochs, lr, save_path, tag):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_iou = -1.0
    rows = []

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for X, y, *_ in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        model.eval()
        val_iou = []
        val_dice = []
        val_area = []

        with torch.no_grad():
            for X, y, *_ in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(X)
                iou, dice, area = calc_metrics(logits, y)

                val_iou.extend(iou.tolist())
                val_dice.extend(dice.tolist())
                val_area.extend(area.tolist())

        row = {
            "tag": tag,
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            "val_iou": float(np.mean(val_iou)),
            "val_dice": float(np.mean(val_dice)),
            "val_area_rel_error": float(np.mean(val_area)),
        }
        rows.append(row)

        print(
            f"[{tag}] epoch {ep:03d} | "
            f"loss={row['train_loss']:.4f} "
            f"val_iou={row['val_iou']:.4f} "
            f"val_dice={row['val_dice']:.4f} "
            f"val_area_err={row['val_area_rel_error']:.4f}"
        )

        if row["val_iou"] > best_iou:
            best_iou = row["val_iou"]
            torch.save(model.state_dict(), save_path)

    return pd.DataFrame(rows)


def evaluate(model, ds, tag):
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model.to(DEVICE)
    model.eval()

    rows = []

    with torch.no_grad():
        for X, y, sid, i, j in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            iou, dice, area = calc_metrics(logits, y)

            rows.append({
                "eval_tag": tag,
                "case": ds.case,
                "scenario_id": sid[0],
                "i": int(i.item()),
                "j": int(j.item()),
                "horizon": int(j.item()) - int(i.item()),
                "iou": float(iou[0]),
                "dice": float(dice[0]),
                "area_rel_error": float(area[0]),
            })

    return pd.DataFrame(rows)


def binary_metrics_np(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)

    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    pred_area = pred.sum()
    true_area = true.sum()

    iou = inter / (union + 1e-6)
    dice = (2 * inter) / (pred_area + true_area + 1e-6)
    area_rel_error = abs(pred_area - true_area) / (true_area + 1e-6)

    return float(iou), float(dice), float(area_rel_error)


def evaluate_persistence(ds, tag):
    rows = []

    for s in ds.samples:
        z = np.load(s["npz_path"])
        masks = z["plume_mask"].astype(np.uint8)

        i = s["i"]
        j = s["j"]

        pred = masks[i]
        true = masks[j]

        iou, dice, area = binary_metrics_np(pred, true)

        rows.append({
            "eval_tag": tag,
            "case": ds.case,
            "scenario_id": s["scenario_id"],
            "i": i,
            "j": j,
            "horizon": j - i,
            "iou": iou,
            "dice": dice,
            "area_rel_error": area,
        })

    return pd.DataFrame(rows)


def paired_comparison(df):
    idcols = ["case", "scenario_id", "i", "j", "horizon"]
    metric_cols = ["iou", "dice", "area_rel_error"]

    ft_tag = "B_final_finetuned_long_model"
    baselines = [
        "B_final_persistence",
        "B_zero_shot_source_model",
        "B_final_scratch_long_model",
    ]

    rows = []
    ft = df[df["eval_tag"] == ft_tag][idcols + metric_cols].copy()

    for base_tag in baselines:
        base = df[df["eval_tag"] == base_tag][idcols + metric_cols].copy()
        merged = ft.merge(base, on=idcols, suffixes=("_ft", "_base"))

        for metric in metric_cols:
            if metric == "area_rel_error":
                diff = merged[f"{metric}_base"] - merged[f"{metric}_ft"]
            else:
                diff = merged[f"{metric}_ft"] - merged[f"{metric}_base"]

            rows.append({
                "comparison": f"{ft_tag} vs {base_tag}",
                "metric": metric,
                "n": len(diff),
                "mean_improvement": float(diff.mean()),
                "median_improvement": float(diff.median()),
                "std_improvement": float(diff.std()),
                "fraction_positive": float((diff > 0).mean()),
            })

    return pd.DataFrame(rows)


def main():
    index = prepare_index()

    a_train = index[(index["case"] == "spe11a") & (index["source_A_split"] == "train")]["scenario_id"].tolist()
    a_val = index[(index["case"] == "spe11a") & (index["source_A_split"] == "val")]["scenario_id"].tolist()
    a_test = index[(index["case"] == "spe11a") & (index["source_A_split"] == "test")]["scenario_id"].tolist()

    b_ft_train = index[(index["case"] == "spe11b") & (index["target_B_role"] == "finetune_train")]["scenario_id"].tolist()
    b_ft_val = index[(index["case"] == "spe11b") & (index["target_B_role"] == "finetune_val")]["scenario_id"].tolist()
    b_final = index[(index["case"] == "spe11b") & (index["target_B_role"] == "zero_shot_and_final_test")]["scenario_id"].tolist()

    print("A train/val/test scenarios:", len(a_train), len(a_val), len(a_test))
    print("B ft train/val/final scenarios:", len(b_ft_train), len(b_ft_val), len(b_final))

    a_train_ds = PlumePairDataset("spe11a", a_train, index)
    a_val_ds = PlumePairDataset("spe11a", a_val, index)
    a_test_ds = PlumePairDataset("spe11a", a_test, index)

    b_ft_train_ds = PlumePairDataset("spe11b", b_ft_train, index)
    b_ft_val_ds = PlumePairDataset("spe11b", b_ft_val, index)
    b_final_ds = PlumePairDataset("spe11b", b_final, index)

    print("Pair counts:")
    print("A train:", len(a_train_ds))
    print("A val:", len(a_val_ds))
    print("A test:", len(a_test_ds))
    print("B ft train:", len(b_ft_train_ds))
    print("B ft val:", len(b_ft_val_ds))
    print("B final:", len(b_final_ds))

    # ------------------------------------------------------------
    # Persistence baselines
    # ------------------------------------------------------------
    persistence = pd.concat([
        evaluate_persistence(a_test_ds, "A_test_persistence"),
        evaluate_persistence(b_final_ds, "B_final_persistence"),
    ], ignore_index=True)

    # ------------------------------------------------------------
    # SPE11A source model
    # ------------------------------------------------------------
    source_path = OUT / "small_unet_spe11a_source_best.pt"

    source_model = SmallUNet(in_ch=12, base=16)

    hist_a = train_model(
        source_model,
        DataLoader(a_train_ds, batch_size=4, shuffle=True, num_workers=0),
        DataLoader(a_val_ds, batch_size=4, shuffle=False, num_workers=0),
        epochs=25,
        lr=1e-3,
        save_path=source_path,
        tag="SPE11A_source_207"
    )
    hist_a.to_csv(OUT / "history_spe11a_source_207.csv", index=False)

    source_model = SmallUNet(in_ch=12, base=16)
    source_model.load_state_dict(torch.load(source_path, map_location=DEVICE))

    a_test_source = evaluate(source_model, a_test_ds, "A_test_source_model")
    b_zero = evaluate(source_model, b_final_ds, "B_zero_shot_source_model")

    # ------------------------------------------------------------
    # SPE11B fine-tuning from SPE11A source
    # ------------------------------------------------------------
    ft_path = OUT / "small_unet_spe11b_finetuned_long_best.pt"

    ft_model = SmallUNet(in_ch=12, base=16)
    ft_model.load_state_dict(torch.load(source_path, map_location=DEVICE))

    hist_ft = train_model(
        ft_model,
        DataLoader(b_ft_train_ds, batch_size=4, shuffle=True, num_workers=0),
        DataLoader(b_ft_val_ds, batch_size=4, shuffle=False, num_workers=0),
        epochs=30,
        lr=1e-3,
        save_path=ft_path,
        tag="SPE11B_finetune_long_207"
    )
    hist_ft.to_csv(OUT / "history_spe11b_finetune_long_207.csv", index=False)

    ft_model = SmallUNet(in_ch=12, base=16)
    ft_model.load_state_dict(torch.load(ft_path, map_location=DEVICE))

    b_ft_final = evaluate(ft_model, b_final_ds, "B_final_finetuned_long_model")

    # ------------------------------------------------------------
    # SPE11B scratch long
    # ------------------------------------------------------------
    scratch_path = OUT / "small_unet_spe11b_scratch_long_best.pt"

    scratch_model = SmallUNet(in_ch=12, base=16)

    hist_scratch = train_model(
        scratch_model,
        DataLoader(b_ft_train_ds, batch_size=4, shuffle=True, num_workers=0),
        DataLoader(b_ft_val_ds, batch_size=4, shuffle=False, num_workers=0),
        epochs=30,
        lr=1e-3,
        save_path=scratch_path,
        tag="SPE11B_scratch_long_207"
    )
    hist_scratch.to_csv(OUT / "history_spe11b_scratch_long_207.csv", index=False)

    scratch_model = SmallUNet(in_ch=12, base=16)
    scratch_model.load_state_dict(torch.load(scratch_path, map_location=DEVICE))

    b_scratch_final = evaluate(scratch_model, b_final_ds, "B_final_scratch_long_model")

    # ------------------------------------------------------------
    # Save all metrics
    # ------------------------------------------------------------
    all_metrics = pd.concat([
        persistence,
        a_test_source,
        b_zero,
        b_ft_final,
        b_scratch_final,
    ], ignore_index=True)

    all_metrics.to_csv(OUT / "metrics_all_207.csv", index=False)

    summary = all_metrics.groupby("eval_tag")[["iou", "dice", "area_rel_error"]].agg(["mean", "std", "median", "count"])
    by_horizon = all_metrics.groupby(["eval_tag", "horizon"])[["iou", "dice", "area_rel_error"]].agg(["mean", "std", "median", "count"])

    summary.to_csv(OUT / "metrics_summary_207.csv")
    by_horizon.to_csv(OUT / "metrics_by_horizon_207.csv")

    paired = paired_comparison(all_metrics)
    paired.to_csv(OUT / "paired_comparison_207.csv", index=False)

    print()
    print("===== Metrics summary 207 =====")
    print(summary)

    print()
    print("===== Metrics by horizon 207 =====")
    print(by_horizon)

    print()
    print("===== Paired comparison 207 =====")
    print(paired.to_string(index=False))

    print()
    print("Saved results to:", OUT)


if __name__ == "__main__":
    main()
