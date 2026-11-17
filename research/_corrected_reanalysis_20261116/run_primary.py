#!/usr/bin/env python3
import sys,subprocess,time,json
from pathlib import Path
R=Path(__file__).resolve().parent
assert sys.executable=='/home/yssongvd/opm-ai-env/bin/python'
assert (R/'PREFLIGHT_PASS.json').exists()
assert not (R/'RUN_STARTED.json').exists(),'Already started: inspect existing run before resuming'
(R/'RUN_STARTED.json').write_text(json.dumps({'started_unix':time.time(),'python':sys.executable,'seeds':[42,11,22,33,44,55],'device':'cpu','threads':12},indent=2))
def run(args,name):
 command=[sys.executable,'-u','-B',str(R/'train_primary.py'),*args]
 with open(R/f'{name}.log','x') as log:
  p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=R)
  for line in p.stdout:
   log.write(line);log.flush();print(line,end='',flush=True)
  code=p.wait()
 if code:
  (R/'RUN_FAILED.json').write_text(json.dumps({'stage':name,'exit_code':code,'time':time.time()},indent=2))
  raise SystemExit(code)
for seed in [42,11,22,33,44,55]:
 if seed!=42:assert (R/'SEED42_PIPELINE_PASS.json').exists()
 run(['train','--seed',str(seed)],f'train_seed_{seed}')
run(['freeze'],'freeze_models')
run(['evaluate'],'final_evaluation')
(R/'TRAIN_AND_TEST_COMPLETE.json').write_text(json.dumps({'complete_unix':time.time(),'seeds':6,'jobs':18},indent=2))
print('PRIMARY TRAINING AND FINAL EVALUATION COMPLETE',flush=True)
