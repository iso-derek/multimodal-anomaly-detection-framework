from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.robustness_research import synthetic_cases,run_study
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path);args=parser.parse_args()
frame=pd.read_csv(args.input) if args.input else synthetic_cases()
metrics,predictions,metadata=run_study(frame)
out=ROOT/'outputs';out.mkdir(exist_ok=True)
for name,df in [('cases',frame),('metrics',metrics),('predictions',predictions)]:df.to_csv(out/f'{name}.csv',index=False)
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
print(metrics.to_string(index=False));print(json.dumps(metadata,indent=2))
