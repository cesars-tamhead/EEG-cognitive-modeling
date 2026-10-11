"""One-command regeneration from the original CSV or cleaned-data ZIP."""
import argparse,json,subprocess,sys
from pathlib import Path
from threadpoolctl import threadpool_limits
from eeg_pipeline import read_csv,clean_data,extract_features
from optimize_eeg_v2 import extended_features
from improve_original import digest
p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',default='results');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);source=Path(a.csv).resolve();cache=out/'v2_features';cache.mkdir(exist_ok=True)
with threadpool_limits(limits=2):
 base,_=extract_features(clean_data(read_csv(source)),cache)
 ext=extended_features(source,cache)
identity={'raw_sha256':digest(source),'baseline_features_sha256':digest(cache/'channel_features.csv'),'extraction_version':2};(cache/'feature_cache_identity.json').write_text(json.dumps(identity,indent=2))
subprocess.run([sys.executable,str(Path(__file__).with_name('improve_original.py')),'--csv',str(source),'--baseline-features',str(cache/'channel_features.csv'),'--extended-features',str(cache/'extended_channel_features.csv'),'--out',str(out)],check=True)
