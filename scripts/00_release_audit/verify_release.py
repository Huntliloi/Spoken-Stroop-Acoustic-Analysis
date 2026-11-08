"""Verify public sample, feature sets, relationships, and file integrity."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd

def main():
 p=Path(__file__).resolve().parents[2];d=p/'data/public'
 h=pd.read_csv(d/'healthy_participants.csv');c=pd.read_csv(d/'cancer_participants.csv')
 assert len(h)==232 and h.Subject_ID.nunique()==232
 assert len(c)==92 and c.Subject_ID.nunique()==92
 selected=json.loads((d/'selected_features.json').read_text())
 assert len(selected)==49 and len(set(selected))==49
 for co,n in [('hc158',158),('hc74',74)]:
  raw=pd.read_csv(d/f'{co}_item_raw.csv');trans=pd.read_csv(d/f'{co}_item_retained_transformed.csv')
  assert raw.Subject_ID.nunique()==n and set(raw.Subject_ID)<=set(h.Subject_ID)
  assert raw.shape[1]==95 and trans.shape[1]==54
  assert list(trans.columns[5:])==selected
  assert set(trans.Subject_ID)==set(raw.Subject_ID)
 assert set(pd.read_csv(d/'cancer_item_raw.csv').Subject_ID)==set(c.Subject_ID)
 trad=pd.read_csv(d/'healthy_traditional.csv')
 assert len(trad)==696 and not trad.duplicated(['Subject_ID','Task']).any()
 wide=trad.pivot(index='Subject_ID',columns='Task',values='Accuracy')
 assert int(np.isclose(wide.Hanzi-wide.STT,0,rtol=0,atol=1e-12).sum())==144
 base=pd.read_csv(d/'cancer_model_input.csv')
 assert len(base)==92 and not base.isna().any().any()
 q=pd.read_csv(p/'results/06_cancer_fact_cog/adjusted_single_score_associations.csv')
 assert int(((q.Adjustment=='D+T')&(q.HC3_q6<.05)).sum())==0
 for line in (p/'MANIFEST.sha256').read_text().splitlines():
  expected,name=line.split('  ',1);file=p/name
  assert file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==expected,name
 print('PASS: HC232, cancer92, 90/60/49 feature sets, 144 zero accuracy differences, and checksums')

if __name__=='__main__':main()
