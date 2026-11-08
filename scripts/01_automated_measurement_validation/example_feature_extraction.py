"""Extract first-response acoustics and traditional scores from full token times."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import opensmile
import soundfile as sf
from alignment_rules import map_items,accuracy,normalized_colors

def main():
 p=argparse.ArgumentParser()
 package=Path(__file__).resolve().parents[2]
 p.add_argument('--tokens',type=Path,default=package/'data/example/token_results.csv')
 p.add_argument('--audio-root',type=Path,default=package/'data/example/audio')
 p.add_argument('--output-dir',type=Path,default=package/'results/recomputed/example_features')
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 t=pd.read_csv(a.tokens,dtype={'subject_id':str})
 smile=opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.Functionals)
 rows=[];scores=[]
 tasks={'hanzi':'Hanzi','dots':'Dots','stt':'STT','Hanzi':'Hanzi','Dots':'Dots','STT':'STT','W':'Hanzi','C':'Dots','CW':'STT'}
 for (sid,rawtask),group in t.groupby(['subject_id','task']):
  task=tasks[rawtask];g=group.sort_values('token_index').reset_index(drop=True)
  words=normalized_colors(g.token_text.astype(str).tolist())
  if len(words)!=len(g):raise ValueError('Expected normalized color-word tokens')
  if len(words)>12:mapping,_=map_items(words,task)
  else:
   from alignment_rules import ItemMapping,TARGETS
   mapping=[ItemMapping(i+1,TARGETS[task][i],i,i,word,word,word==TARGETS[task][i],'single_response') for i,word in enumerate(words)]
  candidates=[x for x in a.audio_root.rglob('*.wav') if x.name.startswith(str(sid)+'_') and x.stem.lower().endswith('_'+rawtask.lower())]
  if len(candidates)!=1:raise ValueError(f'Expected one example audio for {sid}/{task}')
  signal,sr=sf.read(candidates[0]);signal=signal.mean(axis=1) if signal.ndim==2 else signal
  previous=None
  for item in mapping:
   if item.first_token_index is None:continue
   first=g.iloc[item.first_token_index];last=g.iloc[item.final_token_index]
   start,end=float(first.start_time),float(first.end_time)
   if end<=start:continue
   row={'Subject_ID':sid,'Task':task,'Item_Index':item.item_index,'Absolute_Word_Start':start,'Absolute_Word_End':end,'Duration':end-start,'Gap_Before':start if previous is None else start-previous,'Auto_Word':item.first_word}
   previous=float(last.end_time)
   try:row.update(smile.process_signal(signal[round(start*sr):round(end*sr)].astype(np.float32),sr).iloc[0].to_dict())
   except Exception:row.update({f:np.nan for f in smile.feature_names})
   rows.append(row)
  n,rate=accuracy(mapping)
  scores.append({'Subject_ID':sid,'Task':task,'Correct_Item_Count':n,'Accuracy':rate,'Completion_Time_sec':float(g.end_time.iloc[-1])-float(g.start_time.iloc[0])})
 pd.DataFrame(rows).to_csv(a.output_dir/'acoustic_90_features.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(scores).to_csv(a.output_dir/'traditional_features.csv',index=False)

if __name__=='__main__':main()
