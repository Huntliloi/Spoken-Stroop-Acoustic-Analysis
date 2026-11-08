from pathlib import Path
import sys
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'scripts/04_age_task_lme'))
import fixed_effect_sensitivity as models

def main():
 out=models.ROOT/'08_traditional_performance';out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(P/'data/public/healthy_traditional.csv')
 summary=d.groupby('Task').agg(N=('Subject_ID','size'),Accuracy_mean=('Accuracy','mean'),Accuracy_SD=('Accuracy','std'),Time_mean=('Completion_Time_sec','mean'),Time_SD=('Completion_Time_sec','std'))
 summary.to_csv(out/'traditional_task_summary.csv')
 wide=d.pivot(index='Subject_ID',columns='Task',values='Accuracy')
 print('Zero W/CW accuracy difference:',int(np.isclose(wide.Hanzi-wide.STT,0,rtol=0,atol=1e-12).sum()))
 data,people,features=models.load_data()
 means=models.subject_task_means(models.standardize_acoustic(data,features),features)
 shifts,differences=models.bootstrap_gradient(means,people,features)
 shifts.to_csv(out/'task_shift_magnitude.csv',index=False)
 differences.to_csv(out/'task_shift_pairwise_differences.csv',index=False)

if __name__=='__main__':main()
