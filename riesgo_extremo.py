
from __future__ import annotations
import numpy as np,pandas as pd
from scipy.stats import norm
from estudio_core import RES,PSET,MODEL_NAMES,SEED

B=1600
rng=np.random.default_rng(SEED+202)
MODELS=['logistica','catboost','mlp_erm','mlp_equilibrado','mlp_groupdro']
COMPARE=[('mlp_groupdro','mlp_erm'),('mlp_equilibrado','mlp_erm'),('catboost','mlp_groupdro'),('catboost','mlp_erm')]


def main():
 overall=[]; reps=[]; macro=[]
 for prog in PSET:
  df=pd.read_csv(RES/f'predicciones_{prog.lower()}.csv',dtype={'departamento':str,'PSU':str})
  deps=[]
  for dep,a in df.groupby('departamento',sort=True):
   groups=pd.Categorical(a.PSU);g=groups.codes;w=a.factor_hogar.to_numpy(float);y=a.y.to_numpy(float)
   sums=[np.bincount(g,weights=w,minlength=len(groups.categories))]
   for m in MODELS:
    p=np.clip(a[f'p_{m}'].to_numpy(float),1e-7,1-1e-7)
    loss=-(y*np.log(p)+(1-y)*np.log1p(-p))
    sums.append(np.bincount(g,weights=w*loss,minlength=len(groups.categories)))
   deps.append((dep,np.column_stack(sums)))
  measured=np.array([[x[:,j+1].sum()/x[:,0].sum() for j in range(len(MODELS))] for _,x in deps])
  samp=np.empty((B,len(deps),len(MODELS)),dtype=float)
  for b in range(B):
   for di,(_,x) in enumerate(deps):
    n=len(x);counts=rng.multinomial(n,np.ones(n)/n)
    z=counts@x;samp[b,di,:]=z[1:]/z[0]
  for left,right in COMPARE:
   il=MODELS.index(left);ir=MODELS.index(right)
   delt=samp[:,:,il].max(axis=1)-samp[:,:,ir].max(axis=1)
   point=measured[:,il].max()-measured[:,ir].max()
   se=np.std(delt,ddof=1)
   overall.append({'programa':prog,'comparacion':f'{left} - {right}','estimando':'max loss departamental',
    'diferencia':point,'se_boot':se,'ic95_inf':np.quantile(delt,.025),'ic95_sup':np.quantile(delt,.975),
    'p_normal_aprox':2*norm.sf(abs(point/se)) if se>0 else np.nan,'B':B})
   for b,v in enumerate(delt):reps.append({'programa':prog,'comparacion':f'{left} - {right}','replica':b+1,'diff_max_risk':v})
  for j,m in enumerate(MODELS):
   p=np.array(measured[:,j]);cvar90=np.mean(np.sort(p)[-5:]);q90=np.quantile(p,.9)
   entropy=pd.read_csv(RES/'metricas_departamentales.csv',dtype={'departamento':str})
   en=entropy[(entropy.programa==prog)&(entropy.modelo==m)]
   macro.append({'programa':prog,'modelo':m,'riesgo_nacional':pd.read_csv(RES/'metricas_nacionales.csv').query('programa == @prog and modelo == @m').logloss.iloc[0],
       'media_dept':p.mean(),'peor_dept':p.max(),'media_5_peores':cvar90,'p90_riesgo_dept':q90,
       'peor_ratio_entropia':en.logloss_rel_entropia.max()})
 pd.DataFrame(overall).to_csv(RES/'contrastes_peor_departamento.csv',index=False)
 pd.DataFrame(reps).to_csv(RES/'bootstrap_peor_riesgo_replicas.csv',index=False)
 pd.DataFrame(macro).to_csv(RES/'frontera_robustez.csv',index=False)
 print(pd.DataFrame(overall).to_string(index=False),flush=True)
 print(pd.DataFrame(macro).to_string(index=False),flush=True)

if __name__=='__main__':main()
