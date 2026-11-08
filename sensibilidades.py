
from __future__ import annotations
from pathlib import Path
import pandas as pd,numpy as np,time,sys
from estudio_core import ROOT,RES,DATA,DEPARTMENTS,PSET,SEED,MODEL_NAMES,load_data,get_subset,fit_cat,fit_nn,Encoding,wmetrics,weighted_mean

CFG={'depth':4,'iterations':240}


def fixed_lodo(df,program,label,cat_extra=None,weighted=True):
 target=PSET[program][0]
 output=RES/f'sens_{program.lower()}_{label}.csv'
 if output.exists():
  old=pd.read_csv(output,dtype={'departamento':str})
  if len(old)==len(df):return old
 dep=df.codigo_departamento.to_numpy(str)
 ps=np.zeros(len(df),dtype=float)
 start=time.time()
 for k,d in enumerate(sorted(np.unique(dep))):
  te=dep==d;tr=df.iloc[np.flatnonzero(~te)];va=df.iloc[np.flatnonzero(te)]
  ps[te]=fit_cat(tr,va,target,CFG,weight=weighted,extra=cat_extra)
  if (k+1)%10==0:print(f'SENS {program}/{label} {k+1}/{len(np.unique(dep))} {time.time()-start:.1f}s',flush=True)
 out=pd.DataFrame({'excel_fila':df.excel_fila,'departamento':df.codigo_departamento,'PSU':df.PSU_aproximada,
   'factor_hogar':df.factor_hogar,'y':df[target],'p_catboost':ps})
 out.to_csv(output,index=False);return out


def missingness_report(data):
 records=[]
 for prog in PSET:
  s=get_subset(data,prog);s['missing']=s.jefe_pertenece_pueblo_indigena.isna().astype(int)
  for (d,miss),g in s.groupby(['codigo_departamento','missing']):
   records.append({'programa':prog,'departamento':d,'missing_indigena':miss,'n':len(g),
    'positivos':int(g[PSET[prog][0]].sum()),'prevalencia_ponderada':weighted_mean(g[PSET[prog][0]],g.factor_hogar)})
 pd.DataFrame(records).to_csv(RES/'faltantes_por_departamento.csv',index=False)


def extra_experiments():
 data=load_data(); missingness_report(data)
 rows=[]
 for prog in PSET:
  s=get_subset(data,prog)
                                                          
  for name,extra,weighted in [('base_fija',None,True),('sin_pca', ['_without_pca'],True),
             ('sin_pesos_entrenamiento',None,False),('region_natural', ['region_natural'],True)]:
   out=fixed_lodo(s,prog,name,cat_extra=extra,weighted=weighted)
   rows.append({'programa':prog,'escenario':name,'n':len(out),'beneficiarios':int(out.y.sum()),
      **wmetrics(out.y,out.p_catboost,out.factor_hogar)})
                                                                           
  cc=s[s.jefe_pertenece_pueblo_indigena.notna()].copy().reset_index(drop=True)
  for name,extra in [('cc_sin_indigena',None),('cc_con_indigena',['jefe_pertenece_pueblo_indigena'])]:
   out=fixed_lodo(cc,prog,name,cat_extra=extra)
   rows.append({'programa':prog,'escenario':name,'n':len(out),'beneficiarios':int(out.y.sum()),
      **wmetrics(out.y,out.p_catboost,out.factor_hogar)})
  if prog=='Juntos':
   out=fixed_lodo(data,'Juntos','todos_hogares')
   rows.append({'programa':'Juntos','escenario':'todos_hogares','n':len(out),'beneficiarios':int(out.y.sum()),
       **wmetrics(out.y,out.p_catboost,out.factor_hogar)})
 pd.DataFrame(rows).to_csv(RES/'sensibilidad_resumen.csv',index=False)
 return pd.DataFrame(rows)


def iid_group_validation():
 data=load_data();out=[]; rows=[];rng=np.random.default_rng(SEED+1000)
 for prog in PSET:
  df=get_subset(data,prog);target=PSET[prog][0]
  folds=np.zeros(len(df),dtype=int)
  for dep,ids in df.groupby('codigo_departamento').groups.items():
   group=df.loc[ids,'PSU_aproximada'].unique();rng.shuffle(group)
   mapper={psu:i%5 for i,psu in enumerate(group)}
   folds[ids]=df.loc[ids,'PSU_aproximada'].map(mapper).to_numpy(int)
  r=pd.DataFrame({'programa':prog,'excel_fila':df.excel_fila,'fold':folds,'PSU':df.PSU_aproximada,
  'factor_hogar':df.factor_hogar,'y':df[target]})
  for x in ['prevalencia_fuente','logistica','catboost','mlp_erm','mlp_groupdro']:r[f'p_{x}']=np.nan
  from sklearn.linear_model import LogisticRegression
  for fold in range(5):
   tr=df.iloc[np.flatnonzero(folds!=fold)];te=df.iloc[np.flatnonzero(folds==fold)]
   ww=tr.factor_hogar.to_numpy(float);tt=tr[target].to_numpy(int)
   r.loc[folds==fold,'p_prevalencia_fuente']=weighted_mean(tt,ww)
   enc=Encoding().fit(tr,ww);x=enc.transform(tr);z=enc.transform(te)
   m=LogisticRegression(C=.1,max_iter=600)
   m.fit(x,tt,sample_weight=ww/ww.mean())
   r.loc[folds==fold,'p_logistica']=m.predict_proba(z)[:,1]
   r.loc[folds==fold,'p_catboost']=fit_cat(tr,te,target,CFG)
   for model,mode in [('mlp_erm','erm'),('mlp_groupdro','dro')]:
    r.loc[folds==fold,f'p_{model}']=np.mean([fit_nn(x,tt,ww,tr.codigo_departamento.to_numpy(str),z,
    mode=mode,wd=.02,eta=.04,seed=s+fold*29,epochs=95) for s in (13,31,59)],axis=0)
   print('IID',prog,fold+1,'/5',flush=True)
  for model in ['prevalencia_fuente','logistica','catboost','mlp_erm','mlp_groupdro']:
   rows.append({'programa':prog,'validacion':'5-fold conglomerado estratificado por departamento','modelo':model,
      **wmetrics(r.y,r[f'p_{model}'],r.factor_hogar)})
  out.append(r)
 pd.concat(out,ignore_index=True).to_csv(RES/'predicciones_iid_conglomerados.csv',index=False)
 pd.DataFrame(rows).to_csv(RES/'validacion_iid_resumen.csv',index=False)


def leave_region_out():
 data=load_data(); rows=[]
 for prog in PSET:
  s=get_subset(data,prog);target=PSET[prog][0]
  p=np.zeros(len(s))
  for reg in sorted(s.region_natural.unique()):
   te=s.region_natural==reg
   p[te.to_numpy()]=fit_cat(s[~te],s[te],target,CFG)
  rows.append({'programa':prog,'escenario':'LOO region natural','n':len(s),**wmetrics(s[target],p,s.factor_hogar)})
  for reg in sorted(s.region_natural.unique()):
   te=s.region_natural==reg
   rows.append({'programa':prog,'escenario':f'Region excluida {reg}','n':te.sum(),**wmetrics(s.loc[te,target],p[te.to_numpy()],s.loc[te,'factor_hogar'])})
 pd.DataFrame(rows).to_csv(RES/'validacion_regiones.csv',index=False)


def main():
 t=time.time();extra_experiments();print('EXTRA DONE sec',time.time()-t,flush=True)
 iid_group_validation();print('IID DONE sec',time.time()-t,flush=True)
 leave_region_out();print('SENSIBILIDADES COMPLETAS sec',time.time()-t,flush=True)

if __name__=='__main__':main()
