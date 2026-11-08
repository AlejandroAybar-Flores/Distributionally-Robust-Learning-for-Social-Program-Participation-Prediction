
from __future__ import annotations
import os,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm,spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from estudio_core import ROOT, RES, DATA,DEPARTMENTS,PSET,MODEL_NAMES,NUMERIC,CATEGORICAL,FEATURES,WeightedAssetPCA,SEED,load_data,get_subset,wmetrics,wlogloss,weighted_mean

OUT=RES
B=1200
rng=np.random.default_rng(SEED+100)
COMPARE={'CatBoost - logistica':('catboost','logistica'),'CatBoost - ERM':('catboost','mlp_erm'),
         'Group DRO - ERM':('mlp_groupdro','mlp_erm'),'Equilibrado - ERM':('mlp_equilibrado','mlp_erm')}


def ece(y,p,w,bins=10):
 p=np.asarray(p,float);y=np.asarray(y,float);w=np.asarray(w,float)
 idx=np.argsort(p,kind='stable');ws=w[idx];ps=p[idx];ys=y[idx]
 sub=np.minimum(np.floor((np.cumsum(ws)-ws/2)/sum(ws)*bins).astype(int),bins-1)
 return sum((ws[sub==k].sum()/ws.sum())*abs(np.average(ps[sub==k],weights=ws[sub==k])-
        np.average(ys[sub==k],weights=ws[sub==k])) for k in range(bins) if np.any(sub==k))


def slope_cal(y,p,w):
 z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)).reshape(-1,1)
 if np.std(z)<1e-5:return np.nan,np.nan
 try:
  m=LogisticRegression(C=10000.,max_iter=500,solver='lbfgs')
  m.fit(z,y,sample_weight=np.asarray(w,float)/np.mean(w));return float(m.intercept_[0]),float(m.coef_[0][0])
 except Exception:return np.nan,np.nan


def metric_tables():
 nat=[];dep=[];sub=[];byseed=[];bydep_probs=[]
 for prog in PSET:
  a=pd.read_csv(RES/f'predicciones_{prog.lower()}.csv',dtype={'departamento':str,'PSU':str})
  y=a.y.to_numpy(int);w=a.factor_hogar.to_numpy(float)
  for m in MODEL_NAMES:
   p=a[f'p_{m}'].to_numpy(float)
   metrics=wmetrics(y,p,w)
   inter,slope=slope_cal(y,p,w) if m!='prevalencia_fuente' else (np.nan,np.nan)
   table={'programa':prog,'modelo':m,'n':len(a),'beneficiarios':int(y.sum()),'ece10':ece(y,p,w),'cal_intercepto':inter,'cal_pendiente':slope,**metrics}
   table['macro_logloss']=np.mean([wlogloss(d.y,d[f'p_{m}'],d.factor_hogar) for _,d in a.groupby('departamento')])
   table['macro_auc']=np.mean([wmetrics(d.y,d[f'p_{m}'],d.factor_hogar)['auc'] for _,d in a.groupby('departamento')])
   table['peor_logloss']=max([wlogloss(d.y,d[f'p_{m}'],d.factor_hogar) for _,d in a.groupby('departamento')])
   nat.append(table)
   for d,dd in a.groupby('departamento',sort=True):
    v=wmetrics(dd.y,dd[f'p_{m}'],dd.factor_hogar)
    entropy=-(v['p_observada']*np.log(max(v['p_observada'],1e-7))+(1-v['p_observada'])*np.log(max(1-v['p_observada'],1e-7)))
    dep.append({'programa':prog,'departamento':d,'nombre':DEPARTMENTS[d],'modelo':m,'n':len(dd),'beneficiarios':int(dd.y.sum()),'n_psu':dd.PSU.nunique(),
      'n_efectivo_kish':dd.factor_hogar.sum()**2/np.square(dd.factor_hogar).sum(),
      'logloss_rel_entropia':v['logloss']/entropy if entropy>0 else np.nan,**v})
   for field in ('area','region_natural','n_adultos_65mas'):
    groups=a.groupby(field)
    if field=='n_adultos_65mas': groups=a.groupby(pd.cut(a[field],bins=[-.5,.5,1.5,100],labels=['0','1','2+']))
    for key,dd in groups:
     if len(dd)<70:continue
     sub.append({'programa':prog,'modelo':m,'subgrupo':field,'categoria':str(key),'n':len(dd),'beneficiarios':int(dd.y.sum()),**wmetrics(dd.y,dd[f'p_{m}'],dd.factor_hogar)})
  for m in ('mlp_erm','mlp_equilibrado','mlp_groupdro'):
   for s in [13,31,59]:
    byseed.append({'programa':prog,'modelo':m,'semilla':s,**wmetrics(y,a[f'p_{m}_seed{s}'],w)})
  for d,dd in a.groupby('departamento'):
   bydep_probs.append({'programa':prog,'departamento':d,'nombre':DEPARTMENTS[d], 'n':len(dd),'positivos':int(dd.y.sum()),
    'prevalencia_ponderada':weighted_mean(dd.y,dd.factor_hogar),'n_psu':dd.PSU.nunique(),
    'peso_total':dd.factor_hogar.sum()})
 for filename,data in [('metricas_nacionales.csv',nat),('metricas_departamentales.csv',dep),
                       ('metricas_subgrupos.csv',sub),('metricas_semillas.csv',byseed),('descriptivos_departamentos.csv',bydep_probs)]:
  pd.DataFrame(data).to_csv(OUT/filename,index=False)
 return pd.DataFrame(nat),pd.DataFrame(dep)


def prepare_cluster_arrays(a):
 d={}; col_names=MODEL_NAMES
 for dep,g in a.groupby('departamento',sort=True):
  psus=sorted(g.PSU.unique());pid=pd.Categorical(g.PSU,categories=psus).codes;n=len(psus)
  w=g.factor_hogar.to_numpy(float);y=g.y.to_numpy(float)
  awei=np.bincount(pid,weights=w,minlength=n)
  arr=[awei,np.bincount(pid,weights=w*y,minlength=n)]
  for m in col_names:
   p=np.clip(g[f'p_{m}'].to_numpy(float),1e-7,1-1e-7)
   arr.append(np.bincount(pid,weights=w*(-y*np.log(p)-(1-y)*np.log1p(-p)),minlength=n))
   arr.append(np.bincount(pid,weights=w*(p-y),minlength=n))
   arr.append(np.bincount(pid,weights=w*((p-y)**2),minlength=n))
  d[dep]=np.column_stack(arr)
 return d


def bootstraps():
 stats=[];deptcal=[];deptworst=[];rawout=[]
 for prog in PSET:
  a=pd.read_csv(OUT/f'predicciones_{prog.lower()}.csv',dtype={'departamento':str,'PSU':str})
  cluster=prepare_cluster_arrays(a);departments=sorted(cluster)
  nat_tot=np.sum([M.sum(axis=0) for M in cluster.values()],axis=0)
  W0=nat_tot[0]
  obs={m:nat_tot[2+3*j]/W0 for j,m in enumerate(MODEL_NAMES)}
  d_obs={dep:{m:cluster[dep][:,2+3*j].sum()/cluster[dep][:,0].sum() for j,m in enumerate(MODEL_NAMES)} for dep in departments}
  R=np.zeros((B,len(nat_tot)));dep_risks=np.zeros((B,len(departments),len(MODEL_NAMES)))
  dcerr=np.zeros((B,len(departments)))
  for b in range(B):
   for di,dep in enumerate(departments):
    a0=cluster[dep];n=len(a0)
    mult=rng.multinomial(n,np.ones(n)/n)
    t=mult@a0
    R[b]+=t
    dep_risks[b,di,:]=[t[2+3*j]/t[0] for j in range(len(MODEL_NAMES))]
    dcerr[b,di]=t[2+3*MODEL_NAMES.index('catboost')+1]/t[0]
  W=R[:,0]
  bd={m:R[:,2+3*j]/W for j,m in enumerate(MODEL_NAMES)}
  for label,(left,right) in COMPARE.items():
   val=obs[left]-obs[right]
   distribution=bd[left]-bd[right]
   se=float(np.std(distribution,ddof=1));z=val/se if se>0 else np.nan
   rec={'programa':prog,'contraste':label,'diferencia':val,'se_bootstrap':se,
        'ic95_inf':np.quantile(distribution,.025),'ic95_sup':np.quantile(distribution,.975),
        'p_normal_aprox':2*norm.sf(abs(z)) if se>0 else np.nan,
        'n_bootstrap':B,'inferencia':'condicional en predicciones OOF fijas; PSU aproximadas'}
   stats.append(rec)
  for j,m in enumerate(MODEL_NAMES):
   risk=dep_risks[:,:,j].max(axis=1)
   deptworst.append({'programa':prog,'modelo':m,'peor_departamento_observado':max(d_obs[d][m] for d in departments),
      'peor_departamento_mediana_boot':np.median(risk),'peor_departamento_ic95_inf':np.quantile(risk,.025),
      'peor_departamento_ic95_sup':np.quantile(risk,.975)})
  for di,dep in enumerate(departments):
   dd=a[a.departamento==dep]
   actual=weighted_mean(dd.y,dd.factor_hogar);pobs=weighted_mean(dd.p_catboost,dd.factor_hogar)
   diff=pobs-actual;dist=dcerr[:,di]
   se=np.std(dist,ddof=1); z=diff/se if se>0 else np.nan
   deptcal.append({'programa':prog,'departamento':dep,'nombre':DEPARTMENTS[dep],
   'n':len(dd),'n_psu':dd.PSU.nunique(),'participacion_observada':actual,'prediccion_media':pobs,
   'sesgo_calibracion':diff,'ic95_inf':np.quantile(dist,.025),'ic95_sup':np.quantile(dist,.975),
   'se_bootstrap':se,'p_normal_aprox':2*norm.sf(abs(z)) if se>0 else np.nan})
  for i in range(B):
   rawout.append({'programa':prog,'replica':i+1,'riesgo_cat':bd['catboost'][i],
      'riesgo_logistica':bd['logistica'][i],'riesgo_erm':bd['mlp_erm'][i],
      'riesgo_dro':bd['mlp_groupdro'][i]})
 tab=pd.DataFrame(stats);tab['q_bh_ocho_pruebas']=bh(tab.p_normal_aprox.to_numpy())
 tab.to_csv(OUT/'intervalos_contrastes.csv',index=False)
 dt=pd.DataFrame(deptcal);dt['q_bh_50_departamentos']=bh(dt.p_normal_aprox.to_numpy())
 dt.to_csv(OUT/'calibracion_departamentos_ic.csv',index=False)
 pd.DataFrame(deptworst).to_csv(OUT/'peores_riesgos_bootstrap.csv',index=False)
 pd.DataFrame(rawout).to_csv(OUT/'replicas_bootstrap.csv',index=False)
 return tab,dt


def bh(p):
 p=np.asarray(p,float);q=np.full(len(p),np.nan);valid=np.where(np.isfinite(p))[0]
 oo=valid[np.argsort(p[valid])];ll=len(oo)
 if ll==0:return q
 mm=np.minimum.accumulate((p[oo]*ll/np.arange(1,ll+1))[::-1])[::-1]
 q[oo]=np.clip(mm,0,1);return q


def shift_analysis():
 d=load_data(); rows=[];rhos=[];features=NUMERIC
 for prog in PSET:
  s=get_subset(d,prog)
  for dep,test in s.groupby('codigo_departamento',sort=True):
   train=s[s.codigo_departamento!=dep]
   pca=WeightedAssetPCA().fit(train,train.factor_hogar.to_numpy(float))
   X=np.column_stack([train[[f for f in features if f!='indice_activos_pca']].to_numpy(float),pca.transform(train)])
   Z=np.column_stack([test[[f for f in features if f!='indice_activos_pca']].to_numpy(float),pca.transform(test)])
   wt=train.factor_hogar.to_numpy(float);we=test.factor_hogar.to_numpy(float)
   mu=np.average(X,weights=wt,axis=0);md=np.average(Z,weights=we,axis=0)
   var=np.average((X-mu)**2,weights=wt,axis=0);var=np.maximum(var,1e-4)
   smd=np.abs(md-mu)/np.sqrt(var)
   rows.append({'programa':prog,'departamento':dep,'nombre':DEPARTMENTS[dep],
     'shift_smd_rms':float(np.sqrt(np.mean(smd**2))),'shift_smd_max':float(smd.max()),
     'max_variable':features[int(np.argmax(smd))],
     'shift_area_pct':weighted_mean(test.area.astype(float)==2,we)-weighted_mean(train.area.astype(float)==2,wt)})
 S=pd.DataFrame(rows)
 C=pd.read_csv(OUT/'metricas_departamentales.csv',dtype={'departamento':str})
 e=C[C.modelo=='catboost'][['programa','departamento','logloss']].rename(columns={'logloss':'loss_cat'})
 S=S.merge(e,on=['programa','departamento'])
 for prog,grp in S.groupby('programa'):
  s=spearmanr(grp.shift_smd_rms,grp.loss_cat)
  rhos.append({'programa':prog,'rho_spearman_shift_riesgo':s.statistic,'p_exploratorio':s.pvalue,'n_departamentos':len(grp)})
 S.to_csv(OUT/'diagnostico_shift.csv',index=False)
 pd.DataFrame(rhos).to_csv(OUT/'asociacion_shift_error.csv',index=False)
 return S


def make_deciles():
 rows=[]
 for prog in PSET:
  a=pd.read_csv(OUT/f'predicciones_{prog.lower()}.csv')
  for model in MODEL_NAMES[1:]:
   order=np.argsort(a[f'p_{model}'].to_numpy());w=a.factor_hogar.to_numpy()[order];y=a.y.to_numpy()[order];p=a[f'p_{model}'].to_numpy()[order]
   b=np.minimum(np.floor((np.cumsum(w)-w/2)/w.sum()*10).astype(int),9)
   for dec in range(10):
    mask=b==dec
    if mask.sum()==0:continue
    rows.append({'programa':prog,'modelo':model,'decil':dec+1,'n':mask.sum(),
        'p_predicha':weighted_mean(p[mask],w[mask]),'p_observada':weighted_mean(y[mask],w[mask]),
        'masa_expansion':w[mask].sum()})
 pd.DataFrame(rows).to_csv(OUT/'calibracion_deciles.csv',index=False)


def corregir_psu():

    base=load_data()
    mp=dict(zip(base.excel_fila,base.PSU_aproximada))
    for prog in PSET:
        file=OUT/f'predicciones_{prog.lower()}.csv'
        df=pd.read_csv(file,dtype={'departamento':str,'PSU':str})
        df['PSU']=df.excel_fila.map(mp)
        assert df.PSU.notna().all()
        df.to_csv(file,index=False)

def main():
 corregir_psu(); metric_tables(); bootstraps();shift_analysis();make_deciles()
 print('POSTPROCESO COMPLETO',flush=True)
 for f in ['metricas_nacionales.csv','intervalos_contrastes.csv','calibracion_departamentos_ic.csv']:
  print('\n',f,'\n',pd.read_csv(OUT/f).head(12).to_string(index=False))

if __name__=='__main__':main()
