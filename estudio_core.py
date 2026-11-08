





from __future__ import annotations
import os, json, hashlib, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'datos'/'ENAPRES2025_activos_originales.pkl'
RES=ROOT/'resultados'
RES.mkdir(parents=True,exist_ok=True)
SEED=20261106
ASSETS=[f'activo_p189_{i}' for i in range(1,18)]
torch.set_num_threads(2)

DEPARTMENTS={'01':'AMAZONAS','02':'ANCASH','03':'APURIMAC','04':'AREQUIPA','05':'AYACUCHO','06':'CAJAMARCA','07':'CALLAO','08':'CUSCO','09':'HUANCAVELICA','10':'HUANUCO','11':'ICA','12':'JUNIN','13':'LA LIBERTAD','14':'LAMBAYEQUE','15':'LIMA','16':'LORETO','17':'MADRE DE DIOS','18':'MOQUEGUA','19':'PASCO','20':'PIURA','21':'PUNO','22':'SAN MARTIN','23':'TACNA','24':'TUMBES','25':'UCAYALI'}

NUMERIC=['edad_jefe','tamano_hogar','n_ninos_0_11','n_adolescentes_12_17','n_adultos_65mas','n_mujeres','agua_red_publica','saneamiento_red_publica','electricidad_red_publica','hogar_multigeneracional','hogar_con_discapacidad','internet_en_hogar','hogar_con_dispositivo_digital','indice_activos_pca']
NONPCA=[x for x in NUMERIC if x!='indice_activos_pca']
CATEGORICAL=['sexo_jefe','educacion_jefe_4niveles','lengua_materna_jefe','area']
FEATURES=NUMERIC+CATEGORICAL
MODEL_NAMES=['prevalencia_fuente','logistica','catboost','mlp_erm','mlp_equilibrado','mlp_groupdro']
SEEDS=[13,31,59]
PSET={'Pension65':('prog_pension_65','n_adultos_65mas >= 1'), 'Juntos':('prog_juntos','n_ninos_0_11 + n_adolescentes_12_17 >= 1')}


def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda: f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()


def load_data():
 d=pd.read_pickle(DATA)
 for c,w in [('ANIO',4),('MES',2),('codigo_departamento',2)]:d[c]=d[c].str.zfill(w)
 d.insert(0,'excel_fila',np.arange(2,len(d)+2,dtype=int))
 key=['ANIO','MES','codigo_departamento','CONGLOMERADO','VIVIENDA','HOGAR']
 assert len(d)==36184, f'Unexpected base length {len(d)}'
 assert d.duplicated(key).sum()==0,'Duplicate household key'
 assert set(d.codigo_departamento)==set(DEPARTMENTS),'Unknown / absent departments'
 assert np.isfinite(d.factor_hogar).all() and (d.factor_hogar>0).all(),'Invalid household weights'
 assert not d[NONPCA+CATEGORICAL+ASSETS].isna().any().any(),'Missing feature in main prediction set'
 for t,_ in PSET.values():assert set(d[t].unique()).issubset({0,1})
 d['PSU_aproximada']=d.codigo_departamento+'_'+d.CONGLOMERADO
 return d


def get_subset(d,program):
 if program=='Pension65': a=d[d.n_adultos_65mas>=1].copy()
 elif program=='Juntos': a=d[d.n_ninos_0_11+d.n_adolescentes_12_17>=1].copy()
 else:raise ValueError(program)
 return a.reset_index(drop=True)


def weighted_mean(x,w):
 return float(np.average(np.asarray(x,dtype=float),weights=np.asarray(w,dtype=float)))


def wlogloss(y,p,w):
 p=np.clip(np.asarray(p,float),1e-7,1-1e-7);y=np.asarray(y,float)
 return weighted_mean(-y*np.log(p)-(1-y)*np.log1p(-p),w)


def wmetrics(y,p,w):
 y=np.asarray(y,int);p=np.clip(np.asarray(p,float),1e-7,1-1e-7);w=np.asarray(w,float)
 val={'logloss':wlogloss(y,p,w),'brier':weighted_mean((p-y)**2,w),
 'sesgo_calibracion':weighted_mean(p-y,w),'p_observada':weighted_mean(y,w),
 'p_predicha':weighted_mean(p,w),'auc':np.nan,'auprc':np.nan}
 if len(np.unique(y))>1:
  val['auc']=float(roc_auc_score(y,p,sample_weight=w))
  val['auprc']=float(average_precision_score(y,p,sample_weight=w))
 return val


class WeightedAssetPCA:
 def fit(self,data,weights):
  matrix=data[ASSETS].to_numpy(dtype=np.float64)
  w=np.asarray(weights,np.float64)
  self.mean=np.average(matrix,axis=0,weights=w)
  self.sd=np.sqrt(np.average((matrix-self.mean)**2,axis=0,weights=w))
  self.use=np.where(self.sd>1e-10)[0]
  if len(self.use)<2:raise ValueError('Insufficient variance in source asset indicators')
  z=(matrix[:,self.use]-self.mean[self.use])/self.sd[self.use]
  cov=(z*w[:,None]).T@z/w.sum()
  values,vectors=np.linalg.eigh(cov)
  self.loadings=vectors[:,-1]
  score=z@self.loadings
  totals=matrix[:,self.use].sum(axis=1)
  mscore=np.average(score,weights=w)
  if np.average((score-mscore)*(totals-np.average(totals,weights=w)),weights=w)<0:
   self.loadings=-self.loadings
   score=-score
  self.center=np.average(score,weights=w)
  self.scale=np.sqrt(np.average((score-self.center)**2,weights=w))
  if not np.isfinite(self.scale) or self.scale<1e-10:raise ValueError('Degenerate source asset component')
  self.variance_ratio=float(values[-1]/values.sum())
  return self
 def transform(self,data):
  x=data[ASSETS].to_numpy(dtype=np.float64)
  z=(x[:,self.use]-self.mean[self.use])/self.sd[self.use]
  return (z@self.loadings-self.center)/self.scale

class Encoding:
 def fit(self,train,weights):
  self.pca=WeightedAssetPCA().fit(train,weights)
  x=np.column_stack([train[NONPCA].to_numpy(float),self.pca.transform(train)])
  self.mu=np.average(x,axis=0,weights=weights)
  self.sd=np.sqrt(np.average((x-self.mu)**2,axis=0,weights=weights));self.sd[self.sd<1e-6]=1
  self.oh=OneHotEncoder(sparse_output=False,handle_unknown='ignore',dtype=np.float32)
  self.oh.fit(train[CATEGORICAL].astype(str))
  return self
 def transform(self,data):
  x=np.column_stack([data[NONPCA].to_numpy(float),self.pca.transform(data)])
  arr=(x-self.mu)/self.sd
  cat=self.oh.transform(data[CATEGORICAL].astype(str))
  return np.hstack([arr,cat]).astype(np.float32)


def prepare_cat(df,features=None,pca=None):
 if features is None:features=FEATURES
 x=df[[c for c in features if c!='indice_activos_pca']].copy()
 if 'indice_activos_pca' in features:
  if pca is None:raise ValueError('Fit PCA on source households before prediction')
  x['indice_activos_pca']=pca.transform(df)
 for c in CATEGORICAL:
  if c in x:x[c]=x[c].astype(str)
 if 'region_natural' in x:x['region_natural']=x['region_natural'].astype(str)
 return x


def fit_cat(tr,te,target,params,weight=True,extra=None):
 feat=[x for x in FEATURES if x!='indice_activos_pca'] if extra and '_without_pca' in extra else list(FEATURES)
 if extra:feat.extend([x for x in extra if x!='_without_pca' and x not in feat])
 cat=CATEGORICAL+[c for c in ['region_natural'] if c in feat]
 weights=tr.factor_hogar.to_numpy(float)
 weights=weights/weights.mean() if weight else np.ones(len(tr))
 model=CatBoostClassifier(iterations=int(params['iterations']),depth=int(params['depth']),learning_rate=.05,l2_leaf_reg=6,
   loss_function='Logloss',random_seed=SEED,verbose=False,thread_count=2,allow_writing_files=False,
   random_strength=0.5,bootstrap_type='Bernoulli',subsample=0.8)
 pca=WeightedAssetPCA().fit(tr,tr.factor_hogar.to_numpy(float)) if 'indice_activos_pca' in feat else None
 model.fit(prepare_cat(tr,feat,pca),tr[target].to_numpy(int),cat_features=cat,sample_weight=weights)
 return np.clip(model.predict_proba(prepare_cat(te,feat,pca))[:,1],1e-6,1-1e-6)


class MLP(nn.Module):
 def __init__(self,p):
  super().__init__()
  self.layers=nn.Sequential(nn.Linear(p,48),nn.ReLU(),nn.Linear(48,24),nn.ReLU(),nn.Linear(24,1))
 def forward(self,x):return self.layers(x).ravel()


def fit_nn(x,y,w,group,xt,mode='erm',eta=.08,wd=.01,seed=13,epochs=95):
 torch.manual_seed(int(seed));torch.use_deterministic_algorithms(True, warn_only=True)
 X=torch.from_numpy(np.asarray(x,np.float32));Y=torch.from_numpy(np.asarray(y,np.float32))
 W=torch.from_numpy(np.asarray(w/w.mean(),np.float32)); Xte=torch.from_numpy(np.asarray(xt,np.float32))
 _,ii=np.unique(group,return_inverse=True); G=torch.tensor(ii,dtype=torch.int64)
 ngrp=int(G.max())+1;net=MLP(x.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=.004,weight_decay=wd)
 q=torch.ones(ngrp)/ngrp
 for ep in range(epochs):
  opt.zero_grad(set_to_none=True)
  losses=F.binary_cross_entropy_with_logits(net(X),Y,reduction='none')
  if mode=='erm':risk=(losses*W).sum()/W.sum()
  else:
   a=torch.zeros(ngrp).scatter_add_(0,G,losses*W)
   b=torch.zeros(ngrp).scatter_add_(0,G,W)
   group_risk=a/b.clamp_min(1e-8)
   if mode=='balanced':risk=group_risk.mean()
   elif mode=='dro':
    with torch.no_grad():
     q=q*torch.exp(eta*(group_risk.detach()-group_risk.detach().max()))
     q=q/q.sum()
    risk=(q.detach()*group_risk).sum()
   else:raise ValueError(mode)
  risk.backward();nn.utils.clip_grad_norm_(net.parameters(),5.0);opt.step()
 net.eval()
 with torch.no_grad():pred=torch.sigmoid(net(Xte)).cpu().numpy()
 return np.clip(pred,1e-6,1-1e-6)


def grouped_inner_splits(train):
 return list(GroupKFold(n_splits=2).split(train,y=train['dummy_y'] if 'dummy_y' in train else None,groups=train.codigo_departamento))


def tune_options(train,target,method,options,seed):

 groups=train.codigo_departamento.to_numpy(str)
 splits=list(GroupKFold(n_splits=2).split(train,groups=groups))
 out=[]
 for option in options:
  ll_sum=0.;mass=0.
  for vi,(i,j) in enumerate(splits):
   tr=train.iloc[i];val=train.iloc[j]
   y=val[target].to_numpy(int);wv=val.factor_hogar.to_numpy(float)
   if method=='lr':
    enc=Encoding().fit(tr,tr.factor_hogar.to_numpy(float))
    A=enc.transform(tr);B=enc.transform(val)
    ww=tr.factor_hogar.to_numpy(float);ww/=ww.mean()
    lr=LogisticRegression(C=float(option),solver='lbfgs',max_iter=600,random_state=seed)
    lr.fit(A,tr[target].to_numpy(int),sample_weight=ww)
    p=lr.predict_proba(B)[:,1]
   elif method=='cat':p=fit_cat(tr,val,target,option)
   elif method in ('erm','dro'):
    enc=Encoding().fit(tr,tr.factor_hogar.to_numpy(float));A=enc.transform(tr);B=enc.transform(val)
    p=fit_nn(A,tr[target].to_numpy(int),tr.factor_hogar.to_numpy(float),tr.codigo_departamento.to_numpy(str),B,
       mode=method,eta=option.get('eta',.08),wd=option['wd'],seed=seed,epochs=95)
   else:raise ValueError(method)
   pp=np.clip(p,1e-7,1-1e-7)
   ll_sum+=np.sum(wv*(-y*np.log(pp)-(1-y)*np.log1p(-pp)))
   mass+=wv.sum()
  out.append((ll_sum/mass,option))
 best=sorted(out,key=lambda x:x[0])[0]
 return best[1],[{'option':str(opt),'inner_logloss':float(risk)} for risk,opt in out]


def run_program(d,program,force=False):
 target=PSET[program][0]; df=get_subset(d,program); y=df[target].to_numpy(int)
 path=RES/f'predicciones_{program.lower()}.csv'
 tuning_path=RES/f'hiperparametros_{program.lower()}.csv'
 if path.exists() and not force:
  old=pd.read_csv(path,dtype={'departamento':str});print('CACHE',program,len(old),flush=True)
  if len(old)==len(df) and set(old.excel_fila)==set(df.excel_fila) and old[[f'p_{m}' for m in MODEL_NAMES]].notna().all().all():return old
  else:raise RuntimeError('Partial invalid cache. Use --force.')
 P={m:np.full(len(df),np.nan) for m in MODEL_NAMES}
 seed_oof={m:{sd:np.full(len(df),np.nan) for sd in SEEDS} for m in ('mlp_erm','mlp_equilibrado','mlp_groupdro')}
 traces=[]; start=time.time()
 for k,dep in enumerate(sorted(DEPARTMENTS)):
  test=(df.codigo_departamento==dep).to_numpy();te=df.iloc[np.flatnonzero(test)];tr=df.iloc[np.flatnonzero(~test)]
  ytr=tr[target].to_numpy(int);wt=tr.factor_hogar.to_numpy(float)
  P['prevalencia_fuente'][test]=weighted_mean(ytr,wt)
  lrc,lr_trace=tune_options(tr,target,'lr',[0.1,1,10],SEED+k)
  cb,cb_trace=tune_options(tr,target,'cat',[{'depth':4,'iterations':240},{'depth':6,'iterations':240}],SEED+k)
  wd,nn_trace=tune_options(tr,target,'erm',[{'wd':.002},{'wd':.02}],SEED+k)
                                                                                  
  eta,dro_trace=tune_options(tr,target,'dro',[{'wd':wd['wd'],'eta':.04},{'wd':wd['wd'],'eta':.12}],SEED+k)
  for tp,vals in [('logistica',lr_trace),('catboost',cb_trace),('mlp_erm',nn_trace),('mlp_groupdro',dro_trace)]:
   for x in vals:traces.append({'programa':program,'departamento':dep,'modelo':tp,**x})
  enc=Encoding().fit(tr,wt);X=enc.transform(tr);Z=enc.transform(te)
  lr=LogisticRegression(C=float(lrc),solver='lbfgs',max_iter=650,random_state=SEED+k)
  lr.fit(X,ytr,sample_weight=wt/wt.mean());P['logistica'][test]=lr.predict_proba(Z)[:,1]
  P['catboost'][test]=fit_cat(tr,te,target,cb)
  for model,mode in [('mlp_erm','erm'),('mlp_equilibrado','balanced'),('mlp_groupdro','dro')]:
   for s in SEEDS:
    p=fit_nn(X,ytr,wt,tr.codigo_departamento.to_numpy(str),Z,
       mode=mode,eta=eta['eta'],wd=wd['wd'],seed=s+97*k,epochs=95)
    seed_oof[model][s][test]=p
   P[model][test]=np.mean([seed_oof[model][s][test] for s in SEEDS],axis=0)
  print(f'{program} / held-out {dep} {DEPARTMENTS[dep]:15s} [{k+1:02d}/25] {time.time()-start:7.1f}s C={lrc} Cat={cb} WD={wd} eta={eta}',flush=True)
 out=pd.DataFrame({'excel_fila':df.excel_fila,'departamento':df.codigo_departamento,'nombre_departamento':df.codigo_departamento.map(DEPARTMENTS),
  'PSU':df.PSU_aproximada,'factor_hogar':df.factor_hogar,'y':y,'area':df.area.astype(str),
  'region_natural':df.region_natural,'n_adultos_65mas':df.n_adultos_65mas,'n_ninos_0_11':df.n_ninos_0_11,'n_adolescentes_12_17':df.n_adolescentes_12_17})
 for model in MODEL_NAMES:
  assert np.isfinite(P[model]).all(),model
  out[f'p_{model}']=np.clip(P[model],1e-7,1-1e-7)
 for model in seed_oof:
  for s in SEEDS:out[f'p_{model}_seed{s}']=seed_oof[model][s]
 out.to_csv(path,index=False)
 pd.DataFrame(traces).to_csv(tuning_path,index=False)
 print('COMPLETADO',program,'->',path,flush=True)
 return out


def run_all(force=False):
 print('START',time.strftime('%Y-%m-%d %H:%M:%S'),'input:',DATA,'SHA256',sha256(DATA),flush=True)
 data=load_data()
 counts=[]
 for program,(target,filter_rule) in PSET.items():
  sub=get_subset(data,program);w=sub.factor_hogar.to_numpy(float)
  counts.append({'programa':program,'regla_muestra':filter_rule,'n_hogares':len(sub),'beneficiarios_n':int(sub[target].sum()),
   'prevalencia_ponderada':weighted_mean(sub[target],w),'suma_factores':w.sum(),'n_departamentos':sub.codigo_departamento.nunique(),
   'n_psu_aprox':sub.PSU_aproximada.nunique(),'neff_kish':w.sum()**2/np.sum(w*w),
   'faltantes_indigena':int(sub.jefe_pertenece_pueblo_indigena.isna().sum())})
 pd.DataFrame(counts).to_csv(RES/'muestra_principal.csv',index=False)
 audit=pd.DataFrame({'variable':data.columns,'faltantes':[data[c].isna().sum() for c in data.columns],
  'n_valores':[data[c].nunique(dropna=True) for c in data.columns],
  'tipo':[str(data[c].dtype) for c in data.columns]})
 audit.to_csv(RES/'auditoria_variables.csv',index=False)
 for program in PSET:run_program(data,program,force=force)
 manifest={'fecha_ejecucion':time.strftime('%Y-%m-%d %H:%M:%S'), 'datos_sha256':sha256(DATA),'n_fuente':len(data),
  'poblaciones':{k:v[1] for k,v in PSET.items()},'modelos':MODEL_NAMES,
  'numericas':NUMERIC,'categoricas':CATEGORICAL,'excluidas_del_principal':['n_integrantes_con_discapacidad (redundante)','jefe_pertenece_pueblo_indigena (faltantes)','codigo_departamento (identificador del dominio objetivo)','region_natural (sensibilidad)','variables de otros programas (posible fuga/uso operativo)'],
  'semillas_ensemble':SEEDS,'inner_cv':'2-fold GroupKFold por departamento de origen; selección sólo en entrenamiento',
  'asset_pca':'17 ownership variables; survey-weighted source-only PCA refit in every inner and outer training partition; sign oriented to ownership count; source-score standardized',
  'outer_cv':'LODO 25 departamentos','nn':'MLP 48-24, AdamW lr=0.004, 95 epocas; 3 semillas; regularizacion via inner source CV',
  'group_dro':'Exponentiated gradient (eta de 0.04 o 0.12, seleccion en inner source CV); riesgos por departamento normalizados por suma de pesos',
  'pesos':'factor_hogar positivo normalizado a media 1 en entrenamiento; evaluacion con pesos originales',
  'conglomerado':'codigo_departamento+CONGLOMERADO; aproximacion; estratos oficiales no identificados',
  'python':platform.python_version(),'torch':torch.__version__,'catboost':__import__('catboost').__version__,'sklearn':__import__('sklearn').__version__}
 with open(RES/'configuracion.json','w',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
 print('TRAINING COMPLETE',flush=True)
 return manifest

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true');args=parser.parse_args()
 run_all(force=args.force)
