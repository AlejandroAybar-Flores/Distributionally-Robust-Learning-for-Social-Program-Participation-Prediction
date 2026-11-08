
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from estudio_core import ROOT,RES,DEPARTMENTS,MODEL_NAMES,PSET

F=ROOT/'figuras';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelsize':10,
    'figure.dpi':120,'savefig.dpi':240,'axes.grid':True,'grid.alpha':.14,'axes.spines.top':False,'axes.spines.right':False})
LABEL={'prevalencia_fuente':'Prevalencia fuente','logistica':'Logística L2','catboost':'CatBoost',
       'mlp_erm':'MLP · ERM','mlp_equilibrado':'MLP · equilibrado','mlp_groupdro':'MLP · Group DRO'}


def save(name):
 plt.tight_layout(pad=1.35)
 plt.savefig(F/f'{name}.png',bbox_inches='tight')
 plt.savefig(F/f'{name}.pdf',bbox_inches='tight')
 plt.close()


def bar_summary(prog):
 d=pd.read_csv(RES/'metricas_nacionales.csv');d=d[d.programa==prog].set_index('modelo').loc[MODEL_NAMES]
 fig,ax=plt.subplots(figsize=(9,4.5))
 x=np.arange(len(d));val=d.logloss.to_numpy()
 bars=ax.bar(x,val,edgecolor='white')
 ax.set_xticks(x,[LABEL[m] for m in d.index],rotation=25,ha='right')
 ax.set_ylabel('Log-loss nacional ponderado ↓')
 ax.set_title(f'Comparación de modelos · {prog} · LODO')
 ax.set_ylim(0,max(val)*1.15)
 for i,v in enumerate(val): ax.text(i,v+max(val)*.018,f'{v:.4f}',ha='center',va='bottom',fontsize=10)
 ax.grid(axis='x',visible=False)
 save(f'01_riesgo_{prog.lower()}')


def differences(prog,left,right,name):
 df=pd.read_csv(RES/'metricas_departamentales.csv',dtype={'departamento':str});d=df[(df.programa==prog)&df.modelo.isin([left,right])]
 piv=d.pivot(index='departamento',columns='modelo',values='logloss')
 diff=(piv[left]-piv[right]).sort_values();labels=[DEPARTMENTS[k] for k in diff.index]
 fig,ax=plt.subplots(figsize=(10,8));y=np.arange(len(diff));ax.axvline(0,color='0.35',lw=1)
 ax.scatter(diff,y,s=35)
 for yy,x in zip(y,diff):ax.plot([0,x],[yy,yy],linewidth=1.0,alpha=.7)
 ax.set_yticks(y,labels);ax.set_xlabel(f'Log-loss: {LABEL[left]} − {LABEL[right]} (negativo = mejora del primero)')
 ax.set_title(f'Heterogeneidad departamental · {prog}')
 save(f'{name}_{prog.lower()}')


def calibration_plot(prog):
 d=pd.read_csv(RES/'calibracion_departamentos_ic.csv',dtype={'departamento':str});d=d[d.programa==prog]
 fig,ax=plt.subplots(figsize=(7,6.5))
 x=d.participacion_observada;y=d.prediccion_media
 ax.scatter(x,y,s=35)
 ax.plot([0,1],[0,1],linestyle='--',lw=1,color='0.4')
 ext=d.assign(delta=(x-y).abs()).nlargest(6,'delta')
 for _,r in ext.iterrows():ax.annotate(r.nombre,(r.participacion_observada,r.prediccion_media),
    xytext=(5,4),textcoords='offset points',fontsize=8)
 ax.set_xlim(0,max(x.max(),y.max())*1.12);ax.set_ylim(0,max(x.max(),y.max())*1.12)
 ax.set_xlabel('Fracción ponderada de hogares con beneficiario')
 ax.set_ylabel('Probabilidad media predicha por CatBoost')
 ax.set_title(f'Calibración por departamento · {prog} · LODO')
 save(f'04_calibracion_{prog.lower()}')


def gap_ci(prog):
 d=pd.read_csv(RES/'calibracion_departamentos_ic.csv',dtype={'departamento':str})
 d=d[d.programa==prog].sort_values('sesgo_calibracion').copy()
 fig,ax=plt.subplots(figsize=(10,8));idx=np.arange(len(d))
 ax.axvline(0,color='0.4',lw=1)
 vals=d.sesgo_calibracion.to_numpy()*100
 left=(d.sesgo_calibracion-d.ic95_inf).to_numpy()*100
 right=(d.ic95_sup-d.sesgo_calibracion).to_numpy()*100
 ax.errorbar(vals,idx,xerr=[left,right],fmt='o',capsize=2,elinewidth=1,markersize=4)
 ax.set_yticks(idx,d.nombre)
 ax.set_xlabel('Predicción media − participación observada (puntos porcentuales)')
 ax.set_title(f'Error de calibración y bootstrap por conglomerados · {prog}')
 ax.text(.99,.02,'Intervalos exploratorios; predicciones fijas; sin estratos oficiales',
    transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
 save(f'05_ic_calibracion_{prog.lower()}')


def reliability(prog):
 d=pd.read_csv(RES/'calibracion_deciles.csv');d=d[d.programa==prog]
 fig,ax=plt.subplots(figsize=(7,6))
 for model in ['logistica','catboost','mlp_erm','mlp_groupdro']:
  a=d[d.modelo==model].sort_values('decil')
  ax.plot(a.p_predicha,a.p_observada,marker='o',markersize=4,label=LABEL[model],linewidth=1.5)
 ax.plot([0,1],[0,1],linestyle='--',lw=1,color='0.4')
 lim=min(1,1.1*max(d.p_predicha.max(),d.p_observada.max()))
 ax.set_xlim(0,lim);ax.set_ylim(0,lim)
 ax.set_xlabel('Probabilidad predicha media por decil ponderado')
 ax.set_ylabel('Proporción observada de beneficiarios')
 ax.set_title(f'Curvas de fiabilidad agrupadas · {prog} · LODO')
 ax.legend(loc='upper left',fontsize=9)
 save(f'06_reliability_{prog.lower()}')


def shift_scatter(prog):
 d=pd.read_csv(RES/'diagnostico_shift.csv',dtype={'departamento':str});d=d[d.programa==prog]
 fig,ax=plt.subplots(figsize=(8,5.3))
 ax.scatter(d.shift_smd_rms,d.loss_cat,s=40)
 for _,r in d.nlargest(5,'loss_cat').iterrows():
  ax.annotate(r.nombre,(r.shift_smd_rms,r.loss_cat),xytext=(4,5),textcoords='offset points',fontsize=8)
 ax.set_xlabel('RMS de diferencias de medias estandarizadas respecto a otros departamentos')
 ax.set_ylabel('Log-loss CatBoost en departamento excluido')
 ax.set_title(f'Desplazamiento de covariables y error · {prog}')
 save(f'07_shift_{prog.lower()}')


def sample_prevalence(prog):
 d=pd.read_csv(RES/'descriptivos_departamentos.csv',dtype={'departamento':str});d=d[d.programa==prog].sort_values('prevalencia_ponderada')
 fig,ax=plt.subplots(figsize=(9,8));y=np.arange(len(d));ax.barh(y,d.prevalencia_ponderada*100,edgecolor='white')
 ax.set_yticks(y,d.nombre);ax.set_xlabel('% ponderado de hogares que reportan beneficiarios')
 ax.set_title(f'Participación observada por departamento · {prog}')
 save(f'08_participacion_{prog.lower()}')


def idd_comparison(prog):
 l=pd.read_csv(RES/'metricas_nacionales.csv');l=l[l.programa==prog]
 i=pd.read_csv(RES/'validacion_iid_resumen.csv');i=i[i.programa==prog]
 mods=['prevalencia_fuente','logistica','catboost','mlp_erm','mlp_groupdro']
 vals1=[l[l.modelo==m].logloss.iloc[0] for m in mods]; vals2=[i[i.modelo==m].logloss.iloc[0] for m in mods]
 fig,ax=plt.subplots(figsize=(9,4.5));xx=np.arange(len(mods));width=.38
 ax.bar(xx-width/2,vals1,width,label='LODO: departamento no visto')
 ax.bar(xx+width/2,vals2,width,label='CV: conglomerados nuevos')
 ax.set_xticks(xx,[LABEL[m] for m in mods],rotation=18,ha='right')
 ax.set_ylabel('Log-loss ponderado ↓');ax.set_title(f'Validación territorial frente a validación convencional · {prog}')
 ax.legend();save(f'09_lodo_vs_iid_{prog.lower()}')


def sensitivity_fig(prog):
 d=pd.read_csv(RES/'sensibilidad_resumen.csv');d=d[(d.programa==prog)&(d.escenario.isin(['base_fija','sin_pca','sin_pesos_entrenamiento','region_natural']))]
 d=d.set_index('escenario').loc[['base_fija','sin_pca','sin_pesos_entrenamiento','region_natural']]
 fig,ax=plt.subplots(figsize=(8,4))
 bars=ax.barh(range(len(d)),d.logloss)
 ax.set_yticks(range(len(d)),['Base de referencia','Con PCA global (riesgo de fuga)','Sin ponderación al entrenar','Con región natural'])
 for i,v in enumerate(d.logloss):ax.text(v+0.004,i,f'{v:.4f}',fontsize=9,va='center')
 ax.set_xlim(0,max(d.logloss)*1.18)
 ax.invert_yaxis();ax.set_xlabel('Log-loss ponderado · validación LODO');ax.set_title(f'Sensibilidad de CatBoost · {prog}')
 save(f'10_sensibilidades_{prog.lower()}')


def tradeoff_frontier(prog):
 d=pd.read_csv(RES/'frontera_robustez.csv');d=d[d.programa==prog]
 fig,ax=plt.subplots(figsize=(8,6))
 for _,r in d.iterrows():
  ax.scatter([r.riesgo_nacional],[r.peor_dept],s=70)
  ax.annotate(LABEL[r.modelo],(r.riesgo_nacional,r.peor_dept),xytext=(5,6),textcoords='offset points',fontsize=9)
 ax.set_xlabel('Log-loss nacional ponderado (menor es mejor)')
 ax.set_ylabel('Peor log-loss departamental (menor es mejor)')
 ax.set_title(f'Frontera empírica: precisión nacional vs peor departamento · {prog}')
 save(f'11_frontera_{prog.lower()}')

def entropy_risk(prog):
 d=pd.read_csv(RES/'frontera_robustez.csv');d=d[d.programa==prog].sort_values('peor_ratio_entropia')
 fig,ax=plt.subplots(figsize=(9,4.3))
 xx=np.arange(len(d));vals=d.peor_ratio_entropia.to_numpy()
 ax.bar(xx,vals,edgecolor='white')
 ax.set_xticks(xx,[LABEL[m] for m in d.modelo],rotation=20,ha='right')
 ax.set_ylabel('Máximo (log-loss / entropía local)')
 ax.set_title(f'Pérdida peor normalizada por prevalencia · {prog}')
 ax.set_ylim(0,max(vals)*1.13)
 for i,v in enumerate(vals):ax.text(i,v+.02,f'{v:.2f}',ha='center',fontsize=9)
 save(f'12_normalizado_{prog.lower()}')

def main():
 for prog in PSET:
  bar_summary(prog)
  differences(prog,'mlp_groupdro','mlp_erm','02_dro_vs_erm')
  differences(prog,'catboost','logistica','03_catboost_vs_logistica')
  calibration_plot(prog);gap_ci(prog);reliability(prog);shift_scatter(prog);sample_prevalence(prog)
  idd_comparison(prog);sensitivity_fig(prog);tradeoff_frontier(prog);entropy_risk(prog)
 print('FIGURAS GENERADAS:',len(list(F.glob('*.png'))),flush=True)

if __name__=='__main__':main()
