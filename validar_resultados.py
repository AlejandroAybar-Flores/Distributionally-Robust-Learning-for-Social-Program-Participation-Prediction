from pathlib import Path
import json
import numpy as np
import pandas as pd
from estudio_core import load_data,get_subset,PSET,NUMERIC,CATEGORICAL,ASSETS,MODEL_NAMES,RES,ROOT,wlogloss,sha256

def main():
    data=load_data()
    checks=[]
    def record(name,passed,detail):
        checks.append({'verificacion':name,'correcto':bool(passed),'detalle':str(detail)})
        if not passed:raise AssertionError(name+': '+str(detail))
    record('household count',len(data)==36184,len(data))
    record('exactly 25 departments',data.codigo_departamento.nunique()==25,data.codigo_departamento.nunique())
    record('unique household identifiers',data[['ANIO','MES','codigo_departamento','CONGLOMERADO','VIVIENDA','HOGAR']].duplicated().sum()==0,'unique')
    record('17 original asset indicators',len(ASSETS)==17 and data[ASSETS].notna().all().all(),len(ASSETS))
    record('no disability count in predictors','n_integrantes_con_discapacidad' not in NUMERIC+CATEGORICAL,len(NUMERIC+CATEGORICAL))
    record('asset component is predictive field','indice_activos_pca' in NUMERIC,len(NUMERIC+CATEGORICAL))
    record('18 predictors',len(NUMERIC+CATEGORICAL)==18,len(NUMERIC+CATEGORICAL))
    record('no full-sample component in saved input','indice_activos_pca' not in data.columns,data.shape[1])
    pca=pd.read_csv(RES/'auditoria_pca_particiones.csv',dtype={'departamento_excluido':str})
    record('50 outer PCA fits',len(pca)==50,len(pca))
    record('all asset inputs active',pca.n_activos_con_varianza.eq(17).all(),pca.n_activos_con_varianza.unique())
    nat=pd.read_csv(RES/'metricas_nacionales.csv')
    dep=pd.read_csv(RES/'metricas_departamentales.csv',dtype={'departamento':str})
    for programme in PSET:
        sample=get_subset(data,programme)
        pred=pd.read_csv(RES/f'predicciones_{programme.lower()}.csv',dtype={'departamento':str})
        record(f'{programme}: household predictions complete',len(pred)==len(sample),len(pred))
        record(f'{programme}: unique household prediction',pred.excel_fila.is_unique,len(pred))
        record(f'{programme}: all departments held out',pred.departamento.nunique()==25,pred.departamento.nunique())
        expected={'Pension65':(11354,4531),'Juntos':(17671,3341)}[programme]
        record(f'{programme}: survey sample count',len(pred)==expected[0] and int(pred.y.sum())==expected[1],(len(pred),pred.y.sum()))
        px=pca[pca.programa==programme]
        record(f'{programme}: asset PCA refitted per holdout',len(px)==25 and px.varianza_pc1.std()>0,px.varianza_pc1.agg(['min','max']).to_dict())
        record(f'{programme}: source and test partition sizes',all(px.n_train.to_numpy()+px.n_test.to_numpy()==len(sample)),len(px))
        record(f'{programme}: every model evaluated',set(nat[nat.programa==programme].modelo)==set(MODEL_NAMES),len(nat[nat.programa==programme]))
        record(f'{programme}: model-department rows',len(dep[dep.programa==programme])==25*len(MODEL_NAMES),len(dep[dep.programa==programme]))
        for model in MODEL_NAMES:
            probs=pred[f'p_{model}']
            record(f'{programme}/{model}: probabilities in (0,1)',probs.notna().all() and probs.gt(0).all() and probs.lt(1).all(),probs.agg(['min','max']).to_dict())
            score=wlogloss(pred.y,probs,pred.factor_hogar)
            val=nat[(nat.programa==programme)&(nat.modelo==model)].logloss.iat[0]
            record(f'{programme}/{model}: reported loss agrees with predictions',abs(score-val)<1e-8,(score,val))
    for name in ['intervalos_contrastes','contrastes_peor_departamento','frontera_robustez','calibracion_departamentos_ic','sensibilidad_resumen','validacion_iid_resumen','validacion_regiones','asociacion_shift_error','calibracion_deciles','auditoria_pca_particiones']:
        path=RES/(name+'.csv')
        record(f'analysis file {name}',path.exists() and path.stat().st_size>100,(path.exists(),path.stat().st_size if path.exists() else 0))
    result=pd.DataFrame(checks)
    (ROOT/'qa').mkdir(exist_ok=True)
    result.to_csv(ROOT/'qa'/'verificaciones_v7.csv',index=False)
    info={'checks':len(checks),'all_passed':True,'database_sha256':sha256(ROOT/'datos'/'ENAPRES2025_activos_originales.pkl'),'tested_programmes':list(PSET),'pca_fits':len(pca),'report':str(ROOT/'qa'/'verificaciones_v7.csv')}
    (ROOT/'qa'/'resumen_v7.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(info,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
