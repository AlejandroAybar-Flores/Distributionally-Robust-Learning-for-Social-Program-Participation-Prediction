from pathlib import Path
import numpy as np
import pandas as pd
from estudio_core import load_data,get_subset,PSET,DEPARTMENTS,WeightedAssetPCA,ASSETS,RES


def main():
    data=load_data()
    audit=[]
    scores=[]
    for prog in PSET:
        d=get_subset(data,prog)
        for dep in sorted(DEPARTMENTS):
            mask=d.codigo_departamento.eq(dep)
            tr=d.loc[~mask]
            te=d.loc[mask]
            mod=WeightedAssetPCA().fit(tr,tr.factor_hogar.to_numpy(float))
            pts=mod.transform(te)
            audit.append({'programa':prog,'departamento_excluido':dep,'n_train':len(tr),'n_test':len(te),'varianza_pc1':mod.variance_ratio,'n_activos_con_varianza':len(mod.use),'media_pc1_test':float(np.average(pts,weights=te.factor_hogar)),'sd_pc1_test':float(np.std(pts))})
            for name,loading in zip([ASSETS[i] for i in mod.use],mod.loadings):
                audit[-1]['carga_'+name]=float(loading)
            scores.append(pd.DataFrame({'excel_fila':te.excel_fila,'programa':prog,'departamento_excluido':dep,'indice_activos_pca_fitted_without_department':pts}))
    pd.DataFrame(audit).to_csv(RES/'auditoria_pca_particiones.csv',index=False)
    pd.concat(scores,ignore_index=True).to_csv(RES/'scores_pca_departamento_excluido.csv',index=False)
    print(pd.DataFrame(audit).groupby('programa').varianza_pc1.agg(['min','mean','max']).to_string())

if __name__=='__main__':main()
