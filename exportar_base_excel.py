from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook,load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill,Alignment
from estudio_core import ROOT


def main():
    base=pd.read_pickle(ROOT/'datos'/'ENAPRES2025_activos_originales.pkl')
    path=ROOT/'datos'/'ENAPRES2025_hogares_activos.xlsx'
    wb=Workbook(write_only=True)
    ws=wb.create_sheet('Hogares_y_activos')
    header=[]
    for name in base.columns:
        cell=WriteOnlyCell(ws,value=name)
        cell.font=Font(name='Aptos',bold=True,color='FFFFFF',size=10)
        cell.fill=PatternFill(fill_type='solid',fgColor='173B59')
        cell.alignment=Alignment(wrap_text=True,vertical='center')
        header.append(cell)
    ws.append(header)
    for row in base.itertuples(index=False,name=None):
        vals=[]
        for value in row:
            if pd.isna(value):vals.append(None)
            elif isinstance(value,(np.integer,)):vals.append(int(value))
            elif isinstance(value,(np.floating,)):vals.append(float(value))
            else:vals.append(value)
        ws.append(vals)
    wb.save(path)
    a=load_workbook(path,read_only=True,data_only=True)
    s=a.active
    s.reset_dimensions()
    row_iter=s.iter_rows(values_only=True)
    assert len(next(row_iter))==len(base.columns)
    assert sum(1 for _ in row_iter)==len(base)
    assert 'indice_activos_pca' not in base.columns
    assert 'n_integrantes_con_discapacidad' not in base.columns
    print(path,len(base),len(base.columns))

if __name__=='__main__':main()
