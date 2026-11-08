
from pathlib import Path
import os, json
import pandas as pd
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment,Border,Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart,Reference
from estudio_core import ROOT,RES,DEPARTMENTS,MODEL_NAMES

WORKBOOK=ROOT/'resultados'/'Tablas_ENAPRES_2025_V7.xlsx'

SHEETS=[
 ('Muestra','muestra_principal.csv'),('Metricas_nacionales','metricas_nacionales.csv'),('Contrastes_IC','intervalos_contrastes.csv'),
 ('Departamentos','metricas_departamentales.csv'),('Descriptivos','descriptivos_departamentos.csv'),
 ('Calibracion_IC','calibracion_departamentos_ic.csv'),('Riesgos_peor_grupo','peores_riesgos_bootstrap.csv'),
 ('Subgrupos','metricas_subgrupos.csv'),('Semillas_MLP','metricas_semillas.csv'),
 ('Hiper_P65','hiperparametros_pension65.csv'),('Hiper_Juntos','hiperparametros_juntos.csv'),
 ('Sensibilidades','sensibilidad_resumen.csv'),('Validacion_IID','validacion_iid_resumen.csv'),
 ('Validacion_regional','validacion_regiones.csv'),('Shift','diagnostico_shift.csv'),
 ('Shift_asociacion','asociacion_shift_error.csv'),('Frontera_robustez','frontera_robustez.csv'),('Contraste_peor_dept','contrastes_peor_departamento.csv'),('Calibracion_deciles','calibracion_deciles.csv'),
 ('Auditoria_variables','auditoria_variables.csv'),('Auditoria_PCA','auditoria_pca_particiones.csv'),('Scores_PCA_Test','scores_pca_departamento_excluido.csv'),('Faltantes_por_dept','faltantes_por_departamento.csv')
]


def main():
 WORKBOOK.parent.mkdir(exist_ok=True)
 wb=Workbook();ws=wb.active;ws.title='INICIO';ws.sheet_view.showGridLines=False
 notes=[
 ('ESTUDIO','ENAPRES 2025: generalizacion de dominio y robustez territorial'),
 ('UNIDAD','Hogar; indicadores binarios de al menos una persona beneficiaria'),
 ('PROGRAMAS','Pension65 y Juntos, analizados independientemente'),
 ('VALIDACION','25-fold leave-one-department-out (LODO), con seleccion interna solo en origen'),
 ('METRICAS','Log-loss, Brier, AUROC, AUPRC, ECE, sesgo calibracion, riesgos por depto'),
 ('INFERENCIA','Bootstrap aproximado por conglomerado, condicional en predicciones fijas'),
 ('IMPORTANTE','NO es inferencia por diseño certificada: faltan estratos oficiales'),
 ('PCA','Primer componente de 17 activos, calculado por partición de entrenamiento'),
 ('DATOS FUENTE','datos/ENAPRES2025_activos_originales.pkl; 36,184 hogares; 25 departamentos'),
 ('CODIGOS','Codigo de departamento de 01 a 25, ver hoja CODIGOS'),
 ('FUENTE OFICIAL','https://www.gob.pe/inei'),
 ('CITACION','Fuente: 2025 National Survey of Budget Programmes, INEI'),
 ('ARCHIVOS','Códigos de análisis, tablas y predicciones fuera de departamento en el mismo paquete')]
 for i,(tag,val) in enumerate(notes,start=2):
  ws.cell(i,1,tag);ws.cell(i,2,val)
  ws.cell(i,1).font=Font(bold=True,color='577387',size=10)
  ws.cell(i,2).font=Font(color='222E35',size=10)
  if tag.startswith('IMPORTANTE'):
   ws.cell(i,2).fill=PatternFill('solid',fgColor='FFF0DA')
 ws.column_dimensions['A'].width=22;ws.column_dimensions['B'].width=105
 ws.row_dimensions[1].height=32;ws['A1']='RESULTADOS ENAPRES 2025';ws['A1'].font=Font(bold=True,size=17,color='234560')
 for title,filename in SHEETS:
  fp=RES/filename
  if not fp.exists():print('Missing optional:',fp);continue
  df=pd.read_csv(fp,dtype={'departamento':str,'codigo_departamento':str})
  w=wb.create_sheet(title);w.sheet_view.showGridLines=False;w.freeze_panes='A2'
  for j,col in enumerate(df.columns,start=1):
   c=w.cell(1,j,str(col));c.font=Font(name='Aptos',size=10,bold=True,color='FFFFFF')
   c.fill=PatternFill('solid',fgColor='244560');c.alignment=Alignment(vertical='center',wrap_text=True)
  w.row_dimensions[1].height=31
  for i,row in enumerate(df.itertuples(index=False,name=None),start=2):
   for j,v in enumerate(row,start=1):
    if pd.isna(v):v=None
    cell=w.cell(i,j,v)
    cell.font=Font(name='Aptos',size=9,color='167248' if isinstance(v,(float,int)) else '293946')
    if isinstance(v,float):cell.number_format='0.0000' if abs(v)<1e5 else '#,##0.0'
    elif isinstance(v,int):cell.number_format='#,##0'
    if str(df.columns[j-1]) in ('departamento','codigo_departamento'):
     cell.number_format='@';cell.alignment=Alignment(horizontal='center')
  w.auto_filter.ref=f'A1:{get_column_letter(len(df.columns))}{len(df)+1}'
  for j,col in enumerate(df.columns,start=1):
   ext=str(col)
   if ext in ('nombre','nombre_departamento','max_variable','regla_muestra','escenario','contraste','variable','modelo','programa'):
    width=min(45,max(19,len(ext)+12))
   else:width=min(27,max(13,len(ext)+2))
   w.column_dimensions[get_column_letter(j)].width=width
  print('sheet',title,df.shape)
 codes=wb.create_sheet('CODIGOS');codes.append(['Codigo','Departamento']);codes.sheet_view.showGridLines=False
 for k,v in DEPARTMENTS.items():codes.append([k,v])
 codes['A1'].font=Font(bold=True,color='FFFFFF');codes['B1'].font=Font(bold=True,color='FFFFFF')
 for c in codes[1]:c.fill=PatternFill('solid',fgColor='244560')
 codes.column_dimensions['A'].width=13;codes.column_dimensions['B'].width=27
 for row in codes.iter_rows(min_row=2):row[0].number_format='@'
 wb.active=0;wb.save(WORKBOOK)
 print('saved',WORKBOOK,WORKBOOK.stat().st_size)
 verify=load_workbook(WORKBOOK,read_only=True,data_only=True)
 assert len(verify.sheetnames)>=17
 for x in ['INICIO','Metricas_nacionales','Contrastes_IC','CODIGOS']:
  assert x in verify.sheetnames
 verify.close()

if __name__=='__main__':main()
