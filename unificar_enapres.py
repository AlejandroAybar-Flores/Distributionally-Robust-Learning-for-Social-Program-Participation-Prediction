from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
try:
    import pyreadstat
except ImportError:
    from read_sav_native import read_sav
    class NativeReader:
        @staticmethod
        def read_sav(path, **kwargs):
            return read_sav(path,kwargs.get('usecols'))[0],None
    pyreadstat=NativeReader()

@dataclass
class Configuracion:
    directorio_entrada: Path = Path('/mnt/data') if Path('/mnt/data').exists() else Path.cwd()
    directorio_salida: Optional[Path] = None

    resfin_permitidos: tuple[int, ...] = (1,)

    solo_urbano: bool = False
    edad_minima_valida: int = 0
    edad_maxima_valida: int = 120
    incluir_modulo_500: bool = False

    excel_filas_maximas_por_hoja: int = 1_048_575


CLAVE_HOGAR = ['ANIO', 'MES', 'CONGLOMERADO', 'VIVIENDA', 'HOGAR']
CLAVE_PERSONA = CLAVE_HOGAR + ['P201']

PROGRAMAS = {
    'qali_warma_desayuno': 'P306_1',
    'qali_warma_almuerzo': 'P306_2',
    'cuna_mas_guarderia': 'P306_3',
    'cuna_mas_acompanamiento': 'P306_4',
    'pension_65': 'P306_6',
    'comedor_popular': 'P306_7',
    'vaso_de_leche': 'P306_8',
    'juntos': 'P306_9',
    'sis': 'P306_10',
    'otro_programa': 'P306_11',
}
NODOS_ORIGINALES = [f'prog_{nombre}' for nombre in PROGRAMAS]
NODOS = [
    'prog_qali_warma',
    'prog_cuna_mas',
    'prog_pension_65',
    'prog_juntos',
    'prog_vaso_de_leche',
    'prog_comedor_popular',
    'prog_sis',
]
ACTIVOS = [f'P189_{i}' for i in range(1, 18)]

COLUMNAS_CAP200 = CLAVE_PERSONA + [
    'RESFIN', 'AREA', 'CCDD', 'NOMBREDD', 'REGIONNATU', 'P202C', 'P203A',
    'P204', 'P205', 'P206', 'P207', 'P208_A', 'FACTOR',
]
COLUMNAS_CAP300 = CLAVE_PERSONA + [
    'RESFIN', 'P300C_A', 'P300C_C', 'P300_1',
    *[f'P301_{i}' for i in range(1, 8)],
    *PROGRAMAS.values(), 'P307', 'P308', 'P312', 'P313',
]
COLUMNAS_CAP100E = CLAVE_HOGAR + [
    'RESFIN', 'AREA', 'P101', 'P102', 'P103', 'P104', 'P105', 'P106',
    'P113', 'P115A', 'P135', 'P136', 'P138', 'P170', 'P179',
    *ACTIVOS, 'P189_18', 'ESTRATO', 'FACTOR',
]
COLUMNAS_CAP500 = CLAVE_PERSONA + ['RESFIN', 'P500_A', 'P501', 'P502']

ESQUEMA_LECTURA = {
    '200': ('200', COLUMNAS_CAP200),
    '300C': ('300', COLUMNAS_CAP300),
    '100E': ('100E', COLUMNAS_CAP100E),
    '500': ('500', COLUMNAS_CAP500),
}

COLUMNAS_FINALES = ['ANIO', 'MES', 'CONGLOMERADO', 'VIVIENDA', 'HOGAR', 'factor_hogar', 'prog_qali_warma', 'prog_cuna_mas', 'prog_pension_65', 'prog_juntos', 'prog_vaso_de_leche', 'prog_comedor_popular', 'prog_sis', 'edad_jefe', 'sexo_jefe', 'tamano_hogar', 'educacion_jefe_4niveles', 'lengua_materna_jefe', 'jefe_pertenece_pueblo_indigena', 'area', 'codigo_departamento', 'region_natural', 'agua_red_publica', 'saneamiento_red_publica', 'electricidad_red_publica', 'indice_activos_pca', 'n_ninos_0_11', 'n_adolescentes_12_17', 'n_adultos_65mas', 'n_mujeres', 'hogar_multigeneracional', 'hogar_con_discapacidad', 'n_integrantes_con_discapacidad', 'internet_en_hogar', 'hogar_con_dispositivo_digital']
assert len(COLUMNAS_FINALES) == 35 and len(set(COLUMNAS_FINALES)) == 35


def numerica(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors='coerce')


def yn(serie: pd.Series) -> pd.Series:
    s = numerica(serie)
    return pd.Series(np.select([s.eq(1), s.eq(2)], [1.0, 0.0], default=np.nan),
                     index=serie.index, dtype='float64')


def seleccion(serie: pd.Series) -> pd.Series:
    s = numerica(serie)
    return pd.Series(np.select([s.eq(1), s.eq(0)], [1.0, 0.0], default=np.nan),
                     index=serie.index, dtype='float64')


def clave_como_texto(serie: pd.Series) -> pd.Series:
    r = serie.astype('string').str.strip().str.replace(r'\.0+$', '', regex=True)
    r = r.replace({'': pd.NA, 'nan': pd.NA, 'NaN': pd.NA, 'None': pd.NA, '<NA>': pd.NA})
    return r


def estandarizar_claves(datos: pd.DataFrame, columnas: list[str], nombre: str) -> pd.DataFrame:
    faltantes = [c for c in columnas if c not in datos.columns]
    if faltantes:
        raise ValueError(f'{nombre}: faltan identificadores {faltantes}')
    datos = datos.copy()
    for c in columnas:
        datos[c] = clave_como_texto(datos[c])
    n_nulas = int(datos[columnas].isna().any(axis=1).sum())
    if n_nulas:
        raise ValueError(f'{nombre}: {n_nulas} filas con llaves incompletas. No se unen.')
    return datos


def verificar_unicidad(datos: pd.DataFrame, llave: list[str], nombre: str) -> None:
    dup = datos.duplicated(llave, keep=False)
    if dup.any():
        ejemplos = datos.loc[dup, llave].head(5).to_dict('records')
        raise ValueError(f'{nombre}: llave NO unica en {int(dup.sum())} filas. Ejemplos: {ejemplos}')


def descubrir_archivos(entrada: Path, tag: str) -> list[Path]:
    expresion = re.compile(rf'^CAP_{re.escape(tag)}(?:_|$)', flags=re.IGNORECASE)
    return sorted((f for f in entrada.rglob('*') if f.is_file() and
                   f.suffix.lower() == '.sav' and expresion.search(f.stem)),
                  key=lambda p: str(p).lower())


def cargar_capitulo(config: Configuracion, nombre: str, auditoria: dict) -> pd.DataFrame:
    if pyreadstat is None:
        raise ImportError('No se instalo pyreadstat. En Jupyter ejecute: %pip install pyreadstat pyarrow')
    tag, seleccionadas = ESQUEMA_LECTURA[nombre]
    archivos = descubrir_archivos(Path(config.directorio_entrada), tag)
    if not archivos:
        raise FileNotFoundError(f'No se encontro CAP_{tag}_*.sav en {config.directorio_entrada}')
    piezas = []
    for ruta in archivos:
        tabla, metadata = pyreadstat.read_sav(
            str(ruta), usecols=seleccionadas, apply_value_formats=False,
            formats_as_category=False, user_missing=False,
        )
        tabla.columns = [str(c).strip().upper() for c in tabla.columns]
        faltantes = set(seleccionadas) - set(tabla.columns)
        if faltantes:
            raise KeyError(f'{ruta.name}: faltan variables exigidas: {sorted(faltantes)}')
        inicial = len(tabla)
        tabla = tabla.loc[numerica(tabla['RESFIN']).isin(config.resfin_permitidos)].copy()
        auditoria['lectura'].append({
            'modulo': nombre, 'archivo': ruta.name, 'filas_totales': int(inicial),
            'filas_post_resfin': int(len(tabla)),
            'resfin_permitidos': list(config.resfin_permitidos),
        })
        piezas.append(tabla)
    unido = pd.concat(piezas, ignore_index=True)
    llave = CLAVE_HOGAR if nombre == '100E' else CLAVE_PERSONA
    unido = estandarizar_claves(unido, llave, f'CAP{nombre}')
    return unido


def incorporar_persona_objetivo(datos: pd.DataFrame, campo: str, nombre: str, auditoria: dict) -> pd.DataFrame:
    out = datos.copy()
    objetivo = numerica(out[campo])
    objetivo_valido = objetivo.ge(1) & objetivo.le(80)
    objetivo_txt = clave_como_texto(out[campo])
    diferencias = objetivo_valido & objetivo_txt.ne(out['P201']).fillna(False)
    auditoria['personas_objetivo'][nombre] = {
        'diferencias_P201_vs_objetivo': int(diferencias.sum()),
        'objetivo_invalido_o_ausente': int((~objetivo_valido).sum()),
    }
    out['P201'] = out['P201'].where(~objetivo_valido, objetivo_txt)
    return out


def _uno_si_algun_positivo_otro_cero(serie: pd.Series) -> float:
    s = pd.to_numeric(serie, errors='coerce')
    if s.eq(1).any():
        return 1.0
    if s.notna().all() and len(s) and s.eq(0).all():
        return 0.0
    return np.nan


def construir_padron(cap200: pd.DataFrame, config: Configuracion, auditoria: dict):
    d = cap200.copy()
    for c in ['P204', 'P205', 'P206', 'P207', 'P208_A', 'P202C', 'P203A',
              'AREA', 'REGIONNATU', 'FACTOR']:
        d[c] = numerica(d[c])
    residente = ((d['P204'] == 1) & (d['P205'] == 2)) | ((d['P204'] == 2) & (d['P206'] == 1))
    edad_valida = d['P208_A'].between(config.edad_minima_valida, config.edad_maxima_valida)
    conservar = residente & edad_valida
    if config.solo_urbano:
        conservar &= d['AREA'].eq(1)
    auditoria['filtros']['cap200'] = {
        'personas_antes': int(len(d)), 'residentes_habituales': int(residente.sum()),
        'residentes_edad_valida': int((residente & edad_valida).sum()),
        'personas_finales': int(conservar.sum()),
        'no_urban_only': not config.solo_urbano,
    }
    d = d.loc[conservar].copy()
    verificar_unicidad(d, CLAVE_PERSONA, 'CAP200 residentes')
    if d.empty:
        raise ValueError('CAP200 no tiene residentes despues del filtro')

    edad = d['P208_A']
    sexo = d['P207']
    d['_ninos_0_11'] = edad.between(0, 11).astype('int8')
    d['_adolescentes_12_17'] = edad.between(12, 17).astype('int8')
    d['_menores_18'] = edad.lt(18).astype('int8')
    d['_menores_5'] = edad.between(0, 4).astype('int8')
    d['_menores_3'] = edad.between(0, 2).astype('int8')
    d['_adultos_18_64'] = edad.between(18, 64).astype('int8')
    d['_adultos_65mas'] = edad.ge(65).astype('int8')
    d['_mujeres'] = sexo.eq(2).astype('int8')
    d['_sexo_sin_dato'] = (~sexo.isin([1, 2])).astype('int8')
    d['_nucleos'] = d['P203A'].where(d['P203A'].gt(0))

    g = d.groupby(CLAVE_HOGAR, dropna=False, sort=False)
    hh = g.agg(
        tamano_hogar=('P201', 'size'),
        edad_promedio=('P208_A', 'mean'),
        n_ninos_0_11=('_ninos_0_11', 'sum'),
        n_adolescentes_12_17=('_adolescentes_12_17', 'sum'),
        n_menores_18=('_menores_18', 'sum'),
        n_menores_5=('_menores_5', 'sum'),
        n_menores_3=('_menores_3', 'sum'),
        n_adultos_18_64=('_adultos_18_64', 'sum'),
        n_adultos_65mas=('_adultos_65mas', 'sum'),
        n_mujeres=('_mujeres', 'sum'),
        n_sexo_sin_dato=('_sexo_sin_dato', 'sum'),
        n_nucleos_familiares=('_nucleos', 'nunique'),
        area=('AREA', 'first'),
        codigo_departamento=('CCDD', 'first'),
        departamento=('NOMBREDD', 'first'),
        region_natural=('REGIONNATU', 'first'),
        factor_cap200=('FACTOR', 'first'),
    ).reset_index()
    hh['prop_mujeres'] = hh['n_mujeres'] / hh['tamano_hogar']
    hh.loc[hh['n_sexo_sin_dato'] > 0, 'prop_mujeres'] = np.nan
    hh['hogar_con_ninos'] = (hh['n_ninos_0_11'] > 0).astype('int8')
    hh['hogar_con_adolescentes'] = (hh['n_adolescentes_12_17'] > 0).astype('int8')
    hh['hogar_con_adultos_mayores'] = (hh['n_adultos_65mas'] > 0).astype('int8')
    hh['hogar_intergeneracional'] = ((hh['n_menores_18'] > 0) &
                                     (hh['n_adultos_65mas'] > 0)).astype('int8')
    hh['hogar_multigeneracional'] = ((hh['n_menores_18'] > 0) &
                                     (hh['n_adultos_18_64'] > 0) &
                                     (hh['n_adultos_65mas'] > 0)).astype('int8')
    hh['region_natural'] = pd.to_numeric(hh['region_natural'], errors='coerce')
    hh['area'] = pd.to_numeric(hh['area'], errors='coerce')
    hh['factor_cap200'] = pd.to_numeric(hh['factor_cap200'], errors='coerce')

    jefes = d.loc[d['P202C'].eq(1), CLAVE_PERSONA + ['P207', 'P208_A']].copy()
    verificar_unicidad(jefes, CLAVE_HOGAR, 'Jefe/jefa del hogar')
    jefes = jefes.rename(columns={'P207': 'sexo_jefe', 'P208_A': 'edad_jefe',
                                  'P201': 'orden_jefe'})
    hh = hh.merge(jefes, how='left', on=CLAVE_HOGAR, validate='one_to_one', sort=False)
    auditoria['filtros']['hogares_sin_jefe_identificado'] = int(hh['orden_jefe'].isna().sum())
    return d, hh


def agregar_cap300_a_hogares(personas: pd.DataFrame, cap300: pd.DataFrame,
                            hh: pd.DataFrame, config: Configuracion, auditoria: dict):
    d = incorporar_persona_objetivo(cap300, 'P300C_A', 'CAP300C', auditoria)
    verificar_unicidad(d, CLAVE_PERSONA, 'CAP300C')
    solo = CLAVE_PERSONA + ['P300_1', *[f'P301_{i}' for i in range(1, 8)],
                            *PROGRAMAS.values(), 'P307', 'P308', 'P312', 'P313']
    d = d[solo].copy()
    a = personas.merge(d, how='left', on=CLAVE_PERSONA, validate='one_to_one',
                       indicator='_origen_cap300', sort=False)
    a['_registro_cap300'] = a['_origen_cap300'].eq('both').astype('int8')
    a['_discapacidad'] = pd.DataFrame({
        k: yn(a[f'P301_{k}']) for k in range(1, 8)
    }, index=a.index).apply(_uno_si_algun_positivo_otro_cero, axis=1)
    a['_pertenencia_indigena'] = yn(a['P313'])
    a['_no_tiene_dni'] = yn(a['P300_1']).map({0.0: 1.0, 1.0: 0.0})

    grp = a.groupby(CLAVE_HOGAR, dropna=False, sort=False)
    c = grp.agg(
        n_registros_cap300=('_registro_cap300', 'sum'),
        n_personas_discapacidad_observada=('_discapacidad', 'count'),
        n_personas_discapacidad_confirmada=('_discapacidad', lambda x: int(x.eq(1).sum())),
        n_integrantes_indigenas_observados=('_pertenencia_indigena', 'count'),
        n_integrantes_indigenas_confirmados=('_pertenencia_indigena', lambda x: int(x.eq(1).sum())),
        n_integrantes_sin_dni_confirmados=('_no_tiene_dni', lambda x: int(x.eq(1).sum())),
    ).reset_index()
    c = c.merge(hh[CLAVE_HOGAR + ['tamano_hogar']], on=CLAVE_HOGAR,
                how='left', validate='one_to_one')
    c['cobertura_cap300_completa'] = (
        (c['n_registros_cap300'] == c['tamano_hogar'])
    ).astype('int8')

    UNIVERSO_EDAD_PROGRAMAS = {
        'qali_warma_desayuno': (0, 17),
        'qali_warma_almuerzo': (0, 17),
        'cuna_mas_guarderia': (0, 17),
        'cuna_mas_acompanamiento': (0, 17),
        'pension_65': (65, config.edad_maxima_valida),
    }

    def booleano_presencia(nombre: str, campo: str) -> None:
        raw = numerica(a[campo])
        edad = numerica(a['P208_A'])
        intervalo = UNIVERSO_EDAD_PROGRAMAS.get(nombre)
        elegible = (edad.between(*intervalo) if intervalo is not None
                    else pd.Series(True, index=a.index))

        a[f'_si_{nombre}'] = raw.eq(1).astype('int8')
        a[f'_eleg_{nombre}'] = elegible.astype('int8')
        a[f'_contestado_{nombre}'] = (elegible & raw.isin([1, 2])).astype('int8')
        resumen = grp[[f'_si_{nombre}', f'_eleg_{nombre}',
                       f'_contestado_{nombre}']].sum().reset_index()

        n_si = resumen[f'_si_{nombre}']
        n_elegibles = resumen[f'_eleg_{nombre}']
        n_respondieron = resumen[f'_contestado_{nombre}']
        completo = n_elegibles.eq(n_respondieron)

        presencia = np.select([n_si.gt(0), completo],
                              [1.0, 0.0], default=np.nan)
        c[f'prog_{nombre}'] = presencia.astype('float64')
        c[f'n_beneficiarios_{nombre}'] = n_si.astype('float64').to_numpy()
        auditoria.setdefault('programas', {})[nombre] = {
            'hogares_si': int(np.count_nonzero(presencia == 1.0)),
            'hogares_no': int(np.count_nonzero(presencia == 0.0)),
            'hogares_indeterminados': int(np.isnan(presencia).sum()),
            'hogares_sin_personas_en_universo': int(n_elegibles.eq(0).sum()),
            'personas_si': int(n_si.sum()),
        }

    for nombre, campo in PROGRAMAS.items():
        booleano_presencia(nombre, campo)

    print('Resumen verificado de la agregacion de programas por hogar:')
    for nombre, s in auditoria['programas'].items():
        print(f"  {nombre:29} si={s['hogares_si']:>6,} "
              f"no={s['hogares_no']:>6,} "
              f"indeterminados={s['hogares_indeterminados']:>6,}")

    def unir_modalidades(primera: str, segunda: str) -> np.ndarray:
        a_ = c[primera]
        b_ = c[segunda]
        return np.select(
            [a_.eq(1) | b_.eq(1), a_.eq(0) & b_.eq(0)],
            [1.0, 0.0], default=np.nan
        )

    c['prog_qali_warma'] = unir_modalidades(
        'prog_qali_warma_desayuno', 'prog_qali_warma_almuerzo'
    )
    c['prog_cuna_mas'] = unir_modalidades(
        'prog_cuna_mas_guarderia', 'prog_cuna_mas_acompanamiento'
    )
    c['n_programas_hogar'] = c[NODOS].sum(axis=1, min_count=len(NODOS))
    c['hogar_con_discapacidad'] = grp['_discapacidad'].agg(
        _uno_si_algun_positivo_otro_cero
    ).to_numpy(dtype='float64')
    c['n_integrantes_con_discapacidad'] = c['n_personas_discapacidad_confirmada'].where(
        c['n_personas_discapacidad_observada'].eq(c['tamano_hogar']), np.nan
    )
    c['hogar_con_integrante_indigena'] = grp['_pertenencia_indigena'].agg(
        _uno_si_algun_positivo_otro_cero
    ).to_numpy(dtype='float64')
    c['hogar_con_integrante_sin_dni'] = grp['_no_tiene_dni'].agg(
        _uno_si_algun_positivo_otro_cero
    ).to_numpy(dtype='float64')

    j = a.loc[a['P202C'].eq(1), CLAVE_HOGAR + ['P307', 'P308', 'P312', 'P313']].copy()
    verificar_unicidad(j, CLAVE_HOGAR, 'Jefe CAP300C')
    j = j.rename(columns={
        'P307': 'lengua_materna_jefe',
        'P308': 'nivel_educativo_jefe',
        'P312': 'autoidentificacion_etnica_jefe',
        'P313': 'pertenencia_indigena_jefe_raw',
    })
    c = c.merge(j, on=CLAVE_HOGAR, how='left', validate='one_to_one')
    for k in ['lengua_materna_jefe', 'nivel_educativo_jefe',
              'autoidentificacion_etnica_jefe']:
        c[k] = numerica(c[k]).where(numerica(c[k]).gt(0))
    lengua = c['lengua_materna_jefe']
    c['jefe_lengua_originaria'] = np.where(lengua.between(1, 9), 1.0,
                                            np.where(lengua.between(10, 13), 0.0, np.nan))
    edu = c['nivel_educativo_jefe']
    mapa_edu = {1: 1, 2: 1, 3: 1, 4: 1, 7: 1, 5: 2, 6: 2,
                8: 3, 10: 3, 9: 4, 11: 4, 12: 4}
    c['educacion_jefe_4niveles'] = edu.map(mapa_edu).astype('float64')
    c['jefe_pertenece_pueblo_indigena'] = yn(c['pertenencia_indigena_jefe_raw'])
    c['jefe_autoidentificacion_indigena'] = np.where(
        c['autoidentificacion_etnica_jefe'].between(1, 4), 1.0,
        np.where(c['autoidentificacion_etnica_jefe'].between(5, 8), 0.0, np.nan)
    )
    auditoria['uniones']['CAP300C'] = {
        'registros_cap200_residentes': int(len(personas)),
        'registros_cap300_efectivamente_enlazados': int(a['_registro_cap300'].sum()),
        'hogares_cobertura_completa': int(c['cobertura_cap300_completa'].sum()),
        'hogares_cobertura_incompleta': int((c['cobertura_cap300_completa'] == 0).sum()),
    }
    return c.drop(columns=['tamano_hogar']), a


def construir_hogares_cap100e(cap100e: pd.DataFrame, auditoria: dict):
    df = cap100e.copy()
    verificar_unicidad(df, CLAVE_HOGAR, 'CAP100E')
    salida = df[CLAVE_HOGAR].copy()
    p105 = numerica(df['P105'])
    p106 = numerica(df['P106'])
    salida['habitaciones_vivienda'] = p105.where(p105.gt(0))
    salida['tenencia_vivienda'] = p106.where(p106.between(1, 8))
    salida['vivienda_propia'] = np.where(p106.isin([2, 3, 4]), 1.0,
                                       np.where(p106.isin([1, 5, 6, 7, 8]), 0.0, np.nan))
    p101 = numerica(df['P101'])
    p102 = numerica(df['P102'])
    p103 = numerica(df['P103'])
    p104 = numerica(df['P104'])
    salida['vivienda_material_precario_aprox'] = np.where(
        p101.isin([5, 6, 7]) | p102.isin([8, 9]) |
        p103.eq(6) | p104.isin([6, 7, 8]), 1.0,
        np.where(p101.notna() & p102.notna() & p103.notna() & p104.notna(), 0.0, np.nan)
    )
    salida['electricidad_red_publica'] = yn(df['P113'])
    salida['electricidad_diaria'] = yn(df['P115A'])
    fuente_agua = numerica(df['P135'])
    salida['agua_red_publica'] = np.where(fuente_agua.isin([1, 2, 3, 9]), 1.0,
                                          np.where(fuente_agua.between(1, 9), 0.0, np.nan))
    salida['agua_reportada_potable'] = yn(df['P136'])
    p138 = numerica(df['P138'])
    salida['agua_todos_los_dias'] = np.where(p138.eq(1), 1.0,
                                            np.where(p138.isin([2, 3]), 0.0, np.nan))
    saneamiento = numerica(df['P170'])
    salida['saneamiento_red_publica'] = np.where(saneamiento.isin([1, 2, 9]), 1.0,
                                               np.where(saneamiento.between(1, 10), 0.0, np.nan))
    salida['recoleccion_municipal_basura'] = yn(df['P179'])
    salida['n_servicios_basicos'] = salida[
        ['agua_red_publica', 'saneamiento_red_publica', 'electricidad_red_publica']
    ].sum(axis=1, min_count=3)
    salida['tres_servicios_basicos'] = np.where(
        salida['n_servicios_basicos'].notna(),
        (salida['n_servicios_basicos'] == 3).astype('float64'), np.nan
    )
    salida['area_cap100e'] = numerica(df['AREA'])
    salida['estrato_geografico_reportado'] = numerica(df['ESTRATO'])
    salida['factor_hogar'] = numerica(df['FACTOR']).where(numerica(df['FACTOR']).gt(0))

    for codigo in ACTIVOS:
        salida['activo_' + codigo.lower()] = seleccion(df[codigo])
    columnas_activos = ['activo_' + codigo.lower() for codigo in ACTIVOS]
    salida['n_bienes'] = salida[columnas_activos].sum(axis=1, min_count=len(ACTIVOS))
    sin_bienes_reportado = seleccion(df['P189_18']).eq(1)
    auditoria['activos'] = {
        'filas_con_nada_seleccionado_y_al_menos_un_activo': int(
            (sin_bienes_reportado & salida['n_bienes'].gt(0)).sum()
        ),
        'activos_para_pca': columnas_activos,
    }
    salida['internet_en_hogar'] = salida['activo_p189_14']
    salida['hogar_con_computadora'] = salida['activo_p189_9']
    salida['hogar_con_celular_internet'] = salida['activo_p189_11']
    salida['hogar_con_celular'] = pd.concat([
        salida['activo_p189_10'], salida['activo_p189_11']
    ], axis=1).max(axis=1, skipna=False)
    salida['hogar_con_dispositivo_digital'] = pd.concat([
        salida['activo_p189_9'], salida['activo_p189_11']
    ], axis=1).max(axis=1, skipna=False)

    return salida



def chequear_base(base: pd.DataFrame, auditoria: dict) -> None:
    verificar_unicidad(base, CLAVE_HOGAR, 'BASE FINAL HOGARES')
    if not (base['tamano_hogar'] >= 1).all():
        raise ValueError('Hay hogares sin miembros residentes validos')
    if 'prog_beca_18' in base.columns or 'P306_5' in base.columns:
        raise ValueError('Se encontro accidentalmente Beca 18')
    for n in NODOS:
        errores = base[n].notna() & ~base[n].isin([0, 1])
        if errores.any():
            raise ValueError(f'El nodo {n} presenta valores diferentes de 0/1/NA')
    n_eligibles = int(base[NODOS].notna().all(axis=1).sum())
    auditoria['validaciones'] = {
        'hogares_finales': int(len(base)),
        'nodos_ising': NODOS,
        'hogares_con_todos_nodos_observados': n_eligibles,
        'hogares_sin_todos_nodos': int(len(base) - n_eligibles),
        'hogares_sin_factor_positivo': int((~numerica(base['factor_hogar']).gt(0)).sum()),
        'hogares_con_activos_incompletos': int(base[['activo_p189_'+str(i) for i in range(1,18)]].isna().any(axis=1).sum()),
        'hogares_urbanos': int(base['area'].eq(1).sum()),
        'hogares_rurales': int(base['area'].eq(2).sum()),
        'jefes_sin_lengua_materna': int(base['lengua_materna_jefe'].isna().sum()),
    }

COLUMNAS_FINALES=[c for c in COLUMNAS_FINALES if c not in ('indice_activos_pca','n_integrantes_con_discapacidad')]+['activo_'+x.lower() for x in ACTIVOS]


def crear_base(entrada,salida):
    config=Configuracion(directorio_entrada=Path(entrada),directorio_salida=Path(salida),resfin_permitidos=(1,),solo_urbano=False,incluir_modulo_500=False)
    audit={'lectura':[],'filtros':{},'personas_objetivo':{},'uniones':{},'activos':{},'pca':{},'validaciones':{}}
    cap200=cargar_capitulo(config,'200',audit)
    persons,households=construir_padron(cap200,config,audit)
    cap300=cargar_capitulo(config,'300C',audit)
    programmes,people=agregar_cap300_a_hogares(persons,cap300,households,config,audit)
    households=households.merge(programmes,on=CLAVE_HOGAR,how='left',validate='one_to_one',sort=False)
    cap100=cargar_capitulo(config,'100E',audit)
    houses=construir_hogares_cap100e(cap100,audit)
    households=households.merge(houses,on=CLAVE_HOGAR,how='left',validate='one_to_one',sort=False)
    households['hogar_id']=households[CLAVE_HOGAR].astype('string').agg('_'.join,axis=1)
    chequear_base(households,audit)
    base=households[COLUMNAS_FINALES].copy().sort_values(CLAVE_HOGAR).reset_index(drop=True)
    for col in ['ANIO','MES','CONGLOMERADO','VIVIENDA','HOGAR','codigo_departamento']:
        base[col]=base[col].astype(str).str.replace(r'\.0+$','',regex=True)
    base['ANIO']=base.ANIO.str.zfill(4)
    base['MES']=base.MES.str.zfill(2)
    base['codigo_departamento']=base.codigo_departamento.str.zfill(2)
    output=Path(salida)
    output.mkdir(parents=True,exist_ok=True)
    base.to_csv(output/'ENAPRES2025_activos_originales.csv',index=False)
    base.to_pickle(output/'ENAPRES2025_activos_originales.pkl')
    with (output/'auditoria_unificacion.json').open('w',encoding='utf-8') as f:
        json.dump(audit,f,ensure_ascii=False,indent=2,default=str)
    return base


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-dir',required=True)
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    base=crear_base(args.raw_dir,args.out_dir)
    print(base.shape)

if __name__=='__main__':
    main()
