import pandas as pd
import numpy as np

def realizar_auditoria(ruta_archivo):
    df = pd.read_csv(ruta_archivo, sep=';')
    
    print(" REPORTE DE AUDITORÍA DE DATOS ML ")
    
    duplicados = df.duplicated().sum()
    print(f"\n1. Registros duplicados: {duplicados}")
    if duplicados > 0:
        print(" Sugerencia: Evalúa si son reingresos reales o errores de carga.")

    print("\n2. Densidad de Diagnósticos/Procedimientos:")
    cols_codigos = [c for c in df.columns if '_cod' in c and c != 'target_grd']
    df['conteo_info'] = (df[cols_codigos] != "SIN_DATO").sum(axis=1)
    
    print(f"   - Promedio de códigos por paciente: {df['conteo_info'].mean():.2f}")
    print(f"   - Pacientes con CERO códigos: {len(df[df['conteo_info'] == 0])}")

    print("\n3. Integridad de la Meta (target_grd):")
    if "SIN_DATO" in df['target_grd'].values:
        n_invalidos = (df['target_grd'] == "SIN_DATO").sum()
        print(f"   - ❌ ERROR: Hay {n_invalidos} filas sin GRD. ¡Debes eliminarlas!")
    else:
        print("   - ✅ Todos los registros tienen un GRD asignado.")

    print("\n4. Consistencia de Edad:")
    bebes = len(df[df['Edad'] < 1])
    ancianos = len(df[df['Edad'] > 100])
    print(f"   - Recién nacidos (edad 0): {bebes}")
    print(f"   - Centenarios (edad > 100): {ancianos}")

    print("\n5. Cardinalidad (Top 5 Diagnósticos Principales):")
    diag_prin = "Diag 01 Principal (cod+des)_cod"
    if diag_prin in df.columns:
        print(df[diag_prin].value_counts().head(5))

    print("\n6. Columnas sin información (Constantes):")
    for col in df.columns:
        if df[col].nunique() <= 1:
            print(f"   -  La columna '{col}' solo tiene un valor.")

if __name__ == "__main__":
    realizar_auditoria('dataset_elpino_listo.csv')