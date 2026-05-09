import pandas as pd
import numpy as np

def limpieza_total(ruta_archivo):
    print(" Iniciando Extracción Total de Datos ")
    df = pd.read_csv(ruta_archivo, sep=';', low_memory=False)
    
    df = df.replace('-', np.nan).replace('', np.nan)
    
    def extraer_codigo(texto):
        if pd.isna(texto):
            return "0" 
        return str(texto).split(' - ')[0].strip()

    cols_diag = [c for c in df.columns if 'Diag' in c]
    for col in cols_diag:
        df[f"{col}_cod"] = df[col].apply(extraer_codigo)
    
    cols_proc = [c for c in df.columns if 'Proced' in c]
    for col in cols_proc:
        df[f"{col}_cod"] = df[col].apply(extraer_codigo)

    if 'GRD' in df.columns:
        df['target_grd'] = df['GRD'].apply(extraer_codigo)

    df['Edad'] = pd.to_numeric(df['Edad en años'], errors='coerce').fillna(df['Edad en años'].median())

    df.loc[df['Edad'] > 105, 'Edad'] = 105
    
    df['Sexo_bin'] = df['Sexo (Desc)'].map({'Hombre': 0, 'Mujer': 1}).fillna(-1)

    columnas_finales = ['Edad', 'Sexo_bin', 'target_grd'] + \
                       [c for c in df.columns if c.endswith('_cod')]
    
    df_final = df[columnas_finales]
    
    return df_final

if __name__ == "__main__":
    archivo_entrada = 'dataset_elpino.csv'
    df_resultado = limpieza_total(archivo_entrada)

    df_resultado.to_csv('dataset_elpino_listo.csv', index=False, sep=';')
    
    print(f"\n✅ ¡Listo! Dataset generado con {df_resultado.shape[1]} columnas.")
    print(f"Registros procesados: {len(df_resultado)}")
    print("\nPrimeras columnas: ", list(df_resultado.columns[:5]))