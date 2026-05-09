import pandas as pd

def generar_estadisticas():
    df = pd.read_csv('dataset_elpino_listo.csv', sep=';')

    print("Analisis del data set: {archivo}")

    #Edad
    print("\n Estadísticas de la Edad:")  
    stats_edad = {
        "Media (Mean)": df['Edad'].mean(),
        "Mediana (Median)": df['Edad'].median(),
        "Moda (Mode)": df['Edad'].mode()[0],
        "Desviación Estándar (Std Dev)": df['Edad'].std(),
        "Mínimo (Min)": df['Edad'].min(),
        "Máximo (Max)": df['Edad'].max()

    }

    for k, v in stats_edad.items():
        print(f"{k}: {v:.2f}")

    #Sexo
    print("\n Distribución del Sexo:")
    conteo_sexo = df['Sexo_bin'].value_counts(normalize=True) * 100
    print(f"Mujeres: {conteo_sexo.get(1, 0):.2f}%")
    print(f"Hombres: {conteo_sexo.get(0, 0):.2f}%")

    #DRG
    print("\n Distribución de DRGs:")
    top_grd = df['target_grd'].value_counts().head(5)
    print(top_grd)

    #Complejidad
    cols_codigos = [c for c in df.columns if '_cod' in c and c != 'target_grd']
    d_vacio = ["0", "SIN_DATO", "nan", "NaN", "None", "", " 0.0", "0.0", "0.00"]
    c_vacio = ~df[cols_codigos].astype(str).isin(d_vacio)
    conteo_por_paciente = c_vacio.sum(axis=1)

    print("\n Complejidad (Número de códigos por paciente):")
    print(f"promedio diagnostico/procedimiento por paciente: {conteo_por_paciente.mean():.2f}")
    print(f"pacientes sin diagnosticos/procedimientos: {(conteo_por_paciente == 0).sum()} ({(conteo_por_paciente == 0).mean() * 100:.2f}%)")
    print(f"maximo de codigos en un solo paciente: {conteo_por_paciente.max()}")

if __name__ == "__main__":
    try:
        generar_estadisticas()
    except FileNotFoundError:
            print("Error: El archivo 'dataset_elpino_listo.csv' no se encuentra. Asegúrate de ejecutar primero el script de limpieza para generar el dataset.")