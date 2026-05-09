import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generar_EDA():
    df = pd.read_csv('dataset_elpino_listo.csv', sep=';')
    
    sns.set_theme(style="whitegrid")

    print(" Análisis Exploratorio de Datos (EDA) ")
    print("\n Información del Dataset:")
    print(df.info())
    
    print("\n Estadísticas Descriptivas:")
    print(df.describe())
    
    print("\n Distribución de la Edad:")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Edad'], bins=30, kde=True)
    plt.title('Distribución de la Edad')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.show()
    
    print("\n Distribución del genero:")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sexo_bin', data=df)
    plt.title('Distribución del Sexo (0=Hombre, 1=Mujer)')
    plt.xlabel('Sexo')
    plt.ylabel('Frecuencia')
    plt.show()

    #Top 10
    plt.figure(figsize=(12, 6))
    top_grd = df['target_grd'].value_counts().head(10)
    sns.barplot(x=top_grd.index, y=top_grd.values, palette="viridis")
    plt.title('Top 10 Most Frequent DRGs - Hospital El Pino', fontsize=15)
    plt.xlabel('DRG Code', fontsize=12)
    plt.ylabel('Number of patients', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top_10_drg.png')
    print("grafico guardado como 'top_10_drg.png'")

    #districion de edad por genero
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Edad', hue='Sexo_bin', bins=30, kde=True, palette="muted")
    plt.title('Age distribution by Gender (0=Male, 1=Female)', fontsize=15)
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('age_distribution_by_gender.png')
    print("grafico guardado como 'age_distribution_by_gender.png'")

    #Top diagnosticos
    plt.figure(figsize=(12, 6))
    col_diag = 'Diag 01 Principal (cod+des)_cod'
    top_diag = df[col_diag].value_counts().head(10)
    sns.barplot(x=top_diag.index, y=top_diag.values, palette="magma")
    plt.title('Top 10 Primary Diagnoses', fontsize=15)   
    plt.xlabel('ICD-10 Code', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('top_diagnoses.png')
    print("grafico guardado como 'top_diagnoses.png'")

if __name__ == "__main__":
    try:
        generar_EDA()
        print("\n✅ Análisis Exploratorio de Datos completado exitosamente.")
    except Exception as e:
        print(f"\n❌ Error durante el EDA: {e}")

