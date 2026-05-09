import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def entrenar_bosques():
    print(" EXPERIMENTOS CON RANDOM FOREST (10 CONFIGURACIONES) ")
    

    try:
        df = pd.read_csv('dataset_elpino_listo.csv', sep=';')
    except FileNotFoundError:
        print(" Error: No se encontró el dataset.")
        return
    

    print(" Preparando y codificando los datos")
    cols_codigos = [c for c in df.columns if '_cod' in c and c != 'target_grd']
    df_modelo = df.copy()
    
    le = LabelEncoder()
    for col in cols_codigos + ['target_grd']:
        df_modelo[col] = df_modelo[col].fillna('SIN_DATO').astype(str)
        df_modelo[col] = le.fit_transform(df_modelo[col])
        
    X = df_modelo.drop(columns=['target_grd'])
    y = df_modelo['target_grd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    configuraciones = [
        (10, 5),     # RF 1: 10 árboles, profundidad 5 
        (50, 5),     # RF 2: 50 árboles, profundidad 5
        (100, 5),    # RF 3: 100 árboles, profundidad 5
        (10, 10),    # RF 4: 10 árboles, profundidad 10
        (50, 10),    # RF 5: 50 árboles, profundidad 10
        (100, 10),   # RF 6: 100 árboles, profundidad 10
        (50, None),  # RF 7: 50 árboles, profundidad ilimitada
        (100, None), # RF 8: 100 árboles, profundidad ilimitada
        (200, 20),   # RF 9: 200 árboles, profundidad 20
        (200, None)  # RF 10: 200 árboles, profundidad ilimitada 
    ]
    
    resultados = []
    
    print("\n Iniciando entrenamiento secuencial de Bosques")

    for i, (arboles, profundidad) in enumerate(configuraciones, 1):
        nombre_modelo = f'RF {i}'
        prof_str = str(profundidad) if profundidad else 'Sin límite'
        print(f"Entrenando {nombre_modelo} (Árboles: {arboles}, Profundidad: {prof_str})", end=" ")
        
        inicio = time.time()
        

        modelo = RandomForestClassifier(
            n_estimators=arboles, 
            max_depth=profundidad, 
            class_weight='balanced', 
            random_state=42,
            n_jobs=-1 
        )
        
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        tiempo = time.time() - inicio
        
        print(f" Listo! ({tiempo:.1f}s)")
        
        resultados.append({
            'Modelo': nombre_modelo,
            'Árboles': arboles,
            'Profundidad': prof_str,
            'Accuracy': acc,
            'F1_Score': f1,
            'Tiempo_(s)': tiempo
        })
        
    df_resultados = pd.DataFrame(resultados)
    print("\n" + "="*85)
    print(" TABLA COMPARATIVA DE MODELOS RANDOM FOREST")
    print("="*85)
    print(df_resultados.to_string(index=False, float_format="{:.4f}".format))
    print("="*85)
    
    df_resultados.to_csv('tabla_comparativa_rf.csv', index=False, sep=';')

    print("\n Generando gráficos comparativos")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    df_resultados.plot(x='Modelo', y=['Accuracy', 'F1_Score'], kind='bar', ax=axes[0], colormap='magma')
    axes[0].set_title('Rendimiento: Bosque Aleatorio', fontsize=14)
    axes[0].set_ylabel('Puntuación (0 a 1)', fontsize=12)
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis='x', rotation=45)

    df_resultados.plot(x='Modelo', y='Tiempo_(s)', kind='line', marker='s', color='forestgreen', ax=axes[1], linewidth=2)
    axes[1].set_title('Costo Computacional (Tiempo)', fontsize=14)
    axes[1].set_ylabel('Segundos', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('comparativa_random_forest.png', dpi=300)
    print("¡Gráfico guardado como 'comparativa_random_forest.png'!")

if __name__ == "__main__":
    entrenar_bosques()