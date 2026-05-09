import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def entrenar_MPL_simplificada():
    print("EXPERIMENTO MPL SIMPLIFICADO ")
    
    df = pd.read_csv('dataset_elpino_listo.csv', sep=';')
    
    print("1. Codificando todas las columnas")
    cols_codigos = [c for c in df.columns if '_cod' in c and c != 'target_grd']
    df_modelo = df.copy()
    le = LabelEncoder()
    for col in cols_codigos + ['target_grd']:
        df_modelo[col] = df_modelo[col].fillna('SIN_DATO').astype(str)
        df_modelo[col] = le.fit_transform(df_modelo[col])

    X = df_modelo.drop(columns=['target_grd'])
    y = df_modelo['target_grd']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    print("2. Comprimiendo el pensamiento con PCA")
    pca = PCA(n_components=15, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    

    arquitecturas = [(10,), (50,), (100,), (10, 5), (50, 25), 
                     (100, 50), (100, 100), (50, 25, 10), (100, 50, 25), (200, 100, 50)]
    resultados = []
    
    print("\nEntrenando 10 Redes Neuronales Simplificadas")
    for i, arq in enumerate(arquitecturas, 1):
        inicio = time.time()
        modelo = MLPClassifier(hidden_layer_sizes=arq, max_iter=200, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        tiempo = time.time() - inicio
        
        print(f"PCA-RN {i} (Nodos: {arq}) -> Exactitud: {acc*100:.2f}% | Tiempo: {tiempo:.1f}s")
        resultados.append({'Modelo': f'RN-Simplificada {i}', 'Nodos': str(arq), 'Accuracy': acc})

    pd.DataFrame(resultados).to_csv('resultados_mpl_simple.csv', index=False, sep=';')
    print("\n Resultados guardados.")

if __name__ == "__main__":
    entrenar_MPL_simplificada()