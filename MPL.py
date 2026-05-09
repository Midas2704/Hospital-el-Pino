import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

def entrenar_MPL():
    print("MPL")
    

    try:
        df_principal = pd.read_csv('dataset_elpino_listo.csv', sep=';')
        try:
            df_severidad = pd.read_csv('Severidad GRD.csv', sep=';')
            df = pd.merge(df_principal, df_severidad, left_on='target_grd', right_on='GRD', how='left')
        except FileNotFoundError:
            print("Generando Severidad")
            df = df_principal.copy()
            cols_cod = [c for c in df.columns if '_cod' in c and c != 'target_grd']
            complejidad = ~df[cols_cod].astype(str).isin(['0', 'SIN_DATO']).sum(axis=1)
            df['Nivel_Severidad'] = 'Baja'
            df.loc[complejidad > 3, 'Nivel_Severidad'] = 'Media'
            df.loc[(complejidad > 7) | (df['Edad'] > 75), 'Nivel_Severidad'] = 'Alta'
    except FileNotFoundError:
        print(" Error: Dataset principal no encontrado."); return

  
    cols_codigos = [c for c in df.columns if '_cod' in c and c != 'target_grd']
    df_modelo = df.copy()
    
    le = LabelEncoder()
    for col in cols_codigos:
        df_modelo[col] = le.fit_transform(df_modelo[col].fillna('SIN_DATO').astype(str))
        
    df_modelo['target_grd'] = le.fit_transform(df_modelo['target_grd'].astype(str))
    df_modelo['Nivel_Severidad'] = le.fit_transform(df_modelo['Nivel_Severidad'].astype(str))

    X = df_modelo[cols_codigos + ['Edad']]
    y = df_modelo[['target_grd', 'Nivel_Severidad']] 
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
   
    arquitecturas = [
            # --- LOS 10 MODELOS BASE (Línea de control) ---
            (1,), (50,), (100,), (10, 5), (50, 25), 
            (100, 50), (100, 100), (50, 25, 10), (100, 50, 25), (200, 100, 50),
            
            #variaciones post expoerimentales
            (192, 96, 48, 24, 12),      
            (96, 192, 96, 48, 24),      
            
            (100, 200, 100, 50, 25),     
            (128, 384, 128, 64, 32),    
            (64, 256, 64, 32, 16),      
            (128, 256, 128, 64),         
            

            (256, 128, 64, 32, 16),   
            (300, 150, 75, 35, 15),  
            (180, 90, 45, 20, 10),     
            (64, 128, 256, 128, 64, 32), 
            (128, 128, 64, 32, 16),      

            (512, 256, 500, 500, 512, 256, 128, 50, 100, 40)
        ]
    resultados = []
    
    print(f"\n Entrenando {len(arquitecturas)} modelos")
    for i, arq in enumerate(arquitecturas, 1):
        nombre = "Experimento" if i == 35 else f" {i}"
        inicio = time.time()
        
        base_nn = MLPClassifier(
            hidden_layer_sizes=arq, 
            max_iter=300, 
            early_stopping=True,
            learning_rate='adaptive',
            random_state=42
        )
        modelo = MultiOutputClassifier(base_nn, n_jobs=-1)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        acc_grd = accuracy_score(y_test['target_grd'], y_pred[:, 0])
        pre_grd = precision_score(y_test['target_grd'], y_pred[:, 0], average='macro', zero_division=0)
        rec_grd = recall_score(y_test['target_grd'], y_pred[:, 0], average='macro', zero_division=0)
        

        acc_sev = accuracy_score(y_test['Nivel_Severidad'], y_pred[:, 1])
        pre_sev = precision_score(y_test['Nivel_Severidad'], y_pred[:, 1], average='macro', zero_division=0)
        rec_sev = recall_score(y_test['Nivel_Severidad'], y_pred[:, 1], average='macro', zero_division=0)
        
        tiempo = time.time() - inicio
        print(f"✅ {nombre} -> GRD Acc: {acc_grd*100:.1f}% | Sev Acc: {acc_sev*100:.1f}% | {tiempo:.1f}s")
        
        resultados.append({
            'Modelo': nombre, 'Nodos': str(arq),
            'Acc_GRD': acc_grd, 'Pre_GRD': pre_grd, 'Rec_GRD': rec_grd,
            'Acc_Sev': acc_sev, 'Pre_Sev': pre_sev, 'Rec_Sev': rec_sev,
            'Tiempo': tiempo
        })

    pd.DataFrame(resultados).to_csv('resultados_MPL.csv', index=False, sep=';')
    print("\nArchivo guardado 'resultados_MPL.csv' ")

if __name__ == "__main__":
    entrenar_MPL()