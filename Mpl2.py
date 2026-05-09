#Para ejecutar en colab por problñemas de version python en mi pc


import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("  MODELO GRD — HOSPITAL EL PINO v2")
print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  TF Version: {tf.__version__}")
print("=" * 70)


# CONFIGURACIÓN
CONFIG = {
    "data_path"   : "/content/dataset_elpino_listo.csv",
    "sep"         : ";",
    "target_grd"  : "target_grd",

    # Columnas numéricas
    "cols_numeric": ["Edad", "Sexo_bin"],

    # Columnas categóricas
    "cols_cat": [
        "Diag 01 Principal (cod+des)_cod",
        "Diag 02 Secundario (cod+des)_cod",
        "Diag 03 Secundario (cod+des)_cod",
        "Diag 04 Secundario (cod+des)_cod",
        "Diag 05 Secundario (cod+des)_cod",
        "Diag 06 Secundario (cod+des)_cod",
        "Diag 07 Secundario (cod+des)_cod",
        "Diag 08 Secundario (cod+des)_cod",
        "Diag 09 Secundario (cod+des)_cod",
        "Diag 10 Secundario (cod+des)_cod",
        "Diag 11 Secundario (cod+des)_cod",
        "Diag 12 Secundario (cod+des)_cod",
        "Diag 13 Secundario (cod+des)_cod",
        "Diag 14 Secundario (cod+des)_cod",
        "Diag 15 Secundario (cod+des)_cod",
        "Diag 16 Secundario (cod+des)_cod",
        "Diag 17 Secundario (cod+des)_cod",
        "Diag 18 Secundario (cod+des)_cod",
        "Diag 19 Secundario (cod+des)_cod",
        "Diag 20 Secundario (cod+des)_cod",
        "Diag 21 Secundario (cod+des)_cod",
        "Diag 22 Secundario (cod+des)_cod",
        "Diag 23 Secundario (cod+des)_cod",
        "Diag 24 Secundario (cod+des)_cod",
        "Diag 25 Secundario (cod+des)_cod",
        "Diag 26 Secundario (cod+des)_cod",
        "Diag 27 Secundario (cod+des)_cod",
        "Diag 28 Secundario (cod+des)_cod",
        "Diag 29 Secundario (cod+des)_cod",
        "Diag 30 Secundario (cod+des)_cod",
        "Diag 31 Secundario (cod+des)_cod",
        "Diag 32 Secundario (cod+des)_cod",
        "Diag 33 Secundario (cod+des)_cod",
        "Diag 34 Secundario (cod+des)_cod",
        "Diag 35 Secundario (cod+des)_cod",
        "Proced 01 Principal (cod+des)_cod",
        "Proced 02 Secundario (cod+des)_cod",
        "Proced 03 Secundario (cod+des)_cod",
        "Proced 04 Secundario (cod+des)_cod",
        "Proced 05 Secundario (cod+des)_cod",
        "Proced 06 Secundario (cod+des)_cod",
        "Proced 07 Secundario (cod+des)_cod",
        "Proced 08 Secundario (cod+des)_cod",
        "Proced 09 Secundario (cod+des)_cod",
        "Proced 10 Secundario (cod+des)_cod",
        "Proced 11 Secundario (cod+des)_cod",
        "Proced 12 Secundario (cod+des)_cod",
        "Proced 13 Secundario (cod+des)_cod",
        "Proced 14 Secundario (cod+des)_cod",
        "Proced 15 Secundario (cod+des)_cod",
        "Proced 16 Secundario (cod+des)_cod",
        "Proced 17 Secundario (cod+des)_cod",
        "Proced 18 Secundario (cod+des)_cod",
        "Proced 19 Secundario (cod+des)_cod",
        "Proced 20 Secundario (cod+des)_cod",
        "Proced 21 Secundario (cod+des)_cod",
        "Proced 22 Secundario (cod+des)_cod",
        "Proced 23 Secundario (cod+des)_cod",
        "Proced 24 Secundario (cod+des)_cod",
        "Proced 25 Secundario (cod+des)_cod",
        "Proced 26 Secundario (cod+des)_cod",
        "Proced 27 Secundario (cod+des)_cod",
        "Proced 28 Secundario (cod+des)_cod",
        "Proced 29 Secundario (cod+des)_cod",
        "Proced 30 Secundario (cod+des)_cod",
    ],

    # Arquitectura
    "embedding_dim"  : 16,      
    "dense_units"    : [512, 256, 128, 64],
    "dropout_rate"   : 0.35,

    # Entrenamiento
    "batch_size"     : 512,
    "epochs"         : 150,
    "learning_rate"  : 3e-4,
    "test_size"      : 0.20,
    "val_size"       : 0.15,

    # Callbacks
    "es_patience"    : 15,
    "rlr_patience"   : 7,
    "rlr_factor"     : 0.5,
    "min_lr"         : 1e-6,
    "checkpoint"     : "mejor_modelo_grd.keras",
}


# 1. CARGA DE DATOS

print("\n[1/5] Cargando datos...")
df = pd.read_csv(CONFIG["data_path"], sep=CONFIG["sep"], low_memory=False)
print(f"  → Shape: {df.shape}")

# Filtrar solo columnas que existen
cols_cat = [c for c in CONFIG["cols_cat"] if c in df.columns]
cols_num = [c for c in CONFIG["cols_numeric"] if c in df.columns]
print(f"  → Columnas categóricas encontradas : {len(cols_cat)}")
print(f"  → Columnas numéricas encontradas   : {cols_num}")
print(f"  → Clases GRD únicas               : {df[CONFIG['target_grd']].nunique()}")


# 2. PREPROCESAMIENTO
print("\n[2/5] Preprocesando...")

for col in cols_cat:
    df[col] = df[col].fillna("MISSING").astype(str).str.strip().str.upper()

for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col].fillna(df[col].median(), inplace=True)

# Eliminar filas sin target
df.dropna(subset=[CONFIG["target_grd"]], inplace=True)
print(f"  → Shape limpio: {df.shape}")

# Encoding del target GRD
le_grd = LabelEncoder()
y_raw  = le_grd.fit_transform(df[CONFIG["target_grd"]].astype(str))
n_classes = len(le_grd.classes_)
y_ohe = tf.keras.utils.to_categorical(y_raw, num_classes=n_classes)
print(f"  → Clases GRD codificadas: {n_classes}")

# Encoding de features 
col_encoders   = {}
col_vocab_sizes = {}
X_cat_list     = []

for col in cols_cat:
    le  = LabelEncoder()
    enc = le.fit_transform(df[col])
    col_encoders[col]    = le
    col_vocab_sizes[col] = len(le.classes_) + 1   
    X_cat_list.append(enc.reshape(-1, 1))

# Estandarizar 
scaler  = StandardScaler()
X_num   = scaler.fit_transform(df[cols_num].values)
print(f"  → Features numéricas shape: {X_num.shape}")

# 3. SPLIT
print("\n[3/5] Dividiendo datos...")

counts     = pd.Series(y_raw).value_counts()
clases_ok  = counts[counts >= 2].index
mask_valid = np.isin(y_raw, clases_ok)

X_cat_list = [arr[mask_valid] for arr in X_cat_list]
X_num      = X_num[mask_valid]
y_ohe      = y_ohe[mask_valid]
y_raw      = y_raw[mask_valid]

print(f"  → Muestras tras filtrar clases únicas: {mask_valid.sum():,}")
print(f"  → Clases GRD restantes: {len(clases_ok)}")

n       = mask_valid.sum()
indices = np.arange(n)

idx_trainval, idx_test = train_test_split(
    indices, test_size=CONFIG["test_size"], random_state=SEED, stratify=y_raw
)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=CONFIG["val_size"],
    random_state=SEED, stratify=y_raw[idx_trainval]
)

print(f"  → Train: {len(idx_train):,}  |  Val: {len(idx_val):,}  |  Test: {len(idx_test):,}")

def make_inputs(idx):
    return [X_cat_list[i][idx] for i in range(len(cols_cat))] + [X_num[idx]]

X_train, X_val, X_test = make_inputs(idx_train), make_inputs(idx_val), make_inputs(idx_test)
y_train, y_val, y_test = y_ohe[idx_train], y_ohe[idx_val], y_ohe[idx_test]

# 4. MODELO
print("\n[4/5] Construyendo modelo...")

def build_model():
    cat_inputs    = []
    embed_outputs = []

    for i, col in enumerate(cols_cat):
        vocab   = col_vocab_sizes[col]
        emb_dim = min(CONFIG["embedding_dim"], max(4, vocab // 4))
        inp = Input(shape=(1,), name=f"in_{i}")
        emb = layers.Embedding(vocab, emb_dim, name=f"emb_{i}")(inp)
        flat = layers.Flatten()(emb)
        cat_inputs.append(inp)
        embed_outputs.append(flat)

    num_input = Input(shape=(X_num.shape[1],), name="in_num")
    x = layers.Concatenate()(embed_outputs + [num_input])

    for i, units in enumerate(CONFIG["dense_units"]):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(CONFIG["dropout_rate"])(x)

    out = layers.Dense(n_classes, activation="softmax", name="output_grd")(x)

    model = Model(inputs=cat_inputs + [num_input], outputs=out, name="GRD_ElPino")
    model.compile(
        optimizer=Adam(CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model()
model.summary()
print(f"\n  → Parámetros entrenables: {model.count_params():,}")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=CONFIG["es_patience"],
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=CONFIG["rlr_factor"],
                      patience=CONFIG["rlr_patience"], min_lr=CONFIG["min_lr"], verbose=1),
    ModelCheckpoint(CONFIG["checkpoint"], monitor="val_loss",
                    save_best_only=True, verbose=0),
]

# 5. ENTRENAMIENTO
print("\n[5/5] Entrenando...")
history = model.fit(
    x               = X_train,
    y               = y_train,
    validation_data = (X_val, y_val),
    epochs          = CONFIG["epochs"],
    batch_size      = CONFIG["batch_size"],
    callbacks       = callbacks,
    verbose         = 1,
)
# 6. EVALUACIÓN
print("\nEvaluando en Test Set...")
pred_probs = model.predict(X_test, batch_size=CONFIG["batch_size"], verbose=0)
pred_grd   = np.argmax(pred_probs, axis=1)
true_grd   = np.argmax(y_test,     axis=1)

acc  = accuracy_score(true_grd, pred_grd)
prec = precision_score(true_grd, pred_grd, average="macro", zero_division=0)
rec  = recall_score(true_grd, pred_grd,    average="macro", zero_division=0)

print("\n" + "=" * 70)
print("  RESULTADOS EN TEST SET — GRD")
print("=" * 70)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}  (macro)")
print(f"  Recall    : {rec:.4f}  (macro)")
print(f"\n  Baseline  :  37.30%  ← referencia a superar")
print(f"  Delta     : {(acc - 0.373)*100:+.2f}pp vs baseline")
print("=" * 70)

# Reporte top 20 GRDs más frecuentes
top20 = [idx for idx, _ in Counter(true_grd).most_common(20)]
mask  = np.isin(true_grd, top20)
print("\nReporte por clase (Top 20 GRDs más frecuentes en test):")
print(classification_report(
    true_grd[mask], pred_grd[mask],
    labels      = top20,
    target_names= [le_grd.classes_[i] for i in top20],
    zero_division= 0
))

epochs_run = len(history.history["loss"])
print(f"  Épocas entrenadas : {epochs_run}")
print(f"  Mejor val_loss    : {min(history.history['val_loss']):.4f}")
print(f"  Modelo guardado   : '{CONFIG['checkpoint']}'")
print("=" * 70)