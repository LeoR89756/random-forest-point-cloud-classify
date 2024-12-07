# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:02:35 2024

@author: leona
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

# === Configurazione dei percorsi ===
# Configura i percorsi dei file di input e output
input_folder = "./input"
output_folder = "./output"

train_file = os.path.join(input_folder, "train.txt")
eval_file = os.path.join(input_folder, "unc_eva.txt")
gt_file = os.path.join(input_folder, "gt.txt")
prop_file = os.path.join(input_folder, "unc_prop.txt")

model_file = os.path.join(output_folder, "trained_model.pkl")
classified_eval_file = os.path.join(output_folder, "c_eva.txt")
conf_matrix_file = os.path.join(output_folder, "matrix.txt")
classified_prop_file = os.path.join(output_folder, "c_prop.txt")

# === Funzione per leggere i file ===
def read_txt_file(file_path, col_names):
    return pd.read_csv(file_path, sep=" ", header=None, names=col_names)

# === Lettura dei file ===
train_columns = ["x", "y", "z", "r", "g", "b", "Nx", "Ny", "Nz", "SF2", "SF1"]
eval_columns = ["x", "y", "z", "r", "g", "b", "Nx", "Ny", "Nz", "SF2"]
gt_columns = ["x", "y", "z", "r", "g", "b", "Nx", "Ny", "Nz", "SF2", "SF1"]
prop_columns = ["x", "y", "z", "r", "g", "b", "Nx", "Ny", "Nz", "SF2"]

train_data = read_txt_file(train_file, train_columns)
eval_data = read_txt_file(eval_file, eval_columns)
gt_data = read_txt_file(gt_file, gt_columns)
prop_data = read_txt_file(prop_file, prop_columns)

# === Preparazione dei dati per il training ===
X_train = train_data[["x", "y", "z", "r", "g", "b", "Nx", "Ny", "Nz", "SF2"]]
y_train = train_data["SF1"]

# === Creazione e addestramento del modello Random Forest ===
rf_model = RandomForestClassifier(
    n_estimators=200,         # Numero di alberi
    max_depth=20,             # Profondità massima
    class_weight='balanced',  # Bilanciamento delle classi
    random_state=42,          # Per riproducibilità
    n_jobs=-1                 # Usa tutti i core disponibili
)

rf_model.fit(X_train, y_train)
joblib.dump(rf_model, model_file)  # Salvataggio del modello
print(f"Modello addestrato e salvato in: {model_file}")

# === Classificazione del file unc_eva.txt ===
X_eval = eval_data[["x", "y", "z", "r", "g", "b", "Nx", "Ny", "Nz", "SF2"]]
eval_data["SF1"] = rf_model.predict(X_eval)
eval_data.to_csv(classified_eval_file, sep=" ", header=False, index=False)
print(f"Dati classificati salvati in: {classified_eval_file}")

# === Calcolo della matrice di confusione e del report ===
gt_labels = gt_data["SF1"]
pred_labels = eval_data["SF1"]

conf_matrix = confusion_matrix(gt_labels, pred_labels)
report = classification_report(gt_labels, pred_labels)

os.makedirs(output_folder, exist_ok=True)
with open(conf_matrix_file, "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Matrice di confusione e report salvati in: {conf_matrix_file}")

# === Classificazione del file unc_prop.txt ===
X_prop = prop_data[["x", "y", "z", "r", "g", "b", "Nx", "Ny", "Nz", "SF2"]]
prop_data["SF1"] = rf_model.predict(X_prop)
prop_data.to_csv(classified_prop_file, sep=" ", header=False, index=False)
print(f"Classificazione completata per unc_prop.txt. File salvato in: {classified_prop_file}")
