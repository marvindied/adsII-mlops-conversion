import pandas as pd
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1. Daten laden (Wir nutzen unseren eingefrorenen Stand!)
# WICHTIG: Achte darauf, dass der Pfad stimmt, je nachdem von wo du das Skript aufrufst.
data_path = os.path.join(os.path.dirname(__file__), '../data/processed_online_shoppers.csv')
print(f"Lade Daten von: {data_path}")
df = pd.read_csv(data_path)

# 2. Features (X) und Target (y) trennen
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# 3. Train-Test-Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# MLflow Experiment setzen
mlflow.set_experiment("Conversion_Prediction_Project")

print("Starte MLflow Experimente...")

# --- RUN 1: BASELINE MODELL (Logistische Regression) ---
with mlflow.start_run(run_name="Baseline_LogReg"):
    print("Trainiere Baseline Modell (Logistische Regression)...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Vorhersagen treffen
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    
    # Metriken berechnen
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
    
    # MLflow Logging
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", acc_lr)
    mlflow.log_metric("f1_score", f1_lr)
    mlflow.log_metric("roc_auc", auc_lr)
    mlflow.sklearn.log_model(lr_model, "model")

# --- RUN 2: CHAMPION MODELL (XGBoost) ---
with mlflow.start_run(run_name="Champion_XGBoost"):
    print("Trainiere Champion Modell (XGBoost)...")
    # scale_pos_weight hilft bei imbalancierten Daten (85% False / 15% True -> Ratio ca. 5.5)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, scale_pos_weight=5.5, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Vorhersagen treffen
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Metriken berechnen
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    
    # MLflow Logging
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("accuracy", acc_xgb)
    mlflow.log_metric("f1_score", f1_xgb)
    mlflow.log_metric("roc_auc", auc_xgb)
    mlflow.xgboost.log_model(xgb_model, "model")

print("Training abgeschlossen. Du kannst die Ergebnisse nun in MLflow ansehen!")