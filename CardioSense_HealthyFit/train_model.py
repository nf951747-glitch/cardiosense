# train_model.py
# Run this ONCE to generate all model .pkl files and scaler.pkl
# Place heart_disease_dataset.csv in the same folder, then run:
#   python train_model.py

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

print("=" * 60)
print("  Heart Disease — Multi-Model Training Script")
print("=" * 60)

# ── [1/6] Load dataset ────────────────────────────────────────
df = pd.read_csv('heart_disease_dataset.csv')
print(f"\n[1/6] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── [2/6] KNN Imputation ──────────────────────────────────────
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(f"[2/6] KNN Imputation applied")

# ── [3/6] Features & target ───────────────────────────────────
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
    'smoking', 'diabetes', 'bmi'
]

X = df_imputed[FEATURE_ORDER]
y = df_imputed['heart_disease'].astype(int)

# ── [4/6] Train/Test split ────────────────────────────────────
# Use 25% test split for more reliable evaluation with larger dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── [5/6] StandardScaler ──────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"[3/6] StandardScaler fitted")

# ── [6/6] Train all 4 models with regularisation fixes ─────────
print(f"[4/6] Training all 4 ensemble models with anti-overfitting settings...\n")

# Random Forest: reduce depth, increase min_samples, add feature subsampling
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,            # was 10 → shallower trees prevent memorisation
    min_samples_leaf=10,    # was 5 → require more support per leaf
    min_samples_split=20,   # added: higher split threshold
    max_features='sqrt',    # only sqrt(features) considered at each split
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_sc, y_train)
print("  ✅ Random Forest trained")

# Gradient Boosting: shrink depth, lower n_estimators, add subsample/col-sample
gb_model = GradientBoostingClassifier(
    n_estimators=150,       # was 200
    learning_rate=0.05,     # was 0.1 → slower, more regularised
    max_depth=3,            # was 4 → shallower
    min_samples_leaf=10,    # added
    subsample=0.8,          # added: stochastic boosting reduces variance
    max_features='sqrt',    # added: feature subsampling per split
    random_state=42
)
gb_model.fit(X_train_sc, y_train)
print("  ✅ Gradient Boosting trained")

# AdaBoost: already healthy; keep conservatively
ada_model = AdaBoostClassifier(
    n_estimators=150,       # was 200
    learning_rate=0.5,
    random_state=42
)
ada_model.fit(X_train_sc, y_train)
print("  ✅ AdaBoost trained")

# Voting Classifier: uses the regularised sub-models above
voting_model = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            min_samples_split=20, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1)),
        ('gb',  GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            min_samples_leaf=10, subsample=0.8, max_features='sqrt',
            random_state=42)),
        ('ada', AdaBoostClassifier(
            n_estimators=150, learning_rate=0.5, random_state=42))
    ],
    voting='soft'
)
voting_model.fit(X_train_sc, y_train)
print("  ✅ Voting Classifier trained")

# ── Evaluate & Overfitting/Underfitting Check ─────────────────
print("\n" + "=" * 60)
print("  MODEL EVALUATION + OVERFITTING CHECK")
print("=" * 60)
print(f"  {'Model':<22} {'Train Acc':>10} {'Test Acc':>10}  {'Status':>14}")
print("-" * 62)

model_configs = [
    ("Random Forest",     rf_model),
    ("Gradient Boosting", gb_model),
    ("AdaBoost",          ada_model),
    ("Voting Classifier", voting_model),
]

metrics_store = {}
for name, mdl in model_configs:
    train_acc = mdl.score(X_train_sc, y_train) * 100
    test_acc  = mdl.score(X_test_sc,  y_test)  * 100
    gap       = train_acc - test_acc
    if gap > 10:
        status = "⚠️  Overfitting"
    elif test_acc < 60:
        status = "⚠️  Underfitting"
    else:
        status = "✅ Healthy"
    print(f"  {name:<22} {train_acc:>9.2f}% {test_acc:>9.2f}%  {status}")

    # Detailed metrics
    preds = mdl.predict(X_test_sc)
    proba = mdl.predict_proba(X_test_sc)[:, 1]
    acc   = accuracy_score(y_test, preds) * 100
    prec  = precision_score(y_test, preds, zero_division=0) * 100
    rec   = recall_score(y_test, preds) * 100
    f1    = f1_score(y_test, preds) * 100
    roc   = roc_auc_score(y_test, proba) * 100

    gap_flag = "overfitting" if gap > 10 else ("underfitting" if test_acc < 60 else "healthy")
    cm = confusion_matrix(y_test, preds)
    tn_v, fp_v, fn_v, tp_v = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    metrics_store[name] = {
        "accuracy":  round(acc, 1),
        "precision": round(prec, 1),
        "recall":    round(rec, 1),
        "f1":        round(f1, 1),
        "roc_auc":   round(roc, 1),
        "train_acc": round(train_acc, 1),
        "test_acc":  round(test_acc, 1),
        "fit_status": gap_flag,
        "cm_tp": int(tp_v),
        "cm_tn": int(tn_v),
        "cm_fp": int(fp_v),
        "cm_fn": int(fn_v),
    }

print("\nDetailed Test-Set Metrics:")
print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>10}")
print("-" * 74)
for name, m in metrics_store.items():
    print(f"  {name:<22} {m['accuracy']:>9.1f}% {m['precision']:>9.1f}% {m['recall']:>7.1f}% {m['f1']:>7.1f}% {m['roc_auc']:>9.1f}%")

# ── Cross-validation on best model ────────────────────────────
print("\n[CV] 5-Fold Stratified Cross-Validation on Voting Classifier:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(voting_model, X_train_sc, y_train, cv=skf, scoring='accuracy')
print(f"     CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Sample Prediction Sanity Check ────────────────────────────
sample_data = {
    "high_risk_patient": {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
        "smoking": 1, "diabetes": 1, "bmi": 30.5
    },
    "low_risk_patient": {
        "age": 35, "sex": 0, "cp": 0, "trestbps": 110, "chol": 175,
        "fbs": 0, "restecg": 0, "thalach": 165, "exang": 0,
        "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2,
        "smoking": 0, "diabetes": 0, "bmi": 22.1
    }
}
print("\n[Sample Prediction Check — Sanity Test]")
for label, patient in sample_data.items():
    arr = pd.DataFrame([[patient[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    arr_sc = scaler.transform(arr)
    pred  = voting_model.predict(arr_sc)[0]
    prob  = voting_model.predict_proba(arr_sc)[0][1] * 100
    print(f"  {label:<25} → Prediction: {'HIGH RISK' if pred==1 else 'LOW RISK'} ({prob:.1f}% risk)")

# ── Save everything ───────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(rf_model,     'model/rf_model.pkl')
joblib.dump(gb_model,     'model/gb_model.pkl')
joblib.dump(ada_model,    'model/ada_model.pkl')
joblib.dump(voting_model, 'model/heart_model.pkl')
joblib.dump(scaler,       'model/scaler.pkl')

with open('model/metrics.json', 'w') as f:
    json.dump(metrics_store, f, indent=2)

print("\n[5/6] Saved → model/")
print("      rf_model.pkl | gb_model.pkl | ada_model.pkl | heart_model.pkl | scaler.pkl | metrics.json")
print("\n✅ Done! Run:  python app.py")
