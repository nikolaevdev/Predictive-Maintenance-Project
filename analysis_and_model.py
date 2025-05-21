"""
Страница «Анализ и модель».

• Загрузка данных (UCI AI4I 2020 или пользовательский CSV)
• Предобработка ColumnTransformer → SMOTE
• Подбор гиперпараметров HalvingGridSearchCV
• Метрики + графики
• Передача результатов в st.session_state для автоматического отчёта
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# --- scikit-learn / imbalanced-learn -------------------------
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------- Настройка страницы --------------------------
st.set_page_config(page_title="Анализ и модель", layout="wide")
st.title("Анализ данных и обучение модели")

# =============================================================
# 1. Загрузка данных
# =============================================================
@st.cache_data(show_spinner="Загружаем датасет…")
def load_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        ds = fetch_ucirepo(id=601)                   # AI4I 2020
        return pd.concat([ds.data.features, ds.data.targets], axis=1)
    return pd.read_csv(path)

user_csv = st.file_uploader("💾 Загрузите свой CSV (необязательно)", type="csv")
df = load_data(user_csv)

st.subheader("Фрагмент выборки")
st.dataframe(df.head())

# =============================================================
# 2. Предобработка (безопасное удаление «служебных» колонок)
# =============================================================
cols_to_drop = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
df = df.drop(columns=cols_to_drop, errors="ignore")   # errors='ignore'

# Целевая переменная — из датасета или выбор пользователя
DEFAULT_TARGET = "Machine failure"
TARGET = DEFAULT_TARGET if DEFAULT_TARGET in df.columns else st.sidebar.selectbox(
    "Целевая переменная", df.columns, index=len(df.columns)-1
)

# Категориальные / числовые признаки
categorical = [c for c in ["Type"] if c in df.columns]
numeric = [c for c in df.columns if c not in categorical + [TARGET]]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(),                     numeric),
    ],
    verbose_feature_names_out=False,
)

# =============================================================
# 3. Выбор алгоритма и сеток гиперпараметров
# =============================================================
model_name = st.sidebar.selectbox(
    "Алгоритм",
    ("LogisticRegression", "RandomForest", "XGBoost"),
)

def make_estimator(name: str):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, n_jobs=-1, solver="lbfgs")
    if name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            colsample_bytree=0.8, subsample=0.8,
            random_state=42, n_jobs=-1,
        )
    raise ValueError("Неизвестный алгоритм")

param_grids = {
    "LogisticRegression": {
        "clf__C": np.logspace(-3, 2, 6),
        "clf__class_weight": ["balanced"],
    },
    "RandomForest": {
        "clf__n_estimators": [200, 500],
        "clf__max_depth":    [None, 10, 20],
        "clf__class_weight": ["balanced"],
    },
    "XGBoost": {
        "clf__n_estimators":     [300, 500],
        "clf__learning_rate":    [0.03, 0.1],
        "clf__max_depth":        [3, 6],
        "clf__scale_pos_weight": [10],
    },
}

# =============================================================
# 4. Pipeline
# =============================================================

pipe = ImbPipeline(
    steps=[
        ("prep",  preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf",   make_estimator(model_name)),
    ]
)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

with st.spinner("Обучаем модель…"):
    search = HalvingGridSearchCV(
        pipe,
        param_grid=param_grids[model_name],
        scoring="roc_auc",
        factor=3,
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

best_model = search.best_estimator_
joblib.dump(best_model, "best_model.joblib")

st.success("✅ Обучение завершено")

# =============================================================
# 5. Метрики и визуализация
# =============================================================
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)
bal_acc = balanced_accuracy_score(y_test, y_pred)

st.subheader("Метрики")
m1, m2 = st.columns(2)
m1.metric("ROC-AUC", f"{roc_auc:.3f}")
m2.metric("Balanced accuracy", f"{bal_acc:.3f}")

with st.expander("Classification report"):
    st.code(classification_report(y_test, y_pred), language="text")

fig_cm, ax_cm = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Предсказание")
ax_cm.set_ylabel("Истина")
st.pyplot(fig_cm)

fig_roc, ax_roc = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc)
ax_roc.set_title("ROC-кривая")
st.pyplot(fig_roc)

# =============================================================
# 6. Сохраняем результаты для отчёта
# =============================================================
st.session_state["report"] = {
    "model":       model_name,
    "best_params": search.best_params_,
    "roc_auc":     roc_auc,
    "bal_acc":     bal_acc,
}
