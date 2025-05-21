# presentation.py
"""
Страница «Презентация проекта».
▪ Автоматизированный отчёт (метрики + гиперпараметры).
▪ Интерактивные слайды Reveal.js с требованиями ТЗ.
"""
from __future__ import annotations
import streamlit as st
import reveal_slides as rs

# ------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------
st.set_page_config(page_title="Презентация проекта", layout="centered")
st.title("Отчёт и презентация проекта")

# ------------------------------------------------------------
# 1. Автоматизированный отчёт
# ------------------------------------------------------------
if "report" not in st.session_state:
    st.warning("⚠️ Сначала обучите модель на вкладке «Анализ и модель».")
    st.stop()

rep = st.session_state["report"]

st.subheader("Метрики модели")
c1, c2 = st.columns(2)
c1.metric("ROC-AUC",           f"{rep['roc_auc']:.3f}")
c2.metric("Balanced accuracy", f"{rep['bal_acc']:.3f}")

st.subheader("Алгоритм")
st.write(f"**{rep['model']}**")

with st.expander("Лучшие гиперпараметры"):
    st.json(rep["best_params"])

st.markdown("---")   # разделитель между отчётом и слайдами

# ------------------------------------------------------------
# 2. Слайды Reveal.js
# ------------------------------------------------------------
slides_template = """
# Предиктивное обслуживание  
---

## Цель и постановка задачи  
* Бинарная классификация **Machine failure** (0 — нет отказа, 1 — отказ).  
* Датасет **AI4I 2020** — 10 000 записей × 14 признаков.  
---

## Структура проекта (по ТЗ)  
* `app.py` — точка входа  
* `analysis_and_model.py` — обучение и метрики  
* `presentation.py` — презентация / отчёт  
* `requirements.txt`, `README.md`, `data/`, `video/`  
* Многостраничность через `st.navigation` + `st.Page`  
---

## Этапы работы  
1. Загрузка / предобработка данных  
2. Разделение 80 / 20  
3. Балансировка SMOTE  
4. Подбор гиперпараметров **HalvingGridSearchCV**  
5. Оценка метрик  
6. Онлайн-предсказание в интерфейсе  
---

## Итоги модели  
* Алгоритм: **{model}**  
* ROC-AUC ≈ **{roc:.3f}**  
* Balanced acc. ≈ **{bal:.3f}**  
---

## Требования сдачи  
* `requirements.txt` — все зависимости  
* `README.md` — описание и инструкции  
* Видео-демо `video/demo.mp4`  
* DOCX-отчёт с обоснованием решений  
* Рабочее Streamlit-приложение на «чистой» среде  
---

## Перспективы  
* CI/CD (Docker + MLflow)  
* Explainability (SHAP / LIME)  
* Интеграция в SCADA / APM
"""

slides_md = slides_template.format(
    model=rep["model"],
    roc=rep["roc_auc"],
    bal=rep["bal_acc"],
)

# Параметры презентации через сайдбар
with st.sidebar:
    st.header("Настройки слайдов")
    theme      = st.selectbox("Тема", ["simple", "black", "white", "night", "beige", "serif"])
    height     = st.slider("Высота", 400, 900, 600)
    transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])

rs.slides(
    slides_md,
    height=height,
    theme=theme,
    config={"transition": transition},
    markdown_props={"data-separator-vertical": "^---$"},
)
