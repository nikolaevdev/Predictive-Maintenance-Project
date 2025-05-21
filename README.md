# 🛠️ Predictive Maintenance Project

Демонстрационный *Streamlit*-сервис для **предиктивного обслуживания промышленного оборудования** на основе датасета *AI4I 2020* (UCI #601).

---

## 📑 Оглавление
- 🏃 Быстрый старт
- 🗂 Структура репозитория
- 🖱 Пользовательский сценарий
- 🏗 Архитектура и технологии
- 📦 Зависимости
- 🛣 Roadmap
- 📜 Лицензия

---

## 🏃 Быстрый старт
> **Требования**: Python ≥ 3.10, 64-битная ОС, 4 ГБ RAM (рекомендуется 8 ГБ).

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/<username>/predictive_maintenance_project.git
   cd predictive_maintenance_project
   ```

2. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Запустите приложение:
   ```bash
   streamlit run app.py
   ```

---

## 🗂 Структура репозитория
```
├── app.py                    # Точка входа
├── analysis_and_model.py     # Загрузка данных, пайплайн, метрики
├── presentation.py           # Автоматический отчёт и слайды
├── requirements.txt          # Список зависимостей
├── README.md                 # Документация
├── data/
│   └── predictive_maintenance.csv
├── video/
│   └── demo.mp4             # Демонстрация работы
└── docs/
    └── report.docx          # Теоретическое обоснование
```

---

## 🖱 Пользовательский сценарий
1. Запустите приложение: `streamlit run app.py`.
2. Загрузите CSV-файл с данными или используйте встроенный датасет.
3. Служебные столбцы (UDI, Product ID и др.) удаляются автоматически.
4. Выберите целевую переменную (по умолчанию — Machine failure).
5. Настройте алгоритм (LogisticRegression, RandomForest, XGBoost) и обучите модель — прогресс отображается индикатором.
6. Оцените результаты через метрики (ROC-AUC, Balanced Accuracy, матрица ошибок, ROC-кривая).
7. Сохраните модель в формате `.joblib` или откройте вкладку «Презентация» для автоматического отчёта.

---

## 🏗 Архитектура и технологии
| Компонент       | Реализация                                                                 |
|-----------------|---------------------------------------------------------------------------|
| **UI**          | Streamlit 1.45 с поддержкой многостраничности через `st.navigation`       |
| **Датасет**     | UCI AI4I 2020 (`ucimlrepo.fetch_ucirepo(id=601)`) или пользовательский CSV|
| **Пайплайн**    | `ColumnTransformer` → `StandardScaler`/`OneHotEncoder` → `SMOTE` → модель |
| **HPO**         | `HalvingGridSearchCV` (resource factor 3)                                 |
| **Алгоритмы**   | LogisticRegression, RandomForest, XGBoost                                 |
| **Метрики**     | ROC-AUC (основная), Balanced Accuracy, матрица ошибок                    |
| **Отчёт**       | Reveal.js-слайды + JSON-результаты в `st.session_state`                  |
| **Сериализация**| `joblib.dump(best_model, "best_model.joblib")`                           |

---

## 📦 Зависимости
Полный список библиотек указан в `requirements.txt`. Основные: Streamlit, Pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn, ucimlrepo.

---

## 🛣 Roadmap
- ⚙️ Настройка CI/CD через GitHub Actions с публикацией в GHCR.
- 📊 Добавление интерпретируемости моделей (SHAP, LIME) с интерактивной визуализацией.
- 🔁 Реализация MLOps для автоматического дообучения при накоплении данных.
- 📦 Интеграция MLflow для трекинга экспериментов.
- 🛠 Экспорт моделей в ONNX/TF-Lite для инференса на edge-устройствах.

---

## 📜 Лицензия
Проект распространяется под лицензией MIT.  
Датасет AI4I 2020 © UCI Machine Learning Repository.