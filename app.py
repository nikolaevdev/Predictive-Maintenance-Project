import streamlit as st

PAGES = {
    "Проект": [
        st.Page("analysis_and_model.py", title="Анализ и модель"),
        st.Page("presentation.py",       title="Презентация проекта"),
    ]
}

page = st.navigation(PAGES, position="sidebar", expanded=True)
page.run()