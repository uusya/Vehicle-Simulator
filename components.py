# src/web_interface/components.py
import streamlit as st
import plotly.graph_objects as go

def render_parameter_card(title, value, unit, delta=None):
    """Отрисовка карточки с параметром"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.metric(title, f"{value} {unit}", delta=delta)
    
    with col2:
        # Можно добавить иконки или дополнительную информацию
        pass

def create_tire_characteristics_plot(tire_models):
    """Создание графика характеристик шин"""
    fig = go.Figure()
    
    slip_angles = np.linspace(-0.3, 0.3, 100)
    
    for name, model in tire_models.items():
        forces = model.get_characteristics(slip_angles)
        fig.add_trace(go.Scatter(
            x=np.degrees(slip_angles), y=[f/1000 for f in forces],
            name=name, mode='lines'
        ))
    
    fig.update_layout(
        title="Характеристики моделей шин",
        xaxis_title="Угол увода, °",
        yaxis_title="Боковая сила, кН",
        height=400
    )
    
    return fig

def render_maneuver_info(maneuver_name):
    """Отображение информации о маневре"""
    maneuver_descriptions = {
        "Шаг рулем 5°": "Резкий поворот руля на 5 градусов для анализа переходных процессов",
        "Шаг рулем 10°": "Более агрессивный поворот для исследования предельных режимов",
        "Переставка": "Синусоидальное руление для анализа поведения при смене полосы",
        "Двойная переставка": "Сложный маневр уклонения с возвратом на исходную полосу", 
        "Постоянный радиус": "Движение по кругу для исследования поворачиваемости"
    }
    
    description = maneuver_descriptions.get(maneuver_name, "Описание маневра")
    st.info(f"**{maneuver_name}**: {description}")