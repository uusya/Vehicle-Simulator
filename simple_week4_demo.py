# examples/simple_week4_demo.py
"""
Упрощенная демонстрация для недели 4 - без сложных импортов
Запуск: streamlit run simple_week4_demo.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def simple_simulator():
    """Упрощенный симулятор для демонстрации"""
    
    st.set_page_config(
        page_title="🏎️ Упрощенный симулятор управляемости",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title("🏎️ Упрощенный симулятор управляемости автомобиля")
    st.markdown("---")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Параметры")
        
        # Простые параметры
        speed = st.slider("Скорость (м/с)", 10, 30, 20)
        steer_angle = st.slider("Угол поворота (°)", 1, 20, 5)
        simulation_time = st.slider("Время симуляции (с)", 5, 20, 10)
        
        if st.button("🚀 Запустить симуляцию"):
            st.session_state.run_simulation = True
    
    # Основная область
    if st.session_state.get('run_simulation', False):
        show_simulation_results(speed, np.radians(steer_angle), simulation_time)
    else:
        show_welcome_screen()

def show_welcome_screen():
    """Экран приветствия"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Демонстрация симулятора
        
        Этот упрощенный симулятор показывает базовые принципы работы 
        модели управляемости автомобиля.
        
        **Для начала работы:**
        1. Настройте параметры в боковой панели
        2. Нажмите «Запустить симуляцию»
        3. Наслаждайтесь результатами!
        """)
    
    with col2:
        # Простая схема автомобиля
        fig = go.Figure()
        fig.add_shape(type="rect", x0=-2, y0=-1, x1=2, y1=1, 
                     line=dict(color="blue", width=2), fillcolor="lightblue", opacity=0.5)
        fig.update_layout(
            title="Схема автомобиля",
            xaxis=dict(range=[-3, 3], showticklabels=False),
            yaxis=dict(range=[-2, 2], showticklabels=False),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def show_simulation_results(speed, steer_angle, sim_time):
    """Показать результаты симуляции"""
    st.header("📊 Результаты симуляции")
    
    # Генерация демо-данных (в реальном проекте здесь будет вызов модели)
    time = np.linspace(0, sim_time, 100)
    
    # Демо-траектория (круг)
    theta = np.linspace(0, 2*np.pi, 100)
    X = speed * time * np.cos(steer_angle * time)
    Y = speed * time * np.sin(steer_angle * time)
    
    # Демо-параметры
    beta = 0.1 * np.sin(2 * np.pi * time / sim_time)  # угол скольжения
    r = steer_angle * np.cos(2 * np.pi * time / sim_time)  # угловая скорость
    
    # График траектории
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=X, y=Y, mode='lines', name='Траектория',
                             line=dict(color='blue', width=3)))
    fig1.update_layout(
        title='Траектория движения автомобиля',
        xaxis_title='X координата, м',
        yaxis_title='Y координата, м',
        height=400
    )
    
    # График параметров
    fig2 = make_subplots(rows=2, cols=1, subplot_titles=('Угол скольжения β', 'Угловая скорость r'))
    
    fig2.add_trace(
        go.Scatter(x=time, y=np.degrees(beta), name='β', line=dict(color='green')),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(x=time, y=np.degrees(r), name='r', line=dict(color='red')),
        row=2, col=1
    )
    
    fig2.update_layout(height=500, showlegend=True)
    fig2.update_xaxes(title_text="Время, с", row=2, col=1)
    fig2.update_yaxes(title_text="β, °", row=1, col=1)
    fig2.update_yaxes(title_text="r, °/с", row=2, col=1)
    
    # Отображение графиков
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Макс. угол скольжения", f"{np.max(np.abs(np.degrees(beta))):.2f}°")
    with col2:
        st.metric("Макс. угловая скорость", f"{np.max(np.abs(np.degrees(r))):.2f}°/с")
    with col3:
        st.metric("Пройденное расстояние", f"{np.sqrt(X[-1]**2 + Y[-1]**2):.1f} м")

if __name__ == "__main__":
    simple_simulator()