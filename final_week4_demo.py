# examples/final_week4_demo.py
"""
Финальная рабочая версия демо для недели 4
Запуск: streamlit run final_week4_demo.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.vehicle_parameters import VehicleParameters, create_default_parameters
    from src.nonlinear_bicycle_model import create_nonlinear_simulator, get_simulation_results_nonlinear
    from src.maneuvers import StandardManeuvers
    from src.tire_models import LinearTireModel, PiecewiseLinearTireModel, PacejkaTireModel
    
    IMPORTS_WORKING = True
except ImportError as e:
    st.error(f"❌ Ошибка импорта: {e}")
    IMPORTS_WORKING = False

def main():
    """Основная функция приложения"""
    
    st.set_page_config(
        page_title="🏎️ Симулятор управляемости автомобиля",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title("🏎️ Симулятор управляемости автомобиля")
    st.markdown("---")
    
    if not IMPORTS_WORKING:
        st.error("""
        ❌ Не все модули загружены корректно.
        
        **Возможные решения:**
        1. Проверьте структуру проекта
        2. Убедитесь, что все файлы находятся в правильных директориях
        3. Запустите `python check_imports.py` для диагностики
        """)
        return
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Параметры автомобиля")
        
        # Параметры автомобиля
        col1, col2 = st.columns(2)
        
        with col1:
            mass = st.slider("Масса (кг)", 800, 2000, 1200, 50)
            speed = st.slider("Скорость (м/с)", 5, 40, 20, 1)
            cog_ratio = st.slider("Положение ЦТ (a/L)", 0.3, 0.7, 0.47, 0.01)
            
        with col2:
            cf = st.slider("Жесткость перед. шин (кН/рад)", 40, 150, 80, 5) * 1000
            cr = st.slider("Жесткость зад. шин (кН/рад)", 40, 150, 100, 5) * 1000
        
        st.header("🎯 Настройки симуляции")
        
        # Выбор маневра
        maneuver_options = {
            "Шаг рулем 5°": "step_5",
            "Шаг рулем 10°": "step_10", 
            "Переставка": "lane_change",
            "Постоянный радиус": "constant_radius"
        }
        
        selected_maneuver = st.selectbox(
            "Выберите маневр:",
            list(maneuver_options.keys())
        )
        
        # Выбор модели шин
        tire_model_options = {
            "Линейная": "linear",
            "Кусочно-линейная": "piecewise", 
            "Магическая формула": "pacejka"
        }
        
        selected_tire_model = st.selectbox(
            "Модель шин:",
            list(tire_model_options.keys())
        )
        
        # Кнопки управления
        col1, col2 = st.columns(2)
        
        with col1:
            run_simulation = st.button("🚀 Запуск симуляции", use_container_width=True)
        
        with col2:
            if st.button("🔄 Сброс", use_container_width=True):
                st.rerun()
        
        # Сохранение параметров
        if run_simulation:
            st.session_state.params = {
                'mass': mass,
                'speed': speed, 
                'cog_ratio': cog_ratio,
                'cf': cf,
                'cr': cr,
                'maneuver': maneuver_options[selected_maneuver],
                'tire_model': tire_model_options[selected_tire_model]
            }
            st.session_state.run_simulation = True
    
    # Основная область
    if st.session_state.get('run_simulation', False):
        run_simulation_with_params(st.session_state.params)
    else:
        show_welcome_screen()

def show_welcome_screen():
    """Экран приветствия"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 О симуляторе
        
        Этот интерактивный симулятор позволяет исследовать поведение автомобиля 
        в различных маневрах с использованием нелинейной модели «велосипеда».
        
        **Возможности:**
        - 🚗 Настройка параметров автомобиля в реальном времени
        - 🎯 Выбор различных тестовых маневров  
        - 📈 Сравнение разных моделей шин
        - 📊 Визуализация результатов в интерактивных графиках
        
        **Для начала работы:**
        1. Настройте параметры автомобиля в боковой панели
        2. Выберите маневр и модель шин
        3. Нажмите «Запуск симуляции»
        """)
    
    with col2:
        # Схема автомобиля
        fig = create_car_schematic()
        st.plotly_chart(fig, use_container_width=True)

def create_car_schematic():
    """Создание схематического изображения автомобиля"""
    fig = go.Figure()
    
    # Кузов автомобиля
    fig.add_shape(type="rect", x0=-2, y0=-1, x1=2, y1=1, 
                 line=dict(color="blue", width=2), fillcolor="lightblue", opacity=0.5)
    
    # Колеса
    fig.add_shape(type="rect", x0=-2.2, y0=-1.2, x1=-1.8, y1=-0.8, 
                 line=dict(color="black", width=2), fillcolor="black")
    fig.add_shape(type="rect", x0=1.8, y0=-1.2, x1=2.2, y1=-0.8, 
                 line=dict(color="black", width=2), fillcolor="black")
    fig.add_shape(type="rect", x0=-2.2, y0=0.8, x1=-1.8, y1=1.2, 
                 line=dict(color="black", width=2), fillcolor="black")
    fig.add_shape(type="rect", x0=1.8, y0=0.8, x1=2.2, y1=1.2, 
                 line=dict(color="black", width=2), fillcolor="black")
    
    # Центр масс
    cog_x = 0  # По умолчанию в центре
    fig.add_trace(go.Scatter(x=[cog_x], y=[0], mode='markers', 
                           marker=dict(size=15, color='red', symbol='x')))
    
    fig.update_layout(
        title="Схема автомобиля (красный X - центр масс)",
        xaxis=dict(range=[-3, 3], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=300
    )
    
    return fig

def run_simulation_with_params(params):
    """Запуск симуляции с заданными параметрами"""
    st.header("📊 Результаты симуляции")
    
    # Создание параметров автомобиля
    vehicle_params = create_default_parameters()
    vehicle_params.m = params['mass']
    vehicle_params.V = params['speed']
    
    # Пересчет положения центра масс
    L = vehicle_params.a + vehicle_params.b
    vehicle_params.a = params['cog_ratio'] * L
    vehicle_params.b = L - vehicle_params.a
    
    vehicle_params.C_f = params['cf']
    vehicle_params.C_r = params['cr']
    
    # Прогресс-бар
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Подготовка симуляции...")
    progress_bar.progress(30)
    
    try:
        # Создание симулятора
        simulator = create_nonlinear_simulator(
            vehicle_params, 
            params['cf'], 
            params['cr'],
            params['tire_model']
        )
        
        status_text.text("🎯 Запуск маневра...")
        progress_bar.progress(60)
        
        # Получение маневра
        maneuver = get_maneuver(params['maneuver'])
        
        # Запуск симуляции
        solution = simulator.simulate(
            maneuver['delta_func'],
            params['speed'],
            maneuver['t_span']
        )
        
        status_text.text("📊 Обработка результатов...")
        progress_bar.progress(90)
        
        # Получение результатов
        results = get_simulation_results_nonlinear(solution)
        
        progress_bar.progress(100)
        
        # Отображение результатов
        display_results(results, maneuver, params)
        
        status_text.text("✅ Симуляция завершена!")
        
    except Exception as e:
        st.error(f"❌ Ошибка при выполнении симуляции: {e}")
        status_text.text("❌ Симуляция завершена с ошибкой")

def get_maneuver(maneuver_key):
    """Получение объекта маневра"""
    maneuver_map = {
        'step_5': StandardManeuvers.step_steer_maneuver(amplitude=np.radians(5)),
        'step_10': StandardManeuvers.step_steer_maneuver(amplitude=np.radians(10)),
        'lane_change': StandardManeuvers.lane_change_maneuver(amplitude=0.05),
        'constant_radius': StandardManeuvers.constant_radius_maneuver(radius=100)
    }
    return maneuver_map[maneuver_key]

def display_results(results, maneuver, params):
    """Отображение результатов симуляции"""
    st.subheader(f"Маневр: {maneuver['name']}")
    
    # Метрики производительности
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_beta = np.max(np.abs(np.degrees(results['beta'])))
        st.metric("Макс. угол скольжения", f"{max_beta:.2f}°")
        
    with col2:
        max_r = np.max(np.abs(np.degrees(results['angular_velocity'])))
        st.metric("Макс. угловая скорость", f"{max_r:.2f}°/с")
        
    with col3:
        final_y = results['Y'][-1]
        st.metric("Конечное смещение", f"{final_y:.2f} м")
        
    with col4:
        duration = results['time'][-1]
        st.metric("Длительность", f"{duration:.1f} с")
    
    # Интерактивные графики
    create_interactive_plots(results, maneuver, params)

def create_interactive_plots(results, maneuver, params):
    """Создание интерактивных графиков Plotly"""
    # Траектория
    fig_trajectory = go.Figure()
    fig_trajectory.add_trace(go.Scatter(
        x=results['X'], y=results['Y'],
        mode='lines',
        name='Траектория',
        line=dict(color='blue', width=3)
    ))
    fig_trajectory.update_layout(
        title='Траектория движения',
        xaxis_title='X координата, м',
        yaxis_title='Y координата, м',
        height=400
    )
    
    # Параметры движения
    fig_params = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Угол скольжения β', 'Угловая скорость r', 
                      'Курсовой угол ψ', 'Поперечное ускорение')
    )
    
    # Угол скольжения
    fig_params.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['beta']),
                  name='β', line=dict(color='green')),
        row=1, col=1
    )
    
    # Угловая скорость
    fig_params.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['angular_velocity']),
                  name='r', line=dict(color='magenta')),
        row=1, col=2
    )
    
    # Курсовой угол
    fig_params.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['yaw_angle']),
                  name='ψ', line=dict(color='cyan')),
        row=2, col=1
    )
    
    # Поперечное ускорение
    dt = np.diff(results['time'])
    dbeta_dt = np.diff(results['beta']) / dt
    dbeta_dt = np.append(dbeta_dt, dbeta_dt[-1])
    ay = params['speed'] * (dbeta_dt + results['angular_velocity'])
    
    fig_params.add_trace(
        go.Scatter(x=results['time'], y=ay,
                  name='ay', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig_params.update_layout(height=600, showlegend=False)
    
    # Отображение графиков
    st.plotly_chart(fig_trajectory, use_container_width=True)
    st.plotly_chart(fig_params, use_container_width=True)

if __name__ == "__main__":
    main()