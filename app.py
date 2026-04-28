# src/web_interface/app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Абсолютные импорты из src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vehicle_parameters import VehicleParameters, create_default_parameters
from src.nonlinear_bicycle_model import create_nonlinear_simulator, get_simulation_results_nonlinear
from src.maneuvers import StandardManeuvers
from src.tire_models import LinearTireModel, PiecewiseLinearTireModel, PacejkaTireModel

class StreamlitSimulator:
    """Основной класс веб-приложения Streamlit"""
    
    def __init__(self):
        self.setup_page()
        self.params = None
        self.simulation_results = None
        
    def setup_page(self):
        """Настройка страницы Streamlit"""
        st.set_page_config(
            page_title="🏎️ Симулятор управляемости автомобиля",
            page_icon="🏎️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Кастомный CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #2e86ab;
            border-bottom: 2px solid #2e86ab;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">🏎️ Интерактивный симулятор управляемости</h1>', 
                   unsafe_allow_html=True)
        
    def run(self):
        """Запуск основного приложения"""
        # Боковая панель с настройками
        with st.sidebar:
            self.render_sidebar()
        
        # Основная область
        self.render_main_content()
    
    def render_sidebar(self):
        """Отрисовка боковой панели с настройками"""
        st.markdown("### ⚙️ Параметры автомобиля")
        
        # Параметры автомобиля
        col1, col2 = st.columns(2)
        
        with col1:
            mass = st.slider("Масса (кг)", 800, 2000, 1200, 50)
            speed = st.slider("Скорость (м/с)", 5, 40, 20, 1)
            cog_ratio = st.slider("Положение ЦТ (a/L)", 0.3, 0.7, 0.47, 0.01)
            
        with col2:
            cf = st.slider("Жесткость перед. шин (кН/рад)", 40, 150, 80, 5) * 1000
            cr = st.slider("Жесткость зад. шин (кН/рад)", 40, 150, 100, 5) * 1000
        
        st.markdown("### 🎯 Настройки симуляции")
        
        # Выбор маневра
        maneuver_options = {
            "Шаг рулем 5°": "step_5",
            "Шаг рулем 10°": "step_10", 
            "Переставка": "lane_change",
            "Двойная переставка": "double_lane_change",
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            run_simulation = st.button("🚀 Запуск симуляции", use_container_width=True)
        
        with col2:
            compare_models = st.button("📊 Сравнить модели", use_container_width=True)
            
        with col3:
            if st.button("🔄 Сброс", use_container_width=True):
                st.rerun()
        
        # Сохранение параметров в session_state
        st.session_state.params = {
            'mass': mass,
            'speed': speed, 
            'cog_ratio': cog_ratio,
            'cf': cf,
            'cr': cr,
            'maneuver': maneuver_options[selected_maneuver],
            'tire_model': tire_model_options[selected_tire_model]
        }
        
        st.session_state.run_simulation = run_simulation
        st.session_state.compare_models = compare_models
    
    def render_main_content(self):
        """Отрисовка основного контента"""
        if st.session_state.get('run_simulation', False):
            self.run_single_simulation()
        elif st.session_state.get('compare_models', False):
            self.run_model_comparison()
        else:
            self.render_welcome_screen()
    
    def render_welcome_screen(self):
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
            - 🎮 Анализ устойчивости и управляемости
            
            **Для начала работы:**
            1. Настройте параметры автомобиля в боковой панели
            2. Выберите маневр и модель шин
            3. Нажмите «Запуск симуляции»
            """)
        
        with col2:
            # Схема автомобиля
            fig = self.create_car_schematic()
            st.plotly_chart(fig, use_container_width=True)
    
    def create_car_schematic(self):
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
        params = st.session_state.get('params', {})
        cog_x = (params.get('cog_ratio', 0.47) - 0.5) * 4  # Смещение от центра
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
    
    def run_single_simulation(self):
        """Запуск одиночной симуляции"""
        params = st.session_state.params
        
        # Создание параметров автомобиля
        vehicle_params = self.create_vehicle_parameters(params)
        
        # Прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Подготовка симуляции...")
        progress_bar.progress(20)
        
        # Создание симулятора
        simulator = create_nonlinear_simulator(
            vehicle_params, 
            params['cf'], 
            params['cr'],
            params['tire_model']
        )
        
        status_text.text("🎯 Запуск маневра...")
        progress_bar.progress(50)
        
        # Получение маневра
        maneuver = self.get_maneuver(params['maneuver'])
        
        # Запуск симуляции
        solution = simulator.simulate(
            maneuver['delta_func'],
            params['speed'],
            maneuver['t_span']
        )
        
        status_text.text("📊 Обработка результатов...")
        progress_bar.progress(80)
        
        # Получение результатов
        results = get_simulation_results_nonlinear(solution)
        
        status_text.text("✅ Визуализация...")
        progress_bar.progress(100)
        
        # Отображение результатов
        self.display_results(results, maneuver, params)
        
        status_text.text("✅ Симуляция завершена!")
    
    def run_model_comparison(self):
        """Сравнение разных моделей шин"""
        params = st.session_state.params
        
        st.markdown("## 📊 Сравнение моделей шин")
        
        vehicle_params = self.create_vehicle_parameters(params)
        maneuver = self.get_maneuver(params['maneuver'])
        
        # Модели для сравнения
        tire_models = [
            ('linear', 'Линейная', 'blue'),
            ('piecewise', 'Кусочно-линейная', 'red'), 
            ('pacejka', 'Магическая формула', 'green')
        ]
        
        # Запуск симуляций для каждой модели
        all_results = {}
        
        for model_type, model_name, color in tire_models:
            with st.spinner(f"Запуск {model_name} модели..."):
                simulator = create_nonlinear_simulator(
                    vehicle_params, params['cf'], params['cr'], model_type
                )
                solution = simulator.simulate(
                    maneuver['delta_func'], params['speed'], maneuver['t_span']
                )
                all_results[model_name] = {
                    'results': get_simulation_results_nonlinear(solution),
                    'color': color
                }
        
        # Визуализация сравнения
        self.display_comparison(all_results, maneuver)
    
    def create_vehicle_parameters(self, params):
        """Создание объекта параметров автомобиля"""
        vehicle_params = create_default_parameters()
        vehicle_params.m = params['mass']
        vehicle_params.V = params['speed']
        
        # Пересчет положения центра масс
        L = vehicle_params.a + vehicle_params.b
        vehicle_params.a = params['cog_ratio'] * L
        vehicle_params.b = L - vehicle_params.a
        
        vehicle_params.C_f = params['cf']
        vehicle_params.C_r = params['cr']
        
        return vehicle_params
    
    def get_maneuver(self, maneuver_key):
        """Получение объекта маневра"""
        maneuver_map = {
            'step_5': StandardManeuvers.step_steer_maneuver(amplitude=np.radians(5)),
            'step_10': StandardManeuvers.step_steer_maneuver(amplitude=np.radians(10)),
            'lane_change': StandardManeuvers.lane_change_maneuver(amplitude=0.05),
            'double_lane_change': StandardManeuvers.double_lane_change_maneuver(),
            'constant_radius': StandardManeuvers.constant_radius_maneuver(radius=100)
        }
        return maneuver_map[maneuver_key]
    
    def display_results(self, results, maneuver, params):
        """Отображение результатов симуляции"""
        st.markdown(f"## 📈 Результаты: {maneuver['name']}")
        
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
        
        # Интерактивные графики Plotly
        self.create_interactive_plots(results, maneuver, params)
    
    def create_interactive_plots(self, results, maneuver, params):
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
            showlegend=True
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
    
    def display_comparison(self, all_results, maneuver):
        """Отображение сравнения моделей"""
        st.markdown("### 📊 Сравнительный анализ")
        
        # Создание сравнительных графиков
        fig_comparison = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Траектории движения', 'Углы скольжения β',
                          'Угловые скорости r', 'Фазовые портреты')
        )
        
        for model_name, data in all_results.items():
            results = data['results']
            color = data['color']
            
            # Траектории
            fig_comparison.add_trace(
                go.Scatter(x=results['X'], y=results['Y'],
                          name=model_name, line=dict(color=color)),
                row=1, col=1
            )
            
            # Углы скольжения
            fig_comparison.add_trace(
                go.Scatter(x=results['time'], y=np.degrees(results['beta']),
                          name=model_name, line=dict(color=color), showlegend=False),
                row=1, col=2
            )
            
            # Угловые скорости
            fig_comparison.add_trace(
                go.Scatter(x=results['time'], y=np.degrees(results['angular_velocity']),
                          name=model_name, line=dict(color=color), showlegend=False),
                row=2, col=1
            )
            
            # Фазовые портреты
            fig_comparison.add_trace(
                go.Scatter(x=np.degrees(results['beta']), 
                          y=np.degrees(results['angular_velocity']),
                          name=model_name, line=dict(color=color), showlegend=False),
                row=2, col=2
            )
        
        fig_comparison.update_layout(height=800, title_text="Сравнение моделей шин")
        fig_comparison.update_xaxes(title_text="X, м", row=1, col=1)
        fig_comparison.update_yaxes(title_text="Y, м", row=1, col=1)
        fig_comparison.update_xaxes(title_text="Время, с", row=1, col=2)
        fig_comparison.update_yaxes(title_text="β, °", row=1, col=2)
        fig_comparison.update_xaxes(title_text="Время, с", row=2, col=1)
        fig_comparison.update_yaxes(title_text="r, °/с", row=2, col=1)
        fig_comparison.update_xaxes(title_text="β, °", row=2, col=2)
        fig_comparison.update_yaxes(title_text="r, °/с", row=2, col=2)
        
        st.plotly_chart(fig_comparison, use_container_width=True)

def main():
    """Основная функция запуска приложения"""
    app = StreamlitSimulator()
    app.run()

if __name__ == "__main__":
    main()