# week5_complete.py
"""
ПОЛНАЯ ВЕРСИЯ НЕДЕЛИ 5 - все в одном файле
Запуск: streamlit run week5_complete.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
import pandas as pd
import json
import os
from datetime import datetime

# ==================== БАЗОВЫЕ КЛАССЫ ИЗ ПРЕДЫДУЩИХ НЕДЕЛЬ ====================

class VehicleParameters:
    """Параметры автомобиля"""
    def __init__(self):
        self.m = 1200.0
        self.J_z = 1500.0
        self.a = 1.4
        self.b = 1.6
        self.L = self.a + self.b
        self.C_f = 80000.0
        self.C_r = 100000.0
        self.V = 20.0

class LinearTireModel:
    """Линейная модель шины"""
    def __init__(self, stiffness):
        self.stiffness = stiffness
        
    def lateral_force(self, slip_angle):
        return self.stiffness * slip_angle

class PiecewiseLinearTireModel:
    """Кусочно-линейная модель шины"""
    def __init__(self, stiffness, max_force=5000):
        self.stiffness = stiffness
        self.max_force = max_force
        
    def lateral_force(self, slip_angle):
        linear_force = self.stiffness * slip_angle
        if abs(linear_force) <= self.max_force:
            return linear_force
        else:
            return np.sign(slip_angle) * self.max_force

class PacejkaTireModel:
    """Модель Пейджака"""
    def __init__(self, B=8.0, C=1.3, D=5000.0, E=0.5):
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        
    def lateral_force(self, slip_angle):
        x = self.B * slip_angle
        return self.D * np.sin(self.C * np.arctan(x - self.E * (x - np.arctan(x))))

def step_steer(t, amplitude=0.1, start_time=1.0):
    """Ступенчатое рулевое управление"""
    return amplitude if t >= start_time else 0.0

def sine_steer(t, amplitude=0.1, frequency=0.5, start_time=1.0):
    """Синусоидальное рулевое управление"""
    if t < start_time:
        return 0.0
    return amplitude * np.sin(2 * np.pi * frequency * (t - start_time))

def constant_radius_steer(t, radius=100.0, start_time=1.0):
    """Рулевое управление для движения по кругу"""
    if t < start_time:
        return 0.0
    L = 3.0  # колесная база
    return L / radius

class NonlinearBicycleModel:
    """Нелинейная модель велосипеда"""
    def __init__(self, vehicle_params, front_tire_model, rear_tire_model):
        self.params = vehicle_params
        self.front_tire = front_tire_model
        self.rear_tire = rear_tire_model
        
    def equations_of_motion(self, t, state, delta_func, V):
        beta, r, psi, X, Y = state
        delta = delta_func(t)
        
        # Углы увода
        alpha_f = delta - (beta + self.params.a * r / V)
        alpha_r = -(beta - self.params.b * r / V)
        
        # Боковые силы
        F_yf = self.front_tire.lateral_force(alpha_f)
        F_yr = self.rear_tire.lateral_force(alpha_r)
        
        # Уравнения движения
        dbeta_dt = (F_yf + F_yr) / (self.params.m * V) - r
        dr_dt = (F_yf * self.params.a - F_yr * self.params.b) / self.params.J_z
        dpsi_dt = r
        dX_dt = V * np.cos(psi - beta)
        dY_dt = V * np.sin(psi - beta)
        
        return [dbeta_dt, dr_dt, dpsi_dt, dX_dt, dY_dt]
    
    def simulate(self, delta_func, V, t_span, initial_state=None):
        if initial_state is None:
            initial_state = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        solution = solve_ivp(
            fun=lambda t, y: self.equations_of_motion(t, y, delta_func, V),
            t_span=t_span,
            y0=initial_state,
            method='RK45'
        )
        
        return solution

def create_nonlinear_simulator(vehicle_params, front_stiffness, rear_stiffness, tire_model_type):
    """Фабрика для создания симулятора"""
    if tire_model_type == 'linear':
        front_tire = LinearTireModel(front_stiffness)
        rear_tire = LinearTireModel(rear_stiffness)
    elif tire_model_type == 'piecewise':
        front_tire = PiecewiseLinearTireModel(front_stiffness)
        rear_tire = PiecewiseLinearTireModel(rear_stiffness)
    elif tire_model_type == 'pacejka':
        front_tire = PacejkaTireModel()
        rear_tire = PacejkaTireModel()
    else:
        raise ValueError(f"Неизвестный тип модели: {tire_model_type}")
    
    return NonlinearBicycleModel(vehicle_params, front_tire, rear_tire)

def get_simulation_results(solution):
    """Извлечение результатов симуляции"""
    t = solution.t
    beta = solution.y[0, :]
    r = solution.y[1, :]
    psi = solution.y[2, :]
    X = solution.y[3, :]
    Y = solution.y[4, :]
    
    return {
        'time': t,
        'beta': beta,
        'angular_velocity': r,
        'yaw_angle': psi,
        'X': X,
        'Y': Y,
        'success': solution.success
    }

# ==================== НОВЫЕ КЛАССЫ НЕДЕЛИ 5 ====================

class ThreeDAnimator:
    """3D визуализация и анимация"""
    def __init__(self):
        pass
    
    def create_3d_trajectory(self, results, title="3D Траектория движения"):
        """Создание 3D траектории"""
        fig = go.Figure()
        
        # Траектория
        fig.add_trace(go.Scatter3d(
            x=results['X'],
            y=results['Y'], 
            z=np.zeros_like(results['X']),
            mode='lines',
            name='Траектория',
            line=dict(color='blue', width=6),
            opacity=0.8
        ))
        
        # Начальная и конечная точки
        fig.add_trace(go.Scatter3d(
            x=[results['X'][0]],
            y=[results['Y'][0]],
            z=[0],
            mode='markers',
            name='Старт',
            marker=dict(size=10, color='green', symbol='diamond')
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[results['X'][-1]],
            y=[results['Y'][-1]], 
            z=[0],
            mode='markers',
            name='Финиш',
            marker=dict(size=10, color='red', symbol='diamond')
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X, м',
                yaxis_title='Y, м',
                zaxis_title='Z, м',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_energy_analysis(self, results, params):
        """Энергетический анализ"""
        time = results['time']
        beta = results['beta']
        r = results['angular_velocity']
        V = params.V
        
        # Расчет энергий
        rotational_energy = 0.5 * params.J_z * r**2
        lateral_energy = 0.5 * params.m * (V * beta)**2
        total_energy = rotational_energy + lateral_energy
        power = np.gradient(total_energy, time)
        
        # Создание графиков
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Кинетическая энергия вращения',
                'Энергия бокового движения', 
                'Полная энергия системы',
                'Мощность'
            )
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=rotational_energy, name='E_вращ', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=lateral_energy, name='E_бок', 
                      line=dict(color='green')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=total_energy, name='E_полная', 
                      line=dict(color='red')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=power, name='Мощность', 
                      line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="🔋 Энергетический анализ", showlegend=True)
        
        # Метрики
        metrics = {
            'max_rotational_energy': float(np.max(rotational_energy)),
            'max_lateral_energy': float(np.max(lateral_energy)),
            'max_total_energy': float(np.max(total_energy)),
            'max_power': float(np.max(power)),
            'energy_efficiency': float(np.trapz(total_energy, time) / (params.m * V**2 * time[-1]))
        }
        
        return fig, metrics

class DataExporter:
    """Экспорт данных"""
    def __init__(self, export_dir="exports"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_to_csv(self, results, params, maneuver_name):
        """Экспорт в CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_{timestamp}.csv"
        filepath = os.path.join(self.export_dir, filename)
        
        # Создание DataFrame
        data = {
            'time': results['time'],
            'beta_deg': np.degrees(results['beta']),
            'angular_velocity_deg_s': np.degrees(results['angular_velocity']),
            'yaw_angle_deg': np.degrees(results['yaw_angle']),
            'X': results['X'],
            'Y': results['Y']
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_to_excel(self, results, params, maneuver_name, energy_metrics=None):
        """Экспорт в Excel"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_report_{timestamp}.xlsx"
        filepath = os.path.join(self.export_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Основные данные
            main_data = pd.DataFrame({
                'Время (с)': results['time'],
                'Угол скольжения (°)': np.degrees(results['beta']),
                'Угловая скорость (°/с)': np.degrees(results['angular_velocity']),
                'Курсовой угол (°)': np.degrees(results['yaw_angle']),
                'Координата X (м)': results['X'],
                'Координата Y (м)': results['Y']
            })
            main_data.to_excel(writer, sheet_name='Основные данные', index=False)
            
            # Параметры
            params_data = pd.DataFrame([{
                'Маневр': maneuver_name,
                'Масса (кг)': params.m,
                'Скорость (м/с)': params.V,
                'Положение ЦТ (a/L)': params.a / params.L,
                'Жесткость передних шин (Н/рад)': params.C_f,
                'Жесткость задних шин (Н/рад)': params.C_r
            }])
            params_data.to_excel(writer, sheet_name='Параметры', index=False)
            
            # Энергетические метрики
            if energy_metrics:
                energy_data = pd.DataFrame([energy_metrics])
                energy_data.to_excel(writer, sheet_name='Энергетика', index=False)
        
        return filepath

class StabilityAnalyzer:
    """Анализ устойчивости"""
    def __init__(self):
        pass
    
    def analyze_stability(self, vehicle_params, speed_range=(5, 40), num_points=20):
        """Анализ устойчивости по скоростям"""
        speeds = np.linspace(speed_range[0], speed_range[1], num_points)
        
        stability_data = {
            'speeds': speeds,
            'eigenvalues_real': [],
            'eigenvalues_imag': [],
            'damping_ratios': []
        }
        
        for V in speeds:
            # Матрица состояния
            m, J_z, a, b, C_f, C_r = (vehicle_params.m, vehicle_params.J_z, 
                                     vehicle_params.a, vehicle_params.b, 
                                     vehicle_params.C_f, vehicle_params.C_r)
            
            A11 = -(C_f + C_r) / (m * V)
            A12 = -1 - (C_f * a - C_r * b) / (m * V**2)
            A21 = -(C_f * a - C_r * b) / J_z
            A22 = -(C_f * a**2 + C_r * b**2) / (J_z * V)
            
            A = np.array([[A11, A12], [A21, A22]])
            eigenvalues = np.linalg.eigvals(A)
            
            stability_data['eigenvalues_real'].append(eigenvalues.real)
            stability_data['eigenvalues_imag'].append(eigenvalues.imag)
            
            # Коэффициенты демпфирования
            damping = []
            for eig in eigenvalues:
                wn = np.abs(eig)
                zeta = -eig.real / wn if wn > 0 else 0
                damping.append(zeta)
            stability_data['damping_ratios'].append(damping)
        
        return stability_data
    
    def create_stability_plot(self, stability_data):
        """Создание графика устойчивости"""
        speeds = stability_data['speeds']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Действительная часть собственных значений',
                'Мнимая часть собственных значений',
                'Коэффициенты демпфирования',
                'Карта устойчивости'
            )
        )
        
        # Действительные части
        for i in range(2):
            real_parts = [eig[i] for eig in stability_data['eigenvalues_real']]
            fig.add_trace(
                go.Scatter(x=speeds, y=real_parts, name=f'λ{i+1} Re',
                          line=dict(width=3)),
                row=1, col=1
            )
        
        # Мнимые части
        for i in range(2):
            imag_parts = [eig[i] for eig in stability_data['eigenvalues_imag']]
            fig.add_trace(
                go.Scatter(x=speeds, y=imag_parts, name=f'λ{i+1} Im',
                          line=dict(width=3)),
                row=1, col=2
            )
        
        # Коэффициенты демпфирования
        for i in range(2):
            damping = [damp[i] for damp in stability_data['damping_ratios']]
            fig.add_trace(
                go.Scatter(x=speeds, y=damping, name=f'ζ{i+1}',
                          line=dict(width=3)),
                row=2, col=1
            )
        
        # Карта устойчивости
        for i in range(2):
            real_parts = [eig[i] for eig in stability_data['eigenvalues_real']]
            imag_parts = [eig[i] for eig in stability_data['eigenvalues_imag']]
            
            colors = ['red' if real > 0 else 'green' for real in real_parts]
            
            fig.add_trace(
                go.Scatter(x=real_parts, y=imag_parts, 
                          mode='markers+lines',
                          name=f'λ{i+1}',
                          marker=dict(color=colors, size=8),
                          line=dict(color='gray', width=1)),
                row=2, col=2
            )
        
        fig.update_layout(height=700, title_text="📊 Анализ устойчивости", showlegend=True)
        return fig

# ==================== ОСНОВНОЕ ПРИЛОЖЕНИЕ STREAMLIT ====================

def main():
    """Главная функция приложения"""
    
    st.set_page_config(
        page_title="🏎️ Симулятор управляемости - Неделя 5",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title("🏎️ Симулятор управляемости автомобиля")
    st.markdown("### 📈 Неделя 5: Расширенная визуализация и анализ")
    st.markdown("---")
    
    # Инициализация компонентов
    if 'three_d_animator' not in st.session_state:
        st.session_state.three_d_animator = ThreeDAnimator()
    if 'data_exporter' not in st.session_state:
        st.session_state.data_exporter = DataExporter()
    if 'stability_analyzer' not in st.session_state:
        st.session_state.stability_analyzer = StabilityAnalyzer()
    
    # Навигация по вкладкам
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚗 Основная симуляция", 
        "📊 3D Визуализация", 
        "🔬 Анализ устойчивости",
        "💾 Экспорт данных"
    ])
    
    with tab1:
        show_main_simulation()
    
    with tab2:
        show_3d_visualization()
    
    with tab3:
        show_stability_analysis()
    
    with tab4:
        show_data_export()

def show_main_simulation():
    """Вкладка основной симуляции"""
    st.header("🚗 Основная симуляция")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Параметры автомобиля")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mass = st.slider("Масса (кг)", 800, 2000, 1200, 50)
            speed = st.slider("Скорость (м/с)", 5, 40, 20, 1)
            cog_ratio = st.slider("Положение ЦТ (a/L)", 0.3, 0.7, 0.47, 0.01)
            
        with col2:
            cf = st.slider("Жесткость перед. шин (кН/рад)", 40, 150, 80, 5) * 1000
            cr = st.slider("Жесткость зад. шин (кН/рад)", 40, 150, 100, 5) * 1000
        
        st.header("🎯 Настройки симуляции")
        
        maneuver_type = st.selectbox(
            "Маневр:",
            ["Шаг рулем 5°", "Шаг рулем 10°", "Переставка", "Постоянный радиус"]
        )
        
        tire_model = st.selectbox(
            "Модель шин:",
            ["Линейная", "Кусочно-линейная", "Магическая формула"]
        )
        
        # Расширенные настройки
        with st.expander("🔧 Расширенные настройки"):
            sim_time = st.slider("Время симуляции (с)", 5, 30, 10)
            enable_energy_analysis = st.checkbox("Включить энергетический анализ", value=True)
        
        if st.button("🚀 Запустить симуляцию", use_container_width=True):
            st.session_state.run_simulation = True
            st.session_state.params = {
                'mass': mass, 'speed': speed, 'cog_ratio': cog_ratio,
                'cf': cf, 'cr': cr, 'maneuver': maneuver_type, 
                'tire_model': tire_model, 'sim_time': sim_time,
                'energy_analysis': enable_energy_analysis
            }
    
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
        ### 🎯 Расширенные возможности недели 5
        
        **Новые функции:**
        - 📊 **3D визуализация** траектории движения
        - 🔋 **Энергетический анализ** системы
        - 📈 **Анализ устойчивости** по диапазону скоростей
        - 💾 **Полный экспорт** данных в CSV и Excel
        - 🎨 **Интерактивные дашборды** анализа
        
        **Для начала работы:**
        1. Настройте параметры автомобиля в боковой панели
        2. Выберите маневр и модель шин
        3. Запустите симуляцию
        4. Исследуйте результаты в разных вкладках
        """)
    
    with col2:
        # Схема автомобиля
        fig = go.Figure()
        fig.add_shape(type="rect", x0=-2, y0=-1, x1=2, y1=1,
                     line=dict(color="blue", width=3), fillcolor="lightblue", opacity=0.7)
        
        # Колеса
        wheel_positions = [(-2.2, -1.2, -1.8, -0.8), (1.8, -1.2, 2.2, -0.8),
                          (-2.2, 0.8, -1.8, 1.2), (1.8, 0.8, 2.2, 1.2)]
        
        for x0, y0, x1, y1 in wheel_positions:
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                         line=dict(color="black", width=2), fillcolor="black")
        
        fig.update_layout(
            title="Схема автомобиля",
            xaxis=dict(range=[-3, 3], showticklabels=False),
            yaxis=dict(range=[-2, 2], showticklabels=False),
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def run_simulation_with_params(params):
    """Запуск симуляции с заданными параметрами"""
    st.header("📊 Результаты симуляции")
    
    # Создание параметров автомобиля
    vehicle_params = VehicleParameters()
    vehicle_params.m = params['mass']
    vehicle_params.V = params['speed']
    
    # Положение центра масс
    L = vehicle_params.a + vehicle_params.b
    vehicle_params.a = params['cog_ratio'] * L
    vehicle_params.b = L - vehicle_params.a
    
    vehicle_params.C_f = params['cf']
    vehicle_params.C_r = params['cr']
    
    # Выбор модели шин
    tire_model_map = {
        "Линейная": "linear",
        "Кусочно-линейная": "piecewise", 
        "Магическая формула": "pacejka"
    }
    
    # Прогресс-бар
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Подготовка симуляции...")
    progress_bar.progress(20)
    
    try:
        # Создание симулятора
        simulator = create_nonlinear_simulator(
            vehicle_params, params['cf'], params['cr'], 
            tire_model_map[params['tire_model']]
        )
        
        status_text.text("🎯 Запуск маневра...")
        progress_bar.progress(50)
        
        # Определение маневра
        maneuver = get_maneuver(params['maneuver'], params['sim_time'])
        
        # Запуск симуляции
        solution = simulator.simulate(
            maneuver['delta_func'], params['speed'], maneuver['t_span']
        )
        
        status_text.text("📊 Обработка результатов...")
        progress_bar.progress(80)
        
        results = get_simulation_results(solution)
        
        # Энергетический анализ
        if params['energy_analysis']:
            energy_fig, energy_metrics = st.session_state.three_d_animator.create_energy_analysis(
                results, vehicle_params
            )
        else:
            energy_fig, energy_metrics = None, None
        
        progress_bar.progress(100)
        
        # Сохранение результатов
        st.session_state.last_results = results
        st.session_state.last_vehicle_params = vehicle_params
        st.session_state.last_maneuver_name = params['maneuver']
        st.session_state.energy_metrics = energy_metrics
        
        # Отображение результатов
        display_results(results, vehicle_params, energy_fig, energy_metrics)
        status_text.text("✅ Симуляция завершена!")
        
    except Exception as e:
        st.error(f"❌ Ошибка при выполнении симуляции: {e}")
        status_text.text("❌ Симуляция завершена с ошибкой")

def get_maneuver(maneuver_name, sim_time):
    """Получение функции маневра"""
    if maneuver_name == "Шаг рулем 5°":
        return {
            'delta_func': lambda t: step_steer(t, np.radians(5), 1.0),
            't_span': (0, sim_time),
            'name': 'Шаг рулем 5°'
        }
    elif maneuver_name == "Шаг рулем 10°":
        return {
            'delta_func': lambda t: step_steer(t, np.radians(10), 1.0),
            't_span': (0, sim_time),
            'name': 'Шаг рулем 10°'
        }
    elif maneuver_name == "Переставка":
        return {
            'delta_func': lambda t: sine_steer(t, 0.05, 0.5, 1.0),
            't_span': (0, sim_time),
            'name': 'Переставка'
        }
    else:  # Постоянный радиус
        return {
            'delta_func': lambda t: constant_radius_steer(t, 100, 1.0),
            't_span': (0, sim_time),
            'name': 'Постоянный радиус R=100м'
        }

def display_results(results, vehicle_params, energy_fig, energy_metrics):
    """Отображение результатов симуляции"""
    # Основные метрики
    st.subheader("📈 Основные метрики")
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
    
    # Энергетические метрики
    if energy_metrics:
        st.subheader("🔋 Энергетические характеристики")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Макс. энергия вращения", f"{energy_metrics['max_rotational_energy']:.1f} Дж")
        with col2:
            st.metric("Макс. энергия бок. движения", f"{energy_metrics['max_lateral_energy']:.1f} Дж")
        with col3:
            st.metric("Макс. мощность", f"{energy_metrics['max_power']:.1f} Вт")
        with col4:
            st.metric("Энергоэффективность", f"{energy_metrics['energy_efficiency']:.3f}")
    
    # Графики
    st.subheader("📊 Визуализация результатов")
    create_main_plots(results, energy_fig)

def create_main_plots(results, energy_fig):
    """Создание основных графиков"""
    # Базовые графики
    fig_basic = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Траектория движения', 'Угол скольжения β',
                      'Угловая скорость r', 'Фазовый портрет')
    )
    
    # Траектория
    fig_basic.add_trace(
        go.Scatter(x=results['X'], y=results['Y'], name='Траектория',
                  line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Угол скольжения
    fig_basic.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['beta']), name='β',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # Угловая скорость
    fig_basic.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['angular_velocity']), name='r',
                  line=dict(color='magenta', width=2)),
        row=2, col=1
    )
    
    # Фазовый портрет
    fig_basic.add_trace(
        go.Scatter(x=np.degrees(results['beta']), y=np.degrees(results['angular_velocity']),
                  name='Фазовый портрет', line=dict(color='orange', width=2)),
        row=2, col=2
    )
    
    fig_basic.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig_basic, use_container_width=True)
    
    # Энергетические графики
    if energy_fig:
        st.plotly_chart(energy_fig, use_container_width=True)

def show_3d_visualization():
    """Вкладка 3D визуализации"""
    st.header("📊 3D Визуализация")
    
    if 'last_results' not in st.session_state:
        st.warning("⚠️ Сначала выполните симуляцию во вкладке 'Основная симуляция'")
        return
    
    results = st.session_state.last_results
    
    st.subheader("3D Траектория движения")
    fig_3d = st.session_state.three_d_animator.create_3d_trajectory(results)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Дополнительная информация
    with st.expander("📐 Дополнительные проекции"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Проекция X-Z
            fig_xz = go.Figure()
            fig_xz.add_trace(go.Scatter(
                x=results['X'], y=np.zeros_like(results['X']),
                mode='lines', name='X-Z проекция',
                line=dict(color='red', width=3)
            ))
            fig_xz.update_layout(title="Проекция X-Z", height=300)
            st.plotly_chart(fig_xz, use_container_width=True)
        
        with col2:
            # Проекция Y-Z  
            fig_yz = go.Figure()
            fig_yz.add_trace(go.Scatter(
                x=results['Y'], y=np.zeros_like(results['Y']),
                mode='lines', name='Y-Z проекция',
                line=dict(color='green', width=3)
            ))
            fig_yz.update_layout(title="Проекция Y-Z", height=300)
            st.plotly_chart(fig_yz, use_container_width=True)

def show_stability_analysis():
    """Вкладка анализа устойчивости"""
    st.header("🔬 Анализ устойчивости")
    
    if 'last_vehicle_params' not in st.session_state:
        st.warning("⚠️ Сначала выполните симуляцию во вкладке 'Основная симуляция'")
        return
    
    vehicle_params = st.session_state.last_vehicle_params
    
    st.subheader("Настройки анализа")
    col1, col2 = st.columns(2)
    
    with col1:
        min_speed = st.slider("Минимальная скорость (м/с)", 1, 20, 5, key="min_speed")
        max_speed = st.slider("Максимальная скорость (м/с)", 10, 60, 40, key="max_speed")
    
    with col2:
        num_points = st.slider("Количество точек анализа", 10, 50, 20, key="num_points")
    
    if st.button("🔄 Выполнить анализ устойчивости", use_container_width=True):
        with st.spinner("🔬 Выполняется анализ устойчивости..."):
            # Анализ устойчивости
            stability_data = st.session_state.stability_analyzer.analyze_stability(
                vehicle_params, (min_speed, max_speed), num_points
            )
            
            # Создание графиков
            stability_fig = st.session_state.stability_analyzer.create_stability_plot(stability_data)
            
            st.plotly_chart(stability_fig, use_container_width=True)
            
            # Дополнительная информация
            st.subheader("📈 Ключевые показатели")
            
            # Расчет для текущей скорости
            current_speed = vehicle_params.V
            V_crit = calculate_critical_speed(vehicle_params)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Текущая скорость", f"{current_speed} м/с")
            with col2:
                st.metric("Критическая скорость", f"{V_crit:.1f} м/с" if V_crit != float('inf') else "∞")
            with col3:
                stability_status = "Стабильная" if current_speed < V_crit else "Нестабильная"
                st.metric("Статус устойчивости", stability_status)

def calculate_critical_speed(vehicle_params):
    """Расчет критической скорости"""
    m, L, a, b, C_f, C_r = (vehicle_params.m, vehicle_params.L,
                           vehicle_params.a, vehicle_params.b,
                           vehicle_params.C_f, vehicle_params.C_r)
    
    K = (m / L) * (b / C_f - a / C_r)
    
    if K < 0:
        return np.sqrt(-L / K)
    else:
        return float('inf')

def show_data_export():
    """Вкладка экспорта данных"""
    st.header("💾 Экспорт данных")
    
    if 'last_results' not in st.session_state:
        st.warning("⚠️ Сначала выполните симуляцию во вкладке 'Основная симуляция'")
        return
    
    results = st.session_state.last_results
    vehicle_params = st.session_state.last_vehicle_params
    maneuver_name = st.session_state.last_maneuver_name
    energy_metrics = st.session_state.get('energy_metrics', None)
    
    st.subheader("📤 Варианты экспорта")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Экспорт в CSV", use_container_width=True):
            with st.spinner("Экспорт в CSV..."):
                csv_file = st.session_state.data_exporter.export_to_csv(
                    results, vehicle_params, maneuver_name
                )
                st.success(f"✅ Данные экспортированы в: `{csv_file}`")
    
    with col2:
        if st.button("📊 Экспорт в Excel", use_container_width=True):
            with st.spinner("Экспорт в Excel..."):
                excel_file = st.session_state.data_exporter.export_to_excel(
                    results, vehicle_params, maneuver_name, energy_metrics
                )
                st.success(f"✅ Отчет экспортирован в: `{excel_file}`")
    
    # Предпросмотр данных
    st.subheader("👀 Предпросмотр данных")
    
    if st.checkbox("Показать таблицу данных"):
        data = {
            'Время (с)': results['time'],
            'Угол скольжения (°)': np.degrees(results['beta']),
            'Угловая скорость (°/с)': np.degrees(results['angular_velocity']),
            'Курсовой угол (°)': np.degrees(results['yaw_angle']),
            'X (м)': results['X'],
            'Y (м)': results['Y']
        }
        
        df = pd.DataFrame(data)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Статистика
        st.subheader("📊 Статистика данных")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Количество точек", len(results['time']))
        with col2:
            dt = results['time'][1] - results['time'][0] if len(results['time']) > 1 else 0
            st.metric("Временной шаг", f"{dt:.3f} с")
        with col3:
            st.metric("Общее время", f"{results['time'][-1]:.1f} с")

if __name__ == "__main__":
    # Инициализация session_state
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    
    main()