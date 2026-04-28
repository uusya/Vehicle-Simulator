# examples/simple_working_demo.py
"""
Упрощенная рабочая версия симулятора для недели 4
Запуск: streamlit run simple_working_demo.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

# Простые классы прямо в файле для избежания проблем с импортами

class SimpleVehicleParameters:
    """Упрощенные параметры автомобиля"""
    def __init__(self):
        self.m = 1200.0           # масса, кг
        self.J_z = 1500.0         # момент инерции, кг·м²
        self.a = 1.4              # расстояние до передней оси, м
        self.b = 1.6              # расстояние до задней оси, м
        self.L = self.a + self.b  # колесная база, м
        self.C_f = 80000.0        # жесткость передних шин, Н/рад
        self.C_r = 100000.0       # жесткость задних шин, Н/рад
        self.V = 20.0             # скорость, м/с

class SimpleTireModel:
    """Упрощенная модель шины"""
    def __init__(self, stiffness, model_type='linear'):
        self.stiffness = stiffness
        self.model_type = model_type
        
    def lateral_force(self, slip_angle):
        if self.model_type == 'linear':
            return self.stiffness * slip_angle
        elif self.model_type == 'piecewise':
            # Кусочно-линейная модель с насыщением
            max_force = 5000
            linear_force = self.stiffness * slip_angle
            if abs(linear_force) <= max_force:
                return linear_force
            else:
                return np.sign(slip_angle) * max_force
        else:
            return self.stiffness * slip_angle

def step_steer(t, amplitude=0.1, start_time=1.0):
    """Ступенчатое рулевое воздействие"""
    return amplitude if t >= start_time else 0.0

def sine_steer(t, amplitude=0.1, frequency=0.5, start_time=1.0):
    """Синусоидальное рулевое воздействие"""
    if t < start_time:
        return 0.0
    return amplitude * np.sin(2 * np.pi * frequency * (t - start_time))

def simple_bicycle_model(t, state, params, delta_func, front_tire, rear_tire):
    """Упрощенная модель велосипеда"""
    beta, r, psi, X, Y = state
    delta = delta_func(t)
    
    # Углы увода
    alpha_f = delta - (beta + params.a * r / params.V)
    alpha_r = -(beta - params.b * r / params.V)
    
    # Боковые силы
    F_yf = front_tire.lateral_force(alpha_f)
    F_yr = rear_tire.lateral_force(alpha_r)
    
    # Уравнения движения
    dbeta_dt = (F_yf + F_yr) / (params.m * params.V) - r
    dr_dt = (F_yf * params.a - F_yr * params.b) / params.J_z
    dpsi_dt = r
    dX_dt = params.V * np.cos(psi - beta)
    dY_dt = params.V * np.sin(psi - beta)
    
    return [dbeta_dt, dr_dt, dpsi_dt, dX_dt, dY_dt]

def run_simulation(params, delta_func, front_tire, rear_tire, t_span=(0, 10)):
    """Запуск симуляции"""
    initial_state = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    solution = solve_ivp(
        fun=lambda t, y: simple_bicycle_model(t, y, params, delta_func, front_tire, rear_tire),
        t_span=t_span,
        y0=initial_state,
        method='RK45',
        dense_output=True
    )
    
    return solution

def main():
    """Основное приложение Streamlit"""
    
    st.set_page_config(
        page_title="🏎️ Упрощенный симулятор управляемости",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title("🏎️ Упрощенный симулятор управляемости автомобиля")
    st.markdown("---")
    
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
            ["Линейная", "Кусочно-линейная"]
        )
        
        if st.button("🚀 Запустить симуляцию", use_container_width=True):
            st.session_state.run_simulation = True
            st.session_state.params = {
                'mass': mass, 'speed': speed, 'cog_ratio': cog_ratio,
                'cf': cf, 'cr': cr, 'maneuver': maneuver_type, 'tire_model': tire_model
            }
    
    # Основная область
    if st.session_state.get('run_simulation', False):
        show_simulation_results(st.session_state.params)
    else:
        show_welcome_screen()

def show_welcome_screen():
    """Экран приветствия"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 О симуляторе
        
        Этот упрощенный симулятор демонстрирует основы динамики автомобиля 
        с использованием модели «велосипеда».
        
        **Возможности:**
        - 🚗 Настройка параметров автомобиля
        - 🎯 Различные тестовые маневры  
        - 📈 Две модели шин
        - 📊 Интерактивная визуализация
        
        **Для начала работы:**
        1. Настройте параметры в боковой панели
        2. Выберите маневр и модель шин
        3. Нажмите «Запустить симуляцию»
        """)
    
    with col2:
        fig = create_car_schematic()
        st.plotly_chart(fig, use_container_width=True)

def create_car_schematic():
    """Схема автомобиля"""
    fig = go.Figure()
    
    # Кузов
    fig.add_shape(type="rect", x0=-2, y0=-1, x1=2, y1=1, 
                 line=dict(color="blue", width=2), fillcolor="lightblue", opacity=0.5)
    
    # Колеса
    wheel_positions = [(-2.2, -1.2, -1.8, -0.8), (1.8, -1.2, 2.2, -0.8),
                      (-2.2, 0.8, -1.8, 1.2), (1.8, 0.8, 2.2, 1.2)]
    
    for x0, y0, x1, y1 in wheel_positions:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                     line=dict(color="black", width=2), fillcolor="black")
    
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                           marker=dict(size=15, color='red', symbol='x')))
    
    fig.update_layout(
        title="Схема автомобиля",
        xaxis=dict(range=[-3, 3], showticklabels=False),
        yaxis=dict(range=[-2, 2], showticklabels=False),
        showlegend=False,
        height=300
    )
    
    return fig

def show_simulation_results(params):
    """Показать результаты симуляции"""
    st.header("📊 Результаты симуляции")
    
    # Создание параметров
    vehicle_params = SimpleVehicleParameters()
    vehicle_params.m = params['mass']
    vehicle_params.V = params['speed']
    
    # Положение центра масс
    L = vehicle_params.a + vehicle_params.b
    vehicle_params.a = params['cog_ratio'] * L
    vehicle_params.b = L - vehicle_params.a
    
    vehicle_params.C_f = params['cf']
    vehicle_params.C_r = params['cr']
    
    # Модели шин
    tire_type = 'piecewise' if params['tire_model'] == "Кусочно-линейная" else 'linear'
    front_tire = SimpleTireModel(params['cf'], tire_type)
    rear_tire = SimpleTireModel(params['cr'], tire_type)
    
    # Функция рулевого управления
    if params['maneuver'] == "Шаг рулем 5°":
        delta_func = lambda t: step_steer(t, np.radians(5), 1.0)
        t_span = (0, 10)
    elif params['maneuver'] == "Шаг рулем 10°":
        delta_func = lambda t: step_steer(t, np.radians(10), 1.0)
        t_span = (0, 10)
    elif params['maneuver'] == "Переставка":
        delta_func = lambda t: sine_steer(t, 0.05, 0.5, 1.0)
        t_span = (0, 10)
    else:  # Постоянный радиус
        delta_func = lambda t: 0.1 if t >= 1.0 else 0.0
        t_span = (0, 15)
    
    # Прогресс
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Запуск симуляции...")
    
    try:
        # Запуск симуляции
        solution = run_simulation(vehicle_params, delta_func, front_tire, rear_tire, t_span)
        progress_bar.progress(100)
        
        # Извлечение результатов
        t = solution.t
        beta = solution.y[0, :]
        r = solution.y[1, :]
        psi = solution.y[2, :]
        X = solution.y[3, :]
        Y = solution.y[4, :]
        
        results = {
            'time': t, 'beta': beta, 'angular_velocity': r,
            'yaw_angle': psi, 'X': X, 'Y': Y
        }
        
        # Отображение результатов
        display_results(results, params)
        status_text.text("✅ Симуляция завершена!")
        
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        status_text.text("❌ Ошибка симуляции")

def display_results(results, params):
    """Отображение результатов"""
    # Метрики
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
    
    # Графики
    create_plots(results, params)

def create_plots(results, params):
    """Создание графиков"""
    # Траектория
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=results['X'], y=results['Y'],
        mode='lines', name='Траектория',
        line=dict(color='blue', width=3)
    ))
    fig1.update_layout(
        title='Траектория движения',
        xaxis_title='X координата, м',
        yaxis_title='Y координата, м',
        height=400
    )
    
    # Параметры движения
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Угол скольжения β', 'Угловая скорость r', 
                      'Курсовой угол ψ', 'Поперечное ускорение')
    )
    
    fig2.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['beta']),
                  name='β', line=dict(color='green')),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['angular_velocity']),
                  name='r', line=dict(color='magenta')),
        row=1, col=2
    )
    
    fig2.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['yaw_angle']),
                  name='ψ', line=dict(color='cyan')),
        row=2, col=1
    )
    
    # Поперечное ускорение
    dt = np.diff(results['time'])
    dbeta_dt = np.diff(results['beta']) / dt
    dbeta_dt = np.append(dbeta_dt, dbeta_dt[-1])
    ay = params['speed'] * (dbeta_dt + results['angular_velocity'])
    
    fig2.add_trace(
        go.Scatter(x=results['time'], y=ay,
                  name='ay', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig2.update_layout(height=600, showlegend=False)
    
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    # Инициализация session_state
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    
    main()