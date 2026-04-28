# examples/week5_demo.py
"""
Демонстрация возможностей недели 5
Запуск: streamlit run week5_demo.py
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
    from src.visualization.three_d_animator import ThreeDAnimator
    from src.export.data_exporter import DataExporter
    from src.analysis.advanced_stability import AdvancedStabilityAnalysis
    
    IMPORTS_WORKING = True
except ImportError as e:
    st.error(f"❌ Ошибка импорта: {e}")
    IMPORTS_WORKING = False

def main():
    """Основное приложение недели 5"""
    
    st.set_page_config(
        page_title="🏎️ Симулятор управляемости - Неделя 5",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title("🏎️ Симулятор управляемости автомобиля - Неделя 5")
    st.markdown("### 📈 Расширенная визуализация и анализ")
    st.markdown("---")
    
    if not IMPORTS_WORKING:
        st.error("Не все модули загружены корректно")
        return
    
    # Инициализация компонентов
    if 'three_d_animator' not in st.session_state:
        st.session_state.three_d_animator = ThreeDAnimator()
    if 'data_exporter' not in st.session_state:
        st.session_state.data_exporter = DataExporter()
    if 'stability_analyzer' not in st.session_state:
        st.session_state.stability_analyzer = AdvancedStabilityAnalysis()
    
    # Навигация по разделам
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
    """Основная симуляция с расширенными возможностями"""
    st.header("🚗 Основная симуляция")
    
    # Боковая панель
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
            analysis_type = st.selectbox(
                "Тип анализа:",
                ["Базовый", "Энергетический", "Полный"]
            )
        
        if st.button("🚀 Запустить симуляцию", use_container_width=True):
            st.session_state.run_simulation = True
            st.session_state.params = {
                'mass': mass, 'speed': speed, 'cog_ratio': cog_ratio,
                'cf': cf, 'cr': cr, 'maneuver': maneuver_type, 
                'tire_model': tire_model, 'sim_time': sim_time,
                'analysis_type': analysis_type
            }
    
    # Основная область
    if st.session_state.get('run_simulation', False):
        run_advanced_simulation(st.session_state.params)
    else:
        show_simulation_welcome()

def show_simulation_welcome():
    """Приветственный экран симуляции"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Расширенные возможности недели 5
        
        **Новые функции:**
        - 📊 3D визуализация траектории
        - 🔬 Энергетический анализ
        - 📈 Продвинутый анализ устойчивости  
        - 💾 Полный экспорт результатов
        - 🎨 Интерактивные дашборды
        
        **Для начала работы:**
        1. Настройте параметры в боковой панели
        2. Выберите тип анализа
        3. Запустите симуляцию
        4. Исследуйте результаты в разных вкладках
        """)
    
    with col2:
        fig = create_advanced_car_schematic()
        st.plotly_chart(fig, use_container_width=True)

def create_advanced_car_schematic():
    """Улучшенная схема автомобиля"""
    fig = go.Figure()
    
    # Кузов с 3D эффектом
    fig.add_shape(type="rect", x0=-2, y0=-1, x1=2, y1=1,
                 line=dict(color="blue", width=3), fillcolor="lightblue", opacity=0.7)
    
    # Колеса
    wheel_positions = [(-2.2, -1.2, -1.8, -0.8), (1.8, -1.2, 2.2, -0.8),
                      (-2.2, 0.8, -1.8, 1.2), (1.8, 0.8, 2.2, 1.2)]
    
    for x0, y0, x1, y1 in wheel_positions:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                     line=dict(color="black", width=2), fillcolor="black")
    
    # Векторы сил
    fig.add_annotation(x=0, y=0, ax=0, ay=50, xref='x', yref='y',
                      axref='x', ayref='y', text='', showarrow=True,
                      arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='red')
    
    fig.update_layout(
        title="Схема автомобиля с силами",
        xaxis=dict(range=[-3, 3], showticklabels=False),
        yaxis=dict(range=[-2, 2], showticklabels=False),
        showlegend=False,
        height=300
    )
    
    return fig

def run_advanced_simulation(params):
    """Запуск расширенной симуляции"""
    st.header("📊 Результаты расширенной симуляции")
    
    # Создание параметров
    vehicle_params = create_default_parameters()
    vehicle_params.m = params['mass']
    vehicle_params.V = params['speed']
    
    L = vehicle_params.a + vehicle_params.b
    vehicle_params.a = params['cog_ratio'] * L
    vehicle_params.b = L - vehicle_params.a
    
    vehicle_params.C_f = params['cf']
    vehicle_params.C_r = params['cr']
    
    # Модель шин
    tire_model_map = {
        "Линейная": "linear",
        "Кусочно-линейная": "piecewise", 
        "Магическая формула": "pacejka"
    }
    
    # Прогресс
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
        
        # Маневр
        maneuver = get_maneuver(params['maneuver'], params['sim_time'])
        
        # Симуляция
        solution = simulator.simulate(
            maneuver['delta_func'], params['speed'], maneuver['t_span']
        )
        
        status_text.text("📊 Анализ результатов...")
        progress_bar.progress(80)
        
        results = get_simulation_results_nonlinear(solution)
        
        # Расширенный анализ
        if params['analysis_type'] in ["Энергетический", "Полный"]:
            energy_fig, energy_metrics = st.session_state.three_d_animator.create_energy_analysis_plot(
                results, vehicle_params
            )
        else:
            energy_fig, energy_metrics = None, None
        
        progress_bar.progress(100)
        
        # Сохранение результатов в session_state
        st.session_state.last_results = results
        st.session_state.last_params = params
        st.session_state.last_maneuver = maneuver
        st.session_state.energy_metrics = energy_metrics
        
        # Отображение результатов
        display_advanced_results(results, params, energy_fig, energy_metrics)
        status_text.text("✅ Симуляция завершена!")
        
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        status_text.text("❌ Ошибка симуляции")

def get_maneuver(maneuver_name, sim_time):
    """Получение маневра"""
    maneuver_map = {
        "Шаг рулем 5°": StandardManeuvers.step_steer_maneuver(amplitude=np.radians(5), duration=sim_time),
        "Шаг рулем 10°": StandardManeuvers.step_steer_maneuver(amplitude=np.radians(10), duration=sim_time),
        "Переставка": StandardManeuvers.lane_change_maneuver(amplitude=0.05, duration=sim_time),
        "Постоянный радиус": StandardManeuvers.constant_radius_maneuver(radius=100, duration=sim_time)
    }
    return maneuver_map[maneuver_name]

def display_advanced_results(results, params, energy_fig, energy_metrics):
    """Отображение расширенных результатов"""
    # Основные метрики
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
    create_advanced_plots(results, params, energy_fig)

def create_advanced_plots(results, params, energy_fig):
    """Создание расширенных графиков"""
    # Основные графики
    fig_basic = create_basic_plots(results, params)
    st.plotly_chart(fig_basic, use_container_width=True)
    
    # Энергетические графики
    if energy_fig:
        st.plotly_chart(energy_fig, use_container_width=True)

def create_basic_plots(results, params):
    """Базовые графики"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Траектория движения', 'Угол скольжения β',
                      'Угловая скорость r', 'Фазовый портрет')
    )
    
    # Траектория
    fig.add_trace(
        go.Scatter(x=results['X'], y=results['Y'], name='Траектория',
                  line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Угол скольжения
    fig.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['beta']), name='β',
                  line=dict(color='green')),
        row=1, col=2
    )
    
    # Угловая скорость
    fig.add_trace(
        go.Scatter(x=results['time'], y=np.degrees(results['angular_velocity']), name='r',
                  line=dict(color='magenta')),
        row=2, col=1
    )
    
    # Фазовый портрет
    fig.add_trace(
        go.Scatter(x=np.degrees(results['beta']), y=np.degrees(results['angular_velocity']),
                  name='Фазовый портрет', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def show_3d_visualization():
    """3D визуализация"""
    st.header("📊 3D Визуализация")
    
    if 'last_results' not in st.session_state:
        st.warning("⚠️ Сначала выполните симуляцию во вкладке 'Основная симуляция'")
        return
    
    results = st.session_state.last_results
    params = st.session_state.last_params
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3D Траектория")
        fig_3d = st.session_state.three_d_animator.create_3d_trajectory(
            results, "3D Траектория движения"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        st.subheader("Анимированное движение")
        fig_anim = st.session_state.three_d_animator.create_animated_3d_car(results)
        st.plotly_chart(fig_anim, use_container_width=True)
    
    # Дополнительная 3D информация
    with st.expander("📐 Дополнительные 3D проекции"):
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
    """Анализ устойчивости"""
    st.header("🔬 Анализ устойчивости")
    
    if 'last_params' not in st.session_state:
        st.warning("⚠️ Сначала выполните симуляцию во вкладке 'Основная симуляция'")
        return
    
    params = st.session_state.last_params
    
    # Создание параметров для анализа
    vehicle_params = create_default_parameters()
    vehicle_params.m = params['mass']
    
    L = vehicle_params.a + vehicle_params.b
    vehicle_params.a = params['cog_ratio'] * L
    vehicle_params.b = L - vehicle_params.a
    
    vehicle_params.C_f = params['cf']
    vehicle_params.C_r = params['cr']
    
    # Настройки анализа
    col1, col2 = st.columns(2)
    
    with col1:
        min_speed = st.slider("Минимальная скорость (м/с)", 1, 20, 5)
        max_speed = st.slider("Максимальная скорость (м/с)", 10, 60, 40)
    
    with col2:
        num_points = st.slider("Количество точек", 10, 100, 50)
        analysis_type = st.selectbox(
            "Тип анализа:",
            ["Базовый", "Расширенный", "Полный"]
        )
    
    if st.button("🔄 Выполнить анализ устойчивости"):
        with st.spinner("🔬 Выполняется анализ устойчивости..."):
            # Анализ устойчивости
            stability_data = st.session_state.stability_analyzer.analyze_stability_modes(
                vehicle_params, (min_speed, max_speed), num_points
            )
            
            # Создание дашборда
            stability_fig = st.session_state.stability_analyzer.create_stability_dashboard(
                stability_data
            )
            
            st.plotly_chart(stability_fig, use_container_width=True)
            
            # Индикаторы управляемости
            st.subheader("📈 Индикаторы управляемости")
            indicators = st.session_state.stability_analyzer.calculate_handling_indicators(
                vehicle_params, params['speed']
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Градиент недостаточной поворачиваемости", 
                         f"{indicators['understeer_gradient']:.4f}")
            with col2:
                st.metric("Критическая скорость", 
                         f"{indicators['critical_speed']:.1f} м/с" if indicators['critical_speed'] != float('inf') else "∞")
            with col3:
                st.metric("Запас устойчивости", f"{indicators['stability_margin']:.4f}")
            with col4:
                st.metric("Коэффициент демпфирования", f"{indicators['damping_ratio']:.3f}")

def show_data_export():
    """Экспорт данных"""
    st.header("💾 Экспорт данных")
    
    if 'last_results' not in st.session_state:
        st.warning("⚠️ Сначала выполните симуляцию во вкладке 'Основная симуляция'")
        return
    
    results = st.session_state.last_results
    params = st.session_state.last_params
    maneuver = st.session_state.last_maneuver
    energy_metrics = st.session_state.get('energy_metrics', None)
    
    st.subheader("📤 Варианты экспорта")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Экспорт в CSV", use_container_width=True):
            with st.spinner("Экспорт в CSV..."):
                csv_file = st.session_state.data_exporter.export_to_csv(
                    results, params, maneuver['name']
                )
                st.success(f"✅ Данные экспортированы в: {csv_file}")
    
    with col2:
        if st.button("📊 Экспорт в Excel", use_container_width=True):
            with st.spinner("Экспорт в Excel..."):
                excel_file = st.session_state.data_exporter.export_to_excel(
                    results, params, maneuver['name'], energy_metrics
                )
                st.success(f"✅ Данные экспортированы в: {excel_file}")
    
    with col3:
        if st.button("📑 Полный отчет", use_container_width=True):
            with st.spinner("Генерация полного отчета..."):
                # Создание графиков для отчета
                plots = {
                    'trajectory': create_basic_plots(results, params),
                    'energy_analysis': st.session_state.three_d_animator.create_energy_analysis_plot(results, params)[0] if energy_metrics else None
                }
                
                report_dir, files = st.session_state.data_exporter.generate_report(
                    results, params, maneuver['name'], energy_metrics, plots
                )
                st.success(f"✅ Полный отчет создан в: {report_dir}")
    
    # Предпросмотр данных
    st.subheader("👀 Предпросмотр данных")
    
    if st.checkbox("Показать таблицу данных"):
        import pandas as pd
        
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
            st.metric("Временной шаг", f"{results['time'][1] - results['time'][0]:.3f} с")
        with col3:
            st.metric("Общее время", f"{results['time'][-1]:.1f} с")

if __name__ == "__main__":
    # Инициализация session_state
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    
    main()