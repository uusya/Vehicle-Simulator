# examples/week2_analysis.ipynb
"""
# НЕДЕЛЯ 2: ИНТЕРАКТИВНЫЙ АНАЛИЗ ДИНАМИКИ АВТОМОБИЛЯ

Этот ноутбук позволяет интерактивно исследовать поведение автомобиля при различных маневрах.
"""

# Импорт необходимых библиотек
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from src.vehicle_parameters import VehicleParameters
from src.linear_bicycle_model import simulate_maneuver, get_simulation_results
from src.maneuvers import StandardManeuvers
from src.visualization.plotter import ResultPlotter
from src.visualization.animator import VehicleAnimator

# Настройка отображения графиков
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

print("🚗 ИНТЕРАКТИВНЫЙ АНАЛИЗ ДИНАМИКИ АВТОМОБИЛЯ")
print("Используйте слайдеры для изменения параметров и наблюдения за изменением поведения автомобиля")

def interactive_maneuver_analysis(
    maneuver_type='Шаг рулем',
    steering_amplitude=0.1,
    frequency=0.5,
    car_speed=20.0,
    front_stiffness=80000.0,
    rear_stiffness=100000.0
):
    """
    Интерактивный анализ маневра с возможностью изменения параметров
    """
    # Создание параметров автомобиля
    params = VehicleParameters()
    params.V = car_speed
    params.C_f = front_stiffness
    params.C_r = rear_stiffness
    
    # Выбор маневра
    if maneuver_type == 'Шаг рулем':
        maneuver = StandardManeuvers.step_steer_maneuver(
            amplitude=steering_amplitude
        )
    elif maneuver_type == 'Переставка':
        maneuver = StandardManeuvers.lane_change_maneuver(
            amplitude=steering_amplitude,
            frequency=frequency
        )
    elif maneuver_type == 'Двойная переставка':
        maneuver = StandardManeuvers.double_lane_change_maneuver()
    else:
        maneuver = StandardManeuvers.constant_radius_maneuver()
    
    # Запуск симуляции
    solution = simulate_maneuver(
        params=params,
        delta_func=maneuver['delta_func'],
        V=params.V,
        t_span=maneuver['t_span']
    )
    
    results = get_simulation_results(solution)
    
    # Визуализация результатов
    plotter = ResultPlotter()
    fig = plotter.plot_comprehensive_results(
        results, 
        maneuver['delta_func'],
        title=f"Маневр: {maneuver['name']}\n"
              f"Скорость: {car_speed} м/с, "
              f"Жесткости: C_f={front_stiffness/1000:.0f}кН/рад, C_r={rear_stiffness/1000:.0f}кН/рад"
    )
    
    plt.show()
    
    # Вывод основных характеристик
    print(f"\n📊 ОСНОВНЫЕ ХАРАКТЕРИСТИКИ МАНЕВРА:")
    print(f"   Максимальный угол скольжения: {np.max(np.abs(np.degrees(results['beta']))):.2f}°")
    print(f"   Максимальная угловая скорость: {np.max(np.abs(np.degrees(results['angular_velocity']))):.2f} °/с")
    print(f"   Конечное поперечное смещение: {results['Y'][-1]:.2f} м")

# Создание интерактивного интерфейса
interact(
    interactive_maneuver_analysis,
    maneuver_type=widgets.Dropdown(
        options=['Шаг рулем', 'Переставка', 'Двойная переставка', 'Постоянный радиус'],
        value='Шаг рулем',
        description='Тип маневра:'
    ),
    steering_amplitude=widgets.FloatSlider(
        value=0.1,
        min=0.01,
        max=0.2,
        step=0.01,
        description='Амплитуда руля (рад):'
    ),
    frequency=widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=2.0,
        step=0.1,
        description='Частота (Гц):'
    ),
    car_speed=widgets.FloatSlider(
        value=20.0,
        min=5.0,
        max=40.0,
        step=5.0,
        description='Скорость (м/с):'
    ),
    front_stiffness=widgets.FloatSlider(
        value=80000.0,
        min=40000.0,
        max=160000.0,
        step=10000.0,
        description='Жесткость перед. (Н/рад):'
    ),
    rear_stiffness=widgets.FloatSlider(
        value=100000.0,
        min=50000.0,
        max=200000.0,
        step=10000.0,
        description='Жесткость зад. (Н/рад):'
    )
)