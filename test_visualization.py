# tests/test_visualization.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.vehicle_parameters import VehicleParameters
from src.linear_bicycle_model import simulate_maneuver, get_simulation_results
from src.steering_functions import step_steer
from src.visualization.plotter import ResultPlotter
from src.maneuvers import StandardManeuvers

def test_plotter_creation():
    """Тест создания объекта визуализации"""
    print("🧪 Тест 1: Создание plotter")
    plotter = ResultPlotter()
    assert plotter is not None, "Не удалось создать объект ResultPlotter"
    print("   ✓ ResultPlotter создан успешно")

def test_plot_generation():
    """Тест генерации графиков"""
    print("🧪 Тест 2: Генерация графиков")
    
    # Создаем тестовые данные
    params = VehicleParameters()
    solution = simulate_maneuver(
        params=params,
        delta_func=lambda t: step_steer(t, 0.1, 1.0),
        V=params.V,
        t_span=(0, 5)
    )
    results = get_simulation_results(solution)
    
    # Создаем графики
    plotter = ResultPlotter(figsize=(12, 8))
    fig = plotter.plot_comprehensive_results(results, lambda t: step_steer(t, 0.1, 1.0))
    
    assert fig is not None, "Не удалось создать график"
    assert len(fig.axes) > 0, "График не содержит осей"
    
    print("   ✓ Графики сгенерированы успешно")
    
    # Закрываем график чтобы не засорять память
    plt.close(fig)

def test_maneuvers_creation():
    """Тест создания стандартных маневров"""
    print("🧪 Тест 3: Создание маневров")
    
    maneuvers = StandardManeuvers.get_all_maneuvers()
    assert len(maneuvers) > 0, "Не создано ни одного маневра"
    
    # Проверяем, что у каждого маневра есть необходимые поля
    for maneuver in maneuvers:
        assert 'name' in maneuver, "Маневр должен иметь имя"
        assert 'delta_func' in maneuver, "Маневр должен иметь функцию руления"
        assert 't_span' in maneuver, "Маневр должен иметь временной интервал"
    
    print(f"   ✓ Создано {len(maneuvers)} стандартных маневров")

def test_performance_analysis():
    """Тест анализа производительности"""
    print("🧪 Тест 4: Анализ производительности")
    
    params = VehicleParameters()
    maneuver = StandardManeuvers.step_steer_maneuver()
    solution = simulate_maneuver(
        params=params,
        delta_func=maneuver['delta_func'],
        V=params.V,
        t_span=maneuver['t_span']
    )
    results = get_simulation_results(solution)
    
    from src.maneuvers import analyze_maneuver_performance
    performance = analyze_maneuver_performance(results, "Тестовый маневр")
    
    required_keys = ['maneuver', 'completion_time', 'final_displacement', 
                    'max_sideslip_angle', 'max_angular_velocity']
    
    for key in required_keys:
        assert key in performance, f"В результатах анализа отсутствует ключ {key}"
    
    print("   ✓ Анализ производительности работает корректно")

def run_all_visualization_tests():
    """Запуск всех тестов визуализации"""
    print("🚀 ЗАПУСК ТЕСТОВ НЕДЕЛИ 2")
    print("=" * 40)
    
    test_plotter_creation()
    test_plot_generation()
    test_maneuvers_creation()
    test_performance_analysis()
    
    print("=" * 40)
    print("🎉 ВСЕ ТЕСТЫ ВИЗУАЛИЗАЦИИ ПРОЙДЕНЫ УСПЕШНО!")

if __name__ == "__main__":
    run_all_visualization_tests()