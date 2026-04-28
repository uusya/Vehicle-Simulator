# examples/week3_minimal_demo.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

# Прямые импорты чтобы избежать проблем с __init__.py
from src.vehicle_parameters import VehicleParameters
from src.linear_bicycle_model import simulate_maneuver, get_simulation_results
from src.nonlinear_bicycle_model import create_nonlinear_simulator, get_simulation_results_nonlinear
from src.maneuvers import StandardManeuvers

def minimal_demo():
    """Минимальная демонстрация недели 3"""
    print("🚗 МИНИМАЛЬНАЯ ДЕМОНСТРАЦИЯ НЕДЕЛИ 3")
    print("=" * 50)
    
    # 1. Настройка параметров
    print("\n1. 📋 НАСТРОЙКА АВТОМОБИЛЯ")
    params = VehicleParameters()
    print("   ✓ Параметры автомобиля загружены")
    
    # 2. Сравнение линейной и нелинейной моделей
    print("\n2. 🔄 СРАВНЕНИЕ ЛИНЕЙНОЙ И НЕЛИНЕЙНОЙ МОДЕЛЕЙ")
    
    maneuver = StandardManeuvers.step_steer_maneuver(amplitude=0.15)
    print(f"   Маневр: {maneuver['name']}")
    
    # Линейная модель
    print("   Запуск линейной модели...")
    try:
        linear_solution = simulate_maneuver(
            params=params,
            delta_func=maneuver['delta_func'],
            V=params.V,
            t_span=maneuver['t_span']
        )
        linear_results = get_simulation_results(linear_solution)
        print(f"     ✓ Успешно: {len(linear_results['time'])} точек")
    except Exception as e:
        print(f"     ✗ Ошибка: {e}")
        return
    
    # Нелинейная модель
    print("   Запуск нелинейной модели...")
    try:
        nonlinear_simulator = create_nonlinear_simulator(
            params, 
            front_stiffness=params.C_f,
            rear_stiffness=params.C_r,
            tire_model_type='piecewise'
        )
        nonlinear_solution = nonlinear_simulator.simulate(
            delta_func=maneuver['delta_func'],
            V=params.V,
            t_span=maneuver['t_span']
        )
        nonlinear_results = get_simulation_results_nonlinear(nonlinear_solution)
        print(f"     ✓ Успешно: {len(nonlinear_results['time'])} точек")
    except Exception as e:
        print(f"     ✗ Ошибка: {e}")
        return
    
    # 3. Простая визуализация
    print("\n3. 📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    
    plt.figure(figsize=(12, 4))
    
    # Траектории
    plt.subplot(1, 2, 1)
    plt.plot(linear_results['X'], linear_results['Y'], 'b-', linewidth=2, label='Линейная')
    plt.plot(nonlinear_results['X'], nonlinear_results['Y'], 'r-', linewidth=2, label='Нелинейная')
    plt.xlabel('X, м')
    plt.ylabel('Y, м')
    plt.title('Сравнение траекторий')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Углы скольжения
    plt.subplot(1, 2, 2)
    plt.plot(linear_results['time'], np.degrees(linear_results['beta']), 'b-', linewidth=2, label='Линейная')
    plt.plot(nonlinear_results['time'], np.degrees(nonlinear_results['beta']), 'r-', linewidth=2, label='Нелинейная')
    plt.xlabel('Время, с')
    plt.ylabel('Угол скольжения β, °')
    plt.title('Углы бокового скольжения')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week3_minimal_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ График сравнения создан")
    
    # 4. Простой анализ различий
    print("\n4. 📈 АНАЛИЗ РАЗЛИЧИЙ")
    
    max_beta_linear = np.max(np.abs(np.degrees(linear_results['beta'])))
    max_beta_nonlinear = np.max(np.abs(np.degrees(nonlinear_results['beta'])))
    
    max_r_linear = np.max(np.abs(np.degrees(linear_results['angular_velocity'])))
    max_r_nonlinear = np.max(np.abs(np.degrees(nonlinear_results['angular_velocity'])))
    
    final_y_linear = linear_results['Y'][-1]
    final_y_nonlinear = nonlinear_results['Y'][-1]
    
    print("   МАКСИМАЛЬНЫЕ ЗНАЧЕНИЯ:")
    print(f"     Угол скольжения: {max_beta_linear:.2f}° (лин.) → {max_beta_nonlinear:.2f}° (нелин.)")
    print(f"     Угловая скорость: {max_r_linear:.2f}°/с (лин.) → {max_r_nonlinear:.2f}°/с (нелин.)")
    print(f"     Поперечное смещение: {final_y_linear:.2f}м (лин.) → {final_y_nonlinear:.2f}м (нелин.)")
    
    beta_diff_percent = (max_beta_nonlinear - max_beta_linear) / max_beta_linear * 100
    print(f"     Разница в угле скольжения: {beta_diff_percent:+.1f}%")
    
    print("\n" + "=" * 50)
    print("✅ МИНИМАЛЬНАЯ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 50)

if __name__ == "__main__":
    minimal_demo()