# examples/week3_simple_demo.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.vehicle_parameters import VehicleParameters
from src.linear_bicycle_model import simulate_maneuver, get_simulation_results
from src.nonlinear_bicycle_model import create_nonlinear_simulator, get_simulation_results_nonlinear
from src.tire_models import create_tire_model_comparison
from src.maneuvers import StandardManeuvers
from src.analysis_tools import HandlingAnalysis, print_handling_analysis_report

def main():
    print("🚗 ПРОСТАЯ ДЕМОНСТРАЦИЯ НЕДЕЛИ 3: НЕЛИНЕЙНАЯ МОДЕЛЬ")
    print("=" * 60)
    
    # 1. Сравнение моделей шин
    print("\n1. 🔬 СРАВНЕНИЕ МОДЕЛЕЙ ШИН")
    tire_fig = create_tire_model_comparison()
    plt.show()
    
    # 2. Создание параметров автомобиля
    print("\n2. 📋 НАСТРОЙКА АВТОМОБИЛЯ")
    params = VehicleParameters()
    print(params.get_summary())
    
    # 3. Сравнение линейной и нелинейной моделей
    print("\n3. 🔄 СРАВНЕНИЕ ЛИНЕЙНОЙ И НЕЛИНЕЙНОЙ МОДЕЛЕЙ")
    
    # Маневр с большим углом чтобы проявились нелинейности
    maneuver = StandardManeuvers.step_steer_maneuver(amplitude=0.2)  # ~11.5°
    
    # Линейная модель
    print("   Запуск линейной модели...")
    linear_solution = simulate_maneuver(
        params=params,
        delta_func=maneuver['delta_func'],
        V=params.V,
        t_span=maneuver['t_span']
    )
    linear_results = get_simulation_results(linear_solution)
    
    # Нелинейная модель (кусочно-линейная)
    print("   Запуск нелинейной модели...")
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
    
    # 4. Визуализация сравнения
    print("\n4. 📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Траектории
    ax1.plot(linear_results['X'], linear_results['Y'], 'b-', linewidth=2, label='Линейная')
    ax1.plot(nonlinear_results['X'], nonlinear_results['Y'], 'r-', linewidth=2, label='Нелинейная')
    ax1.set_xlabel('X, м')
    ax1.set_ylabel('Y, м')
    ax1.set_title('Сравнение траекторий')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Углы скольжения
    ax2.plot(linear_results['time'], np.degrees(linear_results['beta']), 'b-', linewidth=2, label='Линейная')
    ax2.plot(nonlinear_results['time'], np.degrees(nonlinear_results['beta']), 'r-', linewidth=2, label='Нелинейная')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Угол скольжения β, °')
    ax2.set_title('Углы бокового скольжения')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Угловые скорости
    ax3.plot(linear_results['time'], np.degrees(linear_results['angular_velocity']), 'b-', linewidth=2, label='Линейная')
    ax3.plot(nonlinear_results['time'], np.degrees(nonlinear_results['angular_velocity']), 'r-', linewidth=2, label='Нелинейная')
    ax3.set_xlabel('Время, с')
    ax3.set_ylabel('Угловая скорость r, °/с')
    ax3.set_title('Угловые скорости')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Анализ различий
    print("\n5. 📈 АНАЛИЗ РАЗЛИЧИЙ")
    analysis = HandlingAnalysis.analyze_handling_characteristics(
        linear_results, nonlinear_results, maneuver['name']
    )
    print_handling_analysis_report(analysis)
    
    # 6. Градиент недостаточной поворачиваемости
    print("\n6. 🎯 АНАЛИЗ ПОВОРАЧИВАЕМОСТИ")
    K = HandlingAnalysis.calculate_understeer_gradient(params)
    print(f"   Градиент недостаточной поворачиваемости: K = {K:.6f} рад/с²/м")
    
    if K > 0.001:
        print("   ↳ Автомобиль обладает НЕДОСТАТОЧНОЙ ПОВОРАЧИВАЕМОСТЬЮ")
    elif K < -0.001:
        print("   ↳ Автомобиль обладает ИЗБЫТОЧНОЙ ПОВОРАЧИВАЕМОСТЬЮ")
    else:
        print("   ↳ Автомобиль обладает НЕЙТРАЛЬНОЙ ПОВОРАЧИВАЕМОСТЬЮ")
    
    print("\n" + "=" * 60)
    print("✅ ДЕМОНСТРАЦИЯ НЕДЕЛИ 3 ЗАВЕРШЕНА УСПЕШНО!")
    print("🎯 ЧТО МЫ СДЕЛАЛИ:")
    print("   - Реализовали нелинейные модели шин")
    print("   - Создали нелинейную модель велосипеда") 
    print("   - Показали различия между линейной и нелинейной динамикой")
    print("   - Проанализировали характеристики управляемости")
    print("=" * 60)

if __name__ == "__main__":
    main()