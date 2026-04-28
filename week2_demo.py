# examples/week2_demo.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.vehicle_parameters import VehicleParameters
from src.linear_bicycle_model import simulate_maneuver, get_simulation_results
from src.maneuvers import StandardManeuvers, analyze_maneuver_performance, print_maneuver_performance
from src.visualization.plotter import ResultPlotter, create_comparison_plot
from src.visualization.animator import VehicleAnimator

def main():
    print("🚗 ДЕМОНСТРАЦИЯ НЕДЕЛИ 2: ВИЗУАЛИЗАЦИЯ И АНАЛИЗ")
    print("=" * 60)
    
    # 1. Создание параметров автомобиля
    print("\n1. 📋 НАСТРОЙКА ПАРАМЕТРОВ АВТОМОБИЛЯ")
    params = VehicleParameters()
    print(params.get_summary())
    
    # 2. Запуск различных маневров
    print("\n2. 🎮 ЗАПУСК СТАНДАРТНЫХ МАНЕВРОВ")
    maneuvers = [
        StandardManeuvers.step_steer_maneuver(amplitude=0.1),
        StandardManeuvers.lane_change_maneuver(amplitude=0.05, frequency=0.5),
        StandardManeuvers.double_lane_change_maneuver()
    ]
    
    all_results = []
    performance_data = []
    
    for i, maneuver in enumerate(maneuvers):
        print(f"\n   Маневр {i+1}: {maneuver['name']}")
        print(f"   Описание: {maneuver['description']}")
        
        # Запуск симуляции
        solution = simulate_maneuver(
            params=params,
            delta_func=maneuver['delta_func'],
            V=params.V,
            t_span=maneuver['t_span']
        )
        
        results = get_simulation_results(solution)
        all_results.append(results)
        
        # Анализ производительности
        performance = analyze_maneuver_performance(results, maneuver['name'])
        performance_data.append(performance)
        print_maneuver_performance(performance)
    
    # 3. Визуализация результатов
    print("\n3. 📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    
    # 3.1. Комплексные графики для каждого маневра
    plotter = ResultPlotter()
    for i, (results, maneuver) in enumerate(zip(all_results, maneuvers)):
        fig = plotter.plot_comprehensive_results(
            results, 
            maneuver['delta_func'],
            title=f"Маневр: {maneuver['name']}"
        )
        plt.savefig(f'maneuver_{i+1}_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3.2. Сравнительные графики
    maneuver_names = [m['name'] for m in maneuvers]
    fig_comparison = create_comparison_plot(all_results, maneuver_names)
    plt.savefig('maneuvers_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Создание анимации
    print("\n4. 🎬 СОЗДАНИЕ АНИМАЦИИ")
    print("   Создание анимации для маневра 'Переставка'...")
    
    lane_change_results = all_results[1]  # Берем маневр переставки
    lane_change_maneuver = maneuvers[1]
    
    animator = VehicleAnimator(
        lane_change_results, 
        params, 
        lane_change_maneuver['delta_func']
    )
    
    # Создаем упрощенную анимацию для демонстрации
    anim = animator.create_simple_animation(interval=100)
    plt.show()
    
    print("\n   💡 Совет: Для сохранения анимации раскомментируйте строку с anim.save()")
    
    # 5. Сводный отчет
    print("\n5. 📈 СВОДНЫЙ ОТЧЕТ ПО МАНЕВРАМ")
    print("=" * 60)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    print(f"{'Маневр':<30} {'Смещение, м':<12} {'Макс. β, °':<10} {'Макс. r, °/с':<12}")
    print("-" * 60)
    
    for perf in performance_data:
        print(f"{perf['maneuver']:<30} {perf['final_displacement']:>10.2f} "
              f"{perf['max_sideslip_angle']:>9.2f} {perf['max_angular_velocity']:>11.2f}")
    
    print("=" * 60)
    
    # 6. Анализ влияния параметров
    print("\n6. 🔬 АНАЛИЗ ВЛИЯНИЯ ПАРАМЕТРОВ")
    analyze_parameter_sensitivity(params)
    
    print("\n" + "=" * 60)
    print("✅ НЕДЕЛЯ 2 ЗАВЕРШЕНА УСПЕШНО!")
    print("🎯 ЧТО МЫ СОЗДАЛИ:")
    print("   - Систему визуализации с комплексными графиками")
    print("   - Анимацию движения автомобиля")
    print("   - Библиотеку стандартных маневров")
    print("   - Инструменты анализа производительности")
    print("=" * 60)

def analyze_parameter_sensitivity(base_params):
    """Анализ чувствительности к изменению параметров"""
    print("\n   Анализ влияния жесткости шин на маневр 'Шаг рулем':")
    
    stiffness_factors = [0.5, 1.0, 2.0]  # Коэффициенты изменения жесткости
    results_stiffness = []
    labels_stiffness = []
    
    for factor in stiffness_factors:
        # Создаем копию параметров с измененной жесткостью
        params_modified = VehicleParameters()
        params_modified.C_f = base_params.C_f * factor
        params_modified.C_r = base_params.C_r * factor
        
        maneuver = StandardManeuvers.step_steer_maneuver(amplitude=0.1)
        solution = simulate_maneuver(
            params=params_modified,
            delta_func=maneuver['delta_func'],
            V=base_params.V,
            t_span=maneuver['t_span']
        )
        
        results_stiffness.append(get_simulation_results(solution))
        labels_stiffness.append(f'Жесткость ×{factor}')
    
    # Строим сравнительный график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, results in enumerate(results_stiffness):
        ax1.plot(results['time'], np.degrees(results['beta']), 
                label=labels_stiffness[i], linewidth=2)
        ax2.plot(results['time'], np.degrees(results['angular_velocity']),
                label=labels_stiffness[i], linewidth=2)
    
    ax1.set_xlabel('Время, с')
    ax1.set_ylabel('Угол скольжения β, °')
    ax1.set_title('Влияние жесткости шин на угол скольжения')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Угловая скорость r, °/с')
    ax2.set_title('Влияние жесткости шин на угловую скорость')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stiffness_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()