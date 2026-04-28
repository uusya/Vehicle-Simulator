# examples/week3_demo.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.vehicle_parameters import VehicleParameters
from src.linear_bicycle_model import simulate_maneuver, get_simulation_results
from src.nonlinear_bicycle_model import create_nonlinear_simulator, get_simulation_results_nonlinear
from src.tire_models import LinearTireModel, FialaTireModel, PiecewiseLinearTireModel, create_tire_model_comparison
from src.maneuvers import StandardManeuvers
from src.analysis_tools import HandlingAnalysis, StabilityAnalysis, print_handling_analysis_report
from src.visualization.dashboard import AdvancedDashboard

def main():
    print("🚗 ДЕМОНСТРАЦИЯ НЕДЕЛИ 3: НЕЛИНЕЙНАЯ МОДЕЛЬ И АНАЛИЗ УПРАВЛЯЕМОСТИ")
    print("=" * 70)
    
    # 1. Сравнение моделей шин
    print("\n1. 🔬 СРАВНЕНИЕ МОДЕЛЕЙ ШИН")
    tire_fig = create_tire_model_comparison()
    plt.savefig('tire_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Создание параметров автомобиля
    print("\n2. 📋 НАСТРОЙКА АВТОМОБИЛЯ")
    params = VehicleParameters()
    print(params.get_summary())
    
    # 3. Сравнение линейной и нелинейной моделей для разных маневров
    print("\n3. 🔄 СРАВНЕНИЕ ЛИНЕЙНОЙ И НЕЛИНЕЙНОЙ МОДЕЛЕЙ")
    maneuvers_to_test = [
        StandardManeuvers.step_steer_maneuver(amplitude=0.05),
        StandardManeuvers.step_steer_maneuver(amplitude=0.15),  # Больший угол
        StandardManeuvers.lane_change_maneuver(amplitude=0.08, frequency=0.3),
    ]
    
    dashboard = AdvancedDashboard()
    
    for i, maneuver in enumerate(maneuvers_to_test):
        print(f"\n   Маневр {i+1}: {maneuver['name']}")
        
        # Линейная модель
        linear_solution = simulate_maneuver(
            params=params,
            delta_func=maneuver['delta_func'],
            V=params.V,
            t_span=maneuver['t_span']
        )
        linear_results = get_simulation_results(linear_solution)
        
        # Нелинейная модель (Фьяла)
        nonlinear_simulator = create_nonlinear_simulator(
            params, 
            front_stiffness=params.C_f,
            rear_stiffness=params.C_r,
            tire_model_type='fiala'
        )
        nonlinear_solution = nonlinear_simulator.simulate(
            delta_func=maneuver['delta_func'],
            V=params.V,
            t_span=maneuver['t_span']
        )
        nonlinear_results = get_simulation_results_nonlinear(nonlinear_solution)
        
        # Создание дашборда сравнения
        comparison_fig = dashboard.create_model_comparison_dashboard(
            linear_results, nonlinear_results, maneuver['name'], maneuver['delta_func']
        )
        plt.savefig(f'model_comparison_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Анализ различий
        from src.analysis_tools import HandlingAnalysis
        analysis = HandlingAnalysis.analyze_handling_characteristics(
            linear_results, nonlinear_results, maneuver['name']
        )
        print_handling_analysis_report(analysis)
    
    # 4. Анализ управляемости
    print("\n4. 📊 АНАЛИЗ ХАРАКТЕРИСТИК УПРАВЛЯЕМОСТИ")
    
    # 4.1. Градиент недостаточной поворачиваемости
    K = HandlingAnalysis.calculate_understeer_gradient(params)
    print(f"   Градиент недостаточной поворачиваемости: K = {K:.6f} рад/с²/м")
    
    if K > 0:
        print("   ↳ Автомобиль обладает НЕДОСТАТОЧНОЙ ПОВОРАЧИВАЕМОСТЬЮ")
    elif K < 0:
        print("   ↳ Автомобиль обладает ИЗБЫТОЧНОЙ ПОВОРАЧИВАЕМОСТЬЮ")
    else:
        print("   ↳ Автомобиль обладает НЕЙТРАЛЬНОЙ ПОВОРАЧИВАЕМОСТЬЮ")
    
    # 4.2. Дашборд управляемости
    handling_fig = dashboard.create_handling_analysis_dashboard(params)
    plt.savefig('handling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Анализ влияния параметров шин
    print("\n5. 🎚️ АНАЛИЗ ВЛИЯНИЯ ПАРАМЕТРОВ ШИН")
    analyze_tire_parameter_sensitivity(params)
    
    # 6. Предельные режимы
    print("\n6. ⚠️  АНАЛИЗ ПРЕДЕЛЬНЫХ РЕЖИМОВ")
    analyze_limit_handling(params)
    
    print("\n" + "=" * 70)
    print("✅ НЕДЕЛЯ 3 ЗАВЕРШЕНА УСПЕШНО!")
    print("🎯 ЧТО МЫ СОЗДАЛИ:")
    print("   - Нелинейные модели шин (Фьяла, кусочно-линейная, магическая формула)")
    print("   - Систему сравнения линейной и нелинейной динамики")
    print("   - Инструменты анализа поворачиваемости и устойчивости")
    print("   - Продвинутые дашборды для комплексного анализа")
    print("=" * 70)

def analyze_tire_parameter_sensitivity(base_params):
    """Анализ чувствительности к параметрам шин"""
    print("\n   Анализ влияния соотношения жесткостей шин:")
    
    stiffness_ratios = [0.5, 1.0, 2.0]  # C_f / C_r
    results_linear = []
    results_nonlinear = []
    labels = []
    
    for ratio in stiffness_ratios:
        # Модифицируем параметры
        params_modified = VehicleParameters()
        params_modified.C_f = base_params.C_f * ratio
        params_modified.C_r = base_params.C_r
        
        labels.append(f'C_f/C_r = {ratio}')
        
        # Тестовый маневр
        maneuver = StandardManeuvers.step_steer_maneuver(amplitude=0.1)
        
        # Линейная модель
        linear_solution = simulate_maneuver(
            params=params_modified,
            delta_func=maneuver['delta_func'],
            V=base_params.V,
            t_span=maneuver['t_span']
        )
        results_linear.append(get_simulation_results(linear_solution))
        
        # Нелинейная модель
        nonlinear_simulator = create_nonlinear_simulator(
            params_modified, 
            front_stiffness=params_modified.C_f,
            rear_stiffness=params_modified.C_r,
            tire_model_type='fiala'
        )
        nonlinear_solution = nonlinear_simulator.simulate(
            delta_func=maneuver['delta_func'],
            V=base_params.V,
            t_span=maneuver['t_span']
        )
        results_nonlinear.append(get_simulation_results_nonlinear(nonlinear_solution))
    
    # Визуализация результатов
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (linear_res, nonlinear_res, label) in enumerate(zip(results_linear, results_nonlinear, labels)):
        # Траектории
        axes[0,0].plot(linear_res['X'], linear_res['Y'], linewidth=2, 
                      label=f'{label} (лин.)', alpha=0.7)
        axes[0,0].plot(nonlinear_res['X'], nonlinear_res['Y'], '--', linewidth=2,
                      label=f'{label} (нелин.)', alpha=0.7)
        
        # Углы скольжения
        axes[0,1].plot(linear_res['time'], np.degrees(linear_res['beta']), 
                      linewidth=2, label=f'{label} (лин.)', alpha=0.7)
        axes[0,1].plot(nonlinear_res['time'], np.degrees(nonlinear_res['beta']), '--',
                      linewidth=2, label=f'{label} (нелин.)', alpha=0.7)
    
    axes[0,0].set_xlabel('X, м')
    axes[0,0].set_ylabel('Y, м')
    axes[0,0].set_title('Влияние жесткости шин на траектории')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_xlabel('Время, с')
    axes[0,1].set_ylabel('β, °')
    axes[0,1].set_title('Влияние жесткости шин на углы скольжения')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Градиенты поворачиваемости
    gradients = []
    for ratio in stiffness_ratios:
        params_temp = VehicleParameters()
        params_temp.C_f = base_params.C_f * ratio
        K = HandlingAnalysis.calculate_understeer_gradient(params_temp)
        gradients.append(K)
    
    axes[1,0].bar(labels, gradients, color=['red', 'green', 'blue'], alpha=0.7)
    axes[1,0].set_ylabel('Градиент K, рад/с²/м')
    axes[1,0].set_title('Влияние жесткости на поворачиваемость')
    axes[1,0].grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for i, v in enumerate(gradients):
        axes[1,0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    axes[1,1].axis('off')  # Пустой subplot для симметрии
    
    plt.tight_layout()
    plt.savefig('tire_stiffness_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_limit_handling(params):
    """Анализ поведения в предельных режимах"""
    print("\n   Тестирование предельных режимов (большие углы руля):")
    
    large_angle_maneuvers = [
        StandardManeuvers.step_steer_maneuver(amplitude=0.2),  # ~11.5°
        StandardManeuvers.step_steer_maneuver(amplitude=0.3),  # ~17.2°
    ]
    
    for maneuver in large_angle_maneuvers:
        print(f"\n   Маневр: {maneuver['name']}")
        
        # Линейная модель
        linear_solution = simulate_maneuver(
            params=params,
            delta_func=maneuver['delta_func'],
            V=params.V,
            t_span=maneuver['t_span']
        )
        linear_results = get_simulation_results(linear_solution)
        
        # Нелинейная модель
        nonlinear_simulator = create_nonlinear_simulator(
            params, 
            front_stiffness=params.C_f,
            rear_stiffness=params.C_r,
            tire_model_type='fiala'
        )
        nonlinear_solution = nonlinear_simulator.simulate(
            delta_func=maneuver['delta_func'],
            V=params.V,
            t_span=maneuver['t_span']
        )
        nonlinear_results = get_simulation_results_nonlinear(nonlinear_solution)
        
        # Анализ различий
        from src.analysis_tools import HandlingAnalysis
        analysis = HandlingAnalysis.analyze_handling_characteristics(
            linear_results, nonlinear_results, maneuver['name']
        )
        
        print(f"   Разница в максимальном угле скольжения: {analysis['beta_difference_percent']:+.1f}%")
        print(f"   Разница в поперечном смещении: {analysis['y_difference_percent']:+.1f}%")
        
        if abs(analysis['beta_difference_percent']) > 20:
            print("   ⚠️  ЗНАЧИТЕЛЬНОЕ РАСХОЖДЕНИЕ - нелинейные эффекты существенны!")

if __name__ == "__main__":
    main()