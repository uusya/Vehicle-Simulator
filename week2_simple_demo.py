# examples/week2_simple_demo.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.vehicle_parameters import VehicleParameters
from src.linear_bicycle_model import simulate_maneuver, get_simulation_results
from src.maneuvers import StandardManeuvers
from src.visualization.plotter import ResultPlotter

def main():
    print("🚗 ПРОСТАЯ ДЕМОНСТРАЦИЯ НЕДЕЛИ 2")
    print("=" * 50)
    
    # 1. Создание параметров автомобиля
    params = VehicleParameters()
    print("Параметры автомобиля созданы")
    
    # 2. Запуск маневра "Шаг рулем"
    maneuver = StandardManeuvers.step_steer_maneuver(amplitude=0.1)
    print(f"Запуск маневра: {maneuver['name']}")
    
    solution = simulate_maneuver(
        params=params,
        delta_func=maneuver['delta_func'],
        V=params.V,
        t_span=maneuver['t_span']
    )
    
    results = get_simulation_results(solution)
    print("Симуляция завершена успешно!")
    
    # 3. Визуализация результатов
    plotter = ResultPlotter()
    fig = plotter.plot_comprehensive_results(
        results, 
        maneuver['delta_func'],
        title=f"Маневр: {maneuver['name']}"
    )
    
    plt.show()
    
    # 4. Анализ производительности
    from src.maneuvers import analyze_maneuver_performance, print_maneuver_performance
    performance = analyze_maneuver_performance(results, maneuver['name'])
    print_maneuver_performance(performance)
    
    print("✅ Демонстрация завершена успешно!")

if __name__ == "__main__":
    main()