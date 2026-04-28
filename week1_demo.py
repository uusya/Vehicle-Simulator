import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.vehicle_parameters import VehicleParameters
from src.steering_functions import step_steer
from src.linear_bicycle_model import simulate_maneuver, get_simulation_results
from src.model_utils import print_stability_analysis

def main():
    print("🚗 ДЕМОНСТРАЦИЯ ЛИНЕЙНОЙ МОДЕЛИ 'ВЕЛОСИПЕД' - НЕДЕЛЯ 1")
    print("=" * 60)
    
    # 1. Создание и проверка параметров
    print("\n1. 📋 СОЗДАНИЕ ПАРАМЕТРОВ АВТОМОБИЛЯ")
    params = VehicleParameters()
    params.validate_parameters()
    print(params.get_summary())
    
    # 2. Анализ устойчивости
    print("\n2. 📊 АНАЛИЗ УСТОЙЧИВОСТИ СИСТЕМЫ")
    V = 20.0  # м/с
    is_stable, eigenvalues = print_stability_analysis(params, V)
    
    # 3. Симуляция шага рулем
    print("\n3. 🎮 СИМУЛЯЦИЯ МАНЕВРА 'ШАГ РУЛЕМ'")
    delta_amplitude = 0.1  # ~5.7 градусов
    
    print(f"   Параметры симуляции:")
    print(f"   - Угол поворота руля: {np.degrees(delta_amplitude):.1f}°")
    print(f"   - Скорость: {V} м/с ({V * 3.6:.1f} км/ч)")
    print(f"   - Время симуляции: 10 секунд")
    
    solution = simulate_maneuver(
        params=params,
        delta_func=lambda t: step_steer(t, delta_amplitude, 1.0),
        V=V,
        t_span=(0, 10)
    )
    
    # 4. Извлечение результатов
    results = get_simulation_results(solution)
    
    # 5. Вывод результатов
    print(f"\n4. 📈 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
    print(f"   - Успех решения: {'ДА' if solution.success else 'НЕТ'}")
    print(f"   - Время решения: {results['time'][-1]:.1f} с")
    print(f"   - Количество точек: {len(results['time'])}")
    print(f"   - Финальные координаты: X={results['X'][-1]:.1f} м, Y={results['Y'][-1]:.1f} м")
    print(f"   - Финальный угол скольжения: {np.degrees(results['beta'][-1]):.2f}°")
    print(f"   - Финальная угловая скорость: {np.degrees(results['angular_velocity'][-1]):.2f} °/с")
    
    print("\n" + "=" * 60)
    print("✅ НЕДЕЛЯ 1 ЗАВЕРШЕНА УСПЕШНО!")
    print("🎯 ЧТО МЫ ИМЕЕМ:")
    print("   - Рабочую математическую модель")
    print("   - Инструменты анализа устойчивости") 
    print("   - Возможность запускать симуляции")
    print("   - Готовую структуру для дальнейшего развития")
    print("=" * 60)

if __name__ == "__main__":
    main()