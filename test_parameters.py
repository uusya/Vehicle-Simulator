import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.steering_functions import sine_steer, step_steer
from src.vehicle_parameters import VehicleParameters

def test_parameters_initialization():
    """Тест инициализации параметров"""
    print("🧪 Тест 1: Инициализация параметров")
    params = VehicleParameters()
    
    # Проверка вычисляемых параметров
    assert params.L == params.a + params.b, "Колесная база вычислена неверно"
    
    # Проверка типов
    assert isinstance(params.m, float), "Масса должна быть float"
    assert isinstance(params.C_f, float), "Жесткость шин должна быть float"
    
    print("   ✓ Параметры инициализированы корректно")

def test_parameters_validation():
    """Тест валидации параметров"""
    print("🧪 Тест 2: Валидация параметров")
    params = VehicleParameters()
    
    try:
        result = params.validate_parameters()
        assert result == True, "Валидация должна возвращать True"
        print("   ✓ Валидация параметров прошла успешно")
    except Exception as e:
        print(f"   ✗ Ошибка валидации: {e}")
        raise

def test_steering_functions():
    """Тест функций рулевого управления"""
    print("🧪 Тест 3: Функции рулевого управления")
    
    # Тест ступенчатой функции
    assert step_steer(0.5, 0.1, 1.0) == 0.0, "Ступенчатая функция (до начала)"
    assert step_steer(1.5, 0.1, 1.0) == 0.1, "Ступенчатая функция (после начала)"
    
    # Тест синусоидальной функции с явным указанием start_time
    assert sine_steer(0.5, 0.1, 0.5, 1.0) == 0.0, "Синусоидальная функция (до начала)"
    
    # Проверяем корректность вычислений в момент времени, когда sin = 1
    # sin(2π * f * (t - start_time)) = 1, когда аргумент = π/2 + 2πk
    # Для f=0.5 Гц: 2π * 0.5 * (t - 1.0) = π/2 => (t - 1.0) = 0.5 => t = 1.5
    result_at_peak = sine_steer(1.5, 0.1, 0.5, 1.0)
    assert abs(result_at_peak - 0.1) < 1e-10, f"Синусоидальная функция вернула {result_at_peak}, ожидалось 0.1"
    
    # Проверяем, что функция возвращает число (не NaN) для произвольного времени
    result = sine_steer(1.25, 0.1, 0.5, 1.0)
    assert isinstance(result, float), "Синусоидальная функция должна возвращать float"
    
    # Правильное ожидаемое значение для t=1.25:
    # 2π * 0.5 * (1.25 - 1.0) = 2π * 0.5 * 0.25 = π * 0.25 ≈ 0.7854 радиан
    # sin(0.7854) ≈ 0.7071
    # 0.1 * 0.7071 ≈ 0.07071
    expected_value = 0.1 * np.sin(2 * np.pi * 0.5 * (1.25 - 1.0))
    assert abs(result - expected_value) < 1e-10, f"Синусоидальная функция вернула {result}, ожидалось {expected_value}"
    
    print("   ✓ Функции рулевого управления работают корректно")

def run_all_tests():
    """Запуск всех тестов"""
    print("🚀 ЗАПУСК ТЕСТОВ НЕДЕЛИ 1")
    print("=" * 40)
    
    test_parameters_initialization()
    test_parameters_validation()
    test_steering_functions()
    
    print("=" * 40)
    print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")

if __name__ == "__main__":
    run_all_tests()
