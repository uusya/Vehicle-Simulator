# src/analysis_tools.py
import numpy as np
import matplotlib.pyplot as plt

def calculate_characteristic_equation(params, V):
    """
    Вычисление характеристического уравнения линейной системы
    
    Args:
        params (VehicleParameters): параметры автомобиля
        V (float): скорость, м/с
        
    Returns:
        tuple: коэффициенты характеристического уравнения и собственные значения
    """
    m, J_z, a, b, C_f, C_r = params.m, params.J_z, params.a, params.b, params.C_f, params.C_r
    
    # Элементы матрицы состояния A для [beta, r]
    A11 = -(C_f + C_r) / (m * V)
    A12 = -1 - (C_f * a - C_r * b) / (m * V**2)
    A21 = -(C_f * a - C_r * b) / J_z
    A22 = -(C_f * a**2 + C_r * b**2) / (J_z * V)
    
    A = np.array([[A11, A12], 
                  [A21, A22]])
    
    # Собственные значения
    eigenvalues = np.linalg.eigvals(A)
    
    # Коэффициенты характеристического уравнения: λ² + a1*λ + a0 = 0
    a1 = -np.trace(A)
    a0 = np.linalg.det(A)
    
    return (a1, a0), eigenvalues, A

def check_stability(eigenvalues):
    """
    Проверка устойчивости системы по собственным значениям
    
    Args:
        eigenvalues (array): собственные значения матрицы A
        
    Returns:
        tuple: (is_stable, stability_type)
    """
    real_parts = np.real(eigenvalues)
    
    is_stable = all(real_part < 0 for real_part in real_parts)
    
    # Определение типа устойчивости
    if is_stable:
        if all(np.imag(eigenvalues) == 0):
            stability_type = "апериодическая устойчивость"
        else:
            stability_type = "колебательная устойчивость"
    else:
        if any(real_part > 0):
            stability_type = "неустойчивая"
        else:
            stability_type = "на границе устойчивости"
    
    return is_stable, stability_type

def print_stability_analysis(params, V):
    """
    Печать анализа устойчивости
    """
    coefficients, eigenvalues, A = calculate_characteristic_equation(params, V)
    is_stable, stability_type = check_stability(eigenvalues)
    
    print("=" * 60)
    print("АНАЛИЗ УСТОЙЧИВОСТИ ЛИНЕЙНОЙ МОДЕЛИ")
    print("=" * 60)
    print(f"Скорость: {V} м/с ({V * 3.6:.1f} км/ч)")
    print(f"Матрица состояния A:")
    for row in A:
        print(f"  [{row[0]:8.4f}  {row[1]:8.4f}]")
    print(f"\nСобственные значения:")
    for i, eig in enumerate(eigenvalues):
        print(f"  λ{i+1} = {eig.real:7.4f} + {eig.imag:7.4f}j")
    print(f"\nХарактеристическое уравнение: λ² + {coefficients[0]:.4f}λ + {coefficients[1]:.4f} = 0")
    print(f"\nРезультат: система {stability_type.upper()}")
    print("=" * 60)
    
    return is_stable, eigenvalues

class HandlingAnalysis:
    """Класс для анализа характеристик управляемости"""
    
    @staticmethod
    def calculate_understeer_gradient(vehicle_params):
        """
        Расчет градиента недостаточной поворачиваемости
        """
        m = vehicle_params.m
        L = vehicle_params.a + vehicle_params.b
        a = vehicle_params.a
        b = vehicle_params.b
        C_f = vehicle_params.C_f
        C_r = vehicle_params.C_r
        
        K = (m / L) * (b / C_f - a / C_r)
        return K
    
    @staticmethod
    def analyze_handling_characteristics(linear_results, nonlinear_results, maneuver_name):
        """
        Сравнительный анализ характеристик управляемости
        """
        analysis = {}
        
        # Анализ максимальных значений
        analysis['linear_max_beta'] = np.max(np.abs(np.degrees(linear_results['beta'])))
        analysis['nonlinear_max_beta'] = np.max(np.abs(np.degrees(nonlinear_results['beta'])))
        
        analysis['linear_max_r'] = np.max(np.abs(np.degrees(linear_results['angular_velocity'])))
        analysis['nonlinear_max_r'] = np.max(np.abs(np.degrees(nonlinear_results['angular_velocity'])))
        
        analysis['linear_final_y'] = linear_results['Y'][-1]
        analysis['nonlinear_final_y'] = nonlinear_results['Y'][-1]
        
        # Расчет разницы в процентах
        analysis['beta_difference_percent'] = (
            (analysis['nonlinear_max_beta'] - analysis['linear_max_beta']) / 
            analysis['linear_max_beta'] * 100
        )
        
        analysis['r_difference_percent'] = (
            (analysis['nonlinear_max_r'] - analysis['linear_max_r']) / 
            analysis['linear_max_r'] * 100
        )
        
        if abs(analysis['linear_final_y']) > 0.001:
            analysis['y_difference_percent'] = (
                (analysis['nonlinear_final_y'] - analysis['linear_final_y']) / 
                abs(analysis['linear_final_y']) * 100
            )
        else:
            analysis['y_difference_percent'] = 0
        
        analysis['maneuver'] = maneuver_name
        
        return analysis

def print_handling_analysis_report(analysis_dict):
    """Печать отчета по анализу управляемости"""
    print(f"\n📊 ОТЧЕТ ПО АНАЛИЗУ УПРАВЛЯЕМОСТИ: {analysis_dict['maneuver']}")
    print("=" * 60)
    print("МАКСИМАЛЬНЫЕ ЗНАЧЕНИЯ:")
    print(f"  Угол скольжения: {analysis_dict['linear_max_beta']:.2f}° (лин.) → {analysis_dict['nonlinear_max_beta']:.2f}° (нелин.)")
    print(f"  Разница: {analysis_dict['beta_difference_percent']:+.1f}%")
    print(f"  Угловая скорость: {analysis_dict['linear_max_r']:.2f}°/с (лин.) → {analysis_dict['nonlinear_max_r']:.2f}°/с (нелин.)")
    print(f"  Разница: {analysis_dict['r_difference_percent']:+.1f}%")
    print(f"  Поперечное смещение: {analysis_dict['linear_final_y']:.2f}м (лин.) → {analysis_dict['nonlinear_final_y']:.2f}м (нелин.)")
    print(f"  Разница: {analysis_dict['y_difference_percent']:+.1f}%")
    print("=" * 60)

class StabilityAnalysis:
    """Класс для анализа устойчивости"""
    
    @staticmethod
    def calculate_stability_derivatives(vehicle_params, V):
        """Расчет стабилизирующих производных"""
        # Заглушка - можно реализовать позже
        return {
            'Y_beta': 0,
            'N_beta': 0,
            'Y_r': 0,
            'N_r': 0
        }
    
    @staticmethod
    def analyze_stability_modes(vehicle_params, speed_range=None):
        """Анализ мод устойчивости"""
        # Заглушка - можно реализовать позже
        return {
            'speed': [],
            'eigenvalues_real': [],
            'eigenvalues_imag': [],
            'stability_modes': []
        }