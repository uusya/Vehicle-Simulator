import numpy as np

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