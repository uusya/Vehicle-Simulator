import numpy as np
from scipy.integrate import solve_ivp

def linear_bicycle_model(t, state, params, delta_func, V):
    """
    Линейная модель 'велосипед' для поперечной динамики автомобиля
    
    Args:
        t (float): текущее время, с
        state (list): вектор состояния [beta, r, psi, X, Y]
        params (VehicleParameters): параметры автомобиля
        delta_func (function): функция угла поворота руля от времени
        V (float): продольная скорость, м/с
        
    Returns:
        list: производные вектора состояния [dbeta_dt, dr_dt, dpsi_dt, dX_dt, dY_dt]
    """
    # Распаковка вектора состояния
    beta, r, psi, X, Y = state
    
    # Вычисление текущего угла поворота руля
    delta = delta_func(t)
    
    # Вычисление углов увода
    alpha_f = delta - (beta + params.a * r / V)  # передние колеса
    alpha_r = -(beta - params.b * r / V)         # задние колеса
    
    # Вычисление боковых сил (линейная модель)
    F_yf = params.C_f * alpha_f
    F_yr = params.C_r * alpha_r
    
    # Уравнения движения
    dbeta_dt = (F_yf + F_yr) / (params.m * V) - r
    dr_dt = (F_yf * params.a - F_yr * params.b) / params.J_z
    dpsi_dt = r
    dX_dt = V * np.cos(psi - beta)
    dY_dt = V * np.sin(psi - beta)
    
    return [dbeta_dt, dr_dt, dpsi_dt, dX_dt, dY_dt]

def simulate_maneuver(params, delta_func, V, t_span, initial_state=None, method='RK45'):
    """
    Запуск симуляции маневра
    
    Args:
        params (VehicleParameters): параметры автомобиля
        delta_func (function): функция рулевого управления
        V (float): скорость, м/с
        t_span (tuple): интервал времени (t_start, t_end)
        initial_state (list): начальное состояние
        method (str): метод интегрирования
        
    Returns:
        object: объект решения от solve_ivp
    """
    if initial_state is None:
        # Начальное состояние: автомобиль движется прямо
        initial_state = [0.0, 0.0, 0.0, 0.0, 0.0]  # [beta, r, psi, X, Y]
    
    # Проверка параметров
    params.validate_parameters()
    
    # Решение системы дифференциальных уравнений
    solution = solve_ivp(
        fun=lambda t, y: linear_bicycle_model(t, y, params, delta_func, V),
        t_span=t_span,
        y0=initial_state,
        method=method,
        dense_output=True,
        rtol=1e-6,
        atol=1e-9
    )
    
    if not solution.success:
        print(f"⚠️ Предупреждение: решение не сошлось. Сообщение: {solution.message}")
    
    return solution

def get_simulation_results(solution):
    """
    Извлекает и возвращает результаты симуляции в удобном формате
    
    Args:
        solution: объект решения от solve_ivp
        
    Returns:
        dict: словарь с результатами симуляции
    """
    t = solution.t
    beta = solution.y[0, :]    # угол бокового скольжения
    r = solution.y[1, :]       # угловая скорость
    psi = solution.y[2, :]     # курсовой угол
    X = solution.y[3, :]       # координата X
    Y = solution.y[4, :]       # координата Y
    
    return {
        'time': t,
        'beta': beta,
        'angular_velocity': r,
        'yaw_angle': psi,
        'X': X,
        'Y': Y,
        'success': solution.success
    }