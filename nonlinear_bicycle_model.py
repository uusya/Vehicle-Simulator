import numpy as np
from scipy.integrate import solve_ivp

from .tire_models import (
    LinearTireModel,
    PacejkaTireModel,
    PiecewiseLinearTireModel,
    SimplifiedFialaTireModel,
)

class NonlinearBicycleModel:
    """
    Нелинейная модель 'велосипед' с различными моделями шин
    """
    
    def __init__(self, vehicle_params, front_tire_model, rear_tire_model):
        self.params = vehicle_params
        self.front_tire = front_tire_model
        self.rear_tire = rear_tire_model
        
    def equations_of_motion(self, t, state, delta_func, V):
        """
        Система дифференциальных уравнений нелинейной модели
        """
        beta, r, psi, X, Y = state
        delta = delta_func(t)
        
        # Вычисление углов увода
        alpha_f = delta - (beta + self.params.a * r / V)
        alpha_r = -(beta - self.params.b * r / V)
        
        # Вычисление боковых сил с использованием нелинейных моделей
        F_yf = self.front_tire.lateral_force(alpha_f)
        F_yr = self.rear_tire.lateral_force(alpha_r)
        
        # Уравнения движения
        dbeta_dt = (F_yf + F_yr) / (self.params.m * V) - r
        dr_dt = (F_yf * self.params.a - F_yr * self.params.b) / self.params.J_z
        dpsi_dt = r
        dX_dt = V * np.cos(psi - beta)
        dY_dt = V * np.sin(psi - beta)
        
        return [dbeta_dt, dr_dt, dpsi_dt, dX_dt, dY_dt]
    
    def simulate(self, delta_func, V, t_span, initial_state=None, method='RK45'):
        """
        Запуск симуляции нелинейной модели
        """
        if initial_state is None:
            initial_state = [0.0, 0.0, 0.0, 0.0, 0.0]  # [beta, r, psi, X, Y]
        
        solution = solve_ivp(
            fun=lambda t, y: self.equations_of_motion(t, y, delta_func, V),
            t_span=t_span,
            y0=initial_state,
            method=method,
            dense_output=True,
            rtol=1e-6,
            atol=1e-9
        )
        
        return solution

def create_nonlinear_simulator(vehicle_params, front_stiffness, rear_stiffness, 
                              tire_model_type='piecewise'):
    """
    Фабрика для создания нелинейного симулятора
    """
    if tire_model_type == 'linear':
        front_tire = LinearTireModel(front_stiffness)
        rear_tire = LinearTireModel(rear_stiffness)
    elif tire_model_type == 'fiala':
        front_tire = SimplifiedFialaTireModel(front_stiffness)
        rear_tire = SimplifiedFialaTireModel(rear_stiffness)
    elif tire_model_type == 'simple_fiala':
        front_tire = SimplifiedFialaTireModel(front_stiffness)
        rear_tire = SimplifiedFialaTireModel(rear_stiffness)
    elif tire_model_type == 'piecewise':
        front_tire = PiecewiseLinearTireModel(front_stiffness)
        rear_tire = PiecewiseLinearTireModel(rear_stiffness)
    elif tire_model_type == 'pacejka':
        front_tire = PacejkaTireModel()
        rear_tire = PacejkaTireModel()
    else:
        raise ValueError(f"Неизвестный тип модели шин: {tire_model_type}")
    
    return NonlinearBicycleModel(vehicle_params, front_tire, rear_tire)

def get_simulation_results_nonlinear(solution):
    """
    Извлекает результаты симуляции в удобном формате
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
