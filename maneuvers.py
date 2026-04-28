import numpy as np

from .steering_functions import ramp_steer, sine_steer, step_steer

class StandardManeuvers:
    """Класс для генерации стандартных тестовых маневров"""
    
    @staticmethod
    def step_steer_maneuver(amplitude=0.1, start_time=1.0, duration=10.0):
        """
        Маневр 'Шаг рулем' - резкий поворот с постоянным углом
        """
        return {
            'name': f'Шаг рулем ({np.degrees(amplitude):.1f}°)',
            'delta_func': lambda t: step_steer(t, amplitude, start_time),
            't_span': (0, duration),
            'description': f'Резкий поворот руля на {np.degrees(amplitude):.1f}° в момент {start_time}с'
        }
    
    @staticmethod
    def lane_change_maneuver(amplitude=0.05, frequency=0.5, duration=10.0):
        """
        Маневр 'Переставка' - синусоидальное руление
        """
        return {
            'name': f'Переставка ({np.degrees(amplitude):.1f}°, {frequency}Гц)',
            'delta_func': lambda t: sine_steer(t, amplitude, frequency, 1.0),
            't_span': (0, duration),
            'description': f'Синусоидальное руление с амплитудой {np.degrees(amplitude):.1f}° и частотой {frequency}Гц'
        }
    
    @staticmethod
    def constant_radius_maneuver(radius=100.0, duration=10.0):
        """
        Маневр 'Постоянный радиус' - движение по кругу
        """
        def constant_radius_delta(t):
            if t < 1.0:
                return 0.0
            else:
                L = 3.0  # колесная база
                return L / radius
        
        return {
            'name': f'Постоянный радиус (R={radius}м)',
            'delta_func': constant_radius_delta,
            't_span': (0, duration),
            'description': f'Движение по кругу радиусом {radius}м'
        }
    
    @staticmethod
    def double_lane_change_maneuver(amplitude=0.08, frequency=0.3, duration=12.0):
        """
        Маневр 'Двойная переставка' - более сложный маневр уклонения
        """
        def double_lane_change(t):
            if t < 1.0:
                return 0.0
            elif t < 4.0:
                return amplitude * np.sin(2 * np.pi * frequency * (t - 1.0))
            elif t < 7.0:
                return -amplitude * np.sin(2 * np.pi * frequency * (t - 4.0))
            else:
                return 0.0
        
        return {
            'name': f'Двойная переставка',
            'delta_func': double_lane_change,
            't_span': (0, duration),
            'description': 'Сложный маневр уклонения с возвратом на полосу'
        }
    
    @staticmethod
    def get_all_maneuvers():
        """Возвращает список всех стандартных маневров"""
        return [
            StandardManeuvers.step_steer_maneuver(amplitude=0.1),
            StandardManeuvers.step_steer_maneuver(amplitude=0.05),
            StandardManeuvers.lane_change_maneuver(amplitude=0.05, frequency=0.5),
            StandardManeuvers.lane_change_maneuver(amplitude=0.08, frequency=0.3),
            StandardManeuvers.constant_radius_maneuver(radius=100.0),
            StandardManeuvers.constant_radius_maneuver(radius=50.0),
            StandardManeuvers.double_lane_change_maneuver()
        ]

def analyze_maneuver_performance(results, maneuver_name):
    """
    Анализ характеристик выполненного маневра
    """
    performance = {
        'maneuver': maneuver_name,
        'completion_time': results['time'][-1],
        'final_displacement': results['Y'][-1],
        'max_sideslip_angle': np.max(np.abs(np.degrees(results['beta']))),
        'max_angular_velocity': np.max(np.abs(np.degrees(results['angular_velocity']))),
        'overshoot': None,
        'settling_time': None
    }
    
    # Расчет перерегулирования (для шага рулем)
    if 'Шаг рулем' in maneuver_name:
        final_value = results['Y'][-1]
        max_value = np.max(results['Y'])
        if final_value != 0:
            performance['overshoot'] = (max_value - final_value) / final_value * 100
    
    return performance

def print_maneuver_performance(performance_dict):
    """Печать результатов анализа маневра"""
    print(f"\n📊 АНАЛИЗ МАНЕВРА: {performance_dict['maneuver']}")
    print("=" * 50)
    print(f"Время выполнения: {performance_dict['completion_time']:.1f} с")
    print(f"Конечное смещение: {performance_dict['final_displacement']:.2f} м")
    print(f"Макс. угол скольжения: {performance_dict['max_sideslip_angle']:.2f}°")
    print(f"Макс. угловая скорость: {performance_dict['max_angular_velocity']:.2f} °/с")
    
    if performance_dict['overshoot'] is not None:
        print(f"Перерегулирование: {performance_dict['overshoot']:.1f}%")
    
    print("=" * 50)
