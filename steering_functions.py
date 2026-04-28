import numpy as np

def step_steer(t, amplitude=0.1, start_time=1.0):
    """
    Ступенчатое рулевое воздействие (шаг рулем)
    
    Args:
        t (float): текущее время, с
        amplitude (float): амплитуда угла поворота, рад
        start_time (float): время начала воздействия, с
        
    Returns:
        float: угол поворота рулевого колеса, рад
    """
    return amplitude if t >= start_time else 0.0

def sine_steer(t, amplitude=0.1, frequency=0.5, start_time=1.0):
    """
    Синусоидальное рулевое воздействие
    
    Args:
        t (float): текущее время, с
        amplitude (float): амплитуда угла поворота, рад  
        frequency (float): частота, Гц
        start_time (float): время начала воздействия, с
        
    Returns:
        float: угол поворота рулевого колеса, рад
    """
    if t < start_time:
        return 0.0
    return amplitude * np.sin(2 * np.pi * frequency * (t - start_time))

def ramp_steer(t, slope=0.05, start_time=1.0, duration=2.0):
    """
    Линейно нарастающее рулевое воздействие
    
    Args:
        t (float): текущее время, с
        slope (float): скорость нарастания, рад/с
        start_time (float): время начала воздействия, с
        duration (float): длительность нарастания, с
        
    Returns:
        float: угол поворота рулевого колеса, рад
    """
    if t < start_time:
        return 0.0
    elif t < start_time + duration:
        return slope * (t - start_time)
    else:
        return slope * duration

def test_steering_functions():
    """Тестирование функций рулевого управления"""
    print("🔧 Тестирование функций рулевого управления...")
    
    # Тест ступенчатой функции
    print("  Тестирование step_steer...")
    assert step_steer(0.5, 0.1, 1.0) == 0.0, "Ошибка ступенчатой функции (до начала)"
    assert step_steer(1.5, 0.1, 1.0) == 0.1, "Ошибка ступенчатой функции (после начала)"
    
    # Тест синусоидальной функции
    print("  Тестирование sine_steer...")
    assert sine_steer(0.5, 0.1, 0.5, 1.0) == 0.0, "Ошибка синусоидальной функции (до начала)"
    
    # Проверяем в момент пика (sin = 1)
    peak_time = 1.0 + 0.25 / 0.5  # t = start_time + (π/2) / (2πf) = start_time + 1/(4f)
    result_at_peak = sine_steer(peak_time, 0.1, 0.5, 1.0)
    assert abs(result_at_peak - 0.1) < 1e-10, f"Пиковое значение: {result_at_peak} != 0.1"
    
    # Проверяем в момент нуля (sin = 0)
    zero_time = 1.0 + 0.5 / 0.5  # t = start_time + (π) / (2πf) = start_time + 1/(2f)
    result_at_zero = sine_steer(zero_time, 0.1, 0.5, 1.0)
    assert abs(result_at_zero - 0.0) < 1e-10, f"Нулевое значение: {result_at_zero} != 0.0"
    
    # Тест линейной функции
    print("  Тестирование ramp_steer...")
    assert ramp_steer(0.5, 0.05, 1.0, 2.0) == 0.0, "Ошибка линейной функции (до начала)"
    assert ramp_steer(1.5, 0.05, 1.0, 2.0) == 0.025, "Ошибка линейной функции (во время нарастания)"
    assert ramp_steer(3.5, 0.05, 1.0, 2.0) == 0.1, "Ошибка линейной функции (после нарастания)"
    
    print("✅ Все функции рулевого управления работают корректно")

if __name__ == "__main__":
    test_steering_functions()