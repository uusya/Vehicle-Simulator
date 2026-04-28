import pytest
import numpy as np

from src.web_interface.utils import calculate_performance_metrics

def test_performance_metrics_calculation():
    """Тест расчета метрик производительности"""
    # Создание тестовых данных
    results = {
        'time': np.array([0, 1, 2, 3]),
        'beta': np.array([0, 0.1, -0.05, 0.02]),
        'angular_velocity': np.array([0, 0.5, -0.2, 0.1]),
        'yaw_angle': np.array([0, 0.2, 0.5, 0.7]),
        'X': np.array([0, 10, 20, 30]),
        'Y': np.array([0, 1, 3, 4])
    }
    
    metrics = calculate_performance_metrics(results)
    
    assert 'max_beta' in metrics
    assert 'max_r' in metrics
    assert 'final_y' in metrics
    assert metrics['max_beta'] == pytest.approx(5.729, rel=1e-2)  # 0.1 rad in degrees

def test_default_presets_loading():
    """Тест загрузки предустановленных настроек"""
    from src.web_interface.utils import load_default_presets
    
    presets = load_default_presets()
    
    assert 'sports_car' in presets
    assert 'family_car' in presets
    assert 'race_car' in presets
    assert presets['sports_car']['mass'] == 1200

if __name__ == "__main__":
    pytest.main([__file__])
