# tests/test_week5.py
import pytest
import numpy as np
import os

def test_3d_animator():
    """Тест 3D аниматора"""
    from src.visualization.three_d_animator import ThreeDAnimator
    
    animator = ThreeDAnimator()
    
    # Тестовые данные
    results = {
        'X': np.array([0, 1, 2, 3, 4]),
        'Y': np.array([0, 0.1, 0.4, 0.9, 1.6]),
        'time': np.array([0, 1, 2, 3, 4]),
        'beta': np.array([0, 0.01, 0.02, 0.01, 0]),
        'angular_velocity': np.array([0, 0.1, 0.05, -0.05, -0.1]),
        'yaw_angle': np.array([0, 0.1, 0.3, 0.6, 1.0])
    }
    
    # Тест создания 3D траектории
    fig = animator.create_3d_trajectory(results)
    assert fig is not None
    
    # Тест энергетического анализа
    class MockParams:
        m = 1200
        J_z = 1500
        speed = 20
    
    energy_fig, metrics = animator.create_energy_analysis_plot(results, MockParams())
    assert energy_fig is not None
    assert 'max_rotational_energy' in metrics

def test_data_exporter():
    """Тест экспорта данных"""
    from src.export.data_exporter import DataExporter
    import tempfile
    import shutil
    
    # Создание временной директории
    temp_dir = tempfile.mkdtemp()
    
    try:
        exporter = DataExporter(export_dir=temp_dir)
        
        # Тестовые данные
        results = {
            'time': np.array([0, 1, 2]),
            'beta': np.array([0, 0.1, 0.2]),
            'angular_velocity': np.array([0, 0.5, 0.3]),
            'yaw_angle': np.array([0, 0.2, 0.5]),
            'X': np.array([0, 10, 20]),
            'Y': np.array([0, 1, 3])
        }
        
        params = {
            'mass': 1200,
            'speed': 20,
            'cog_ratio': 0.47,
            'cf': 80000,
            'cr': 100000,
            'tire_model': 'linear'
        }
        
        # Тест экспорта в CSV
        csv_file = exporter.export_to_csv(results, params, "test_maneuver")
        assert os.path.exists(csv_file)
        
        # Тест экспорта в Excel
        excel_file = exporter.export_to_excel(results, params, "test_maneuver")
        assert os.path.exists(excel_file)
        
    finally:
        # Очистка
        shutil.rmtree(temp_dir)

def test_stability_analysis():
    """Тест анализа устойчивости"""
    from src.analysis.advanced_stability import AdvancedStabilityAnalysis
    from src.vehicle_parameters import VehicleParameters
    
    analyzer = AdvancedStabilityAnalysis()
    params = VehicleParameters()
    
    # Тест анализа мод устойчивости
    stability_data = analyzer.analyze_stability_modes(params, (10, 30), 10)
    
    assert 'speeds' in stability_data
    assert 'eigenvalues_real' in stability_data
    assert len(stability_data['speeds']) == 10
    
    # Тест индикаторов управляемости
    indicators = analyzer.calculate_handling_indicators(params, 20)
    
    assert 'understeer_gradient' in indicators
    assert 'critical_speed' in indicators
    assert 'stability_margin' in indicators

if __name__ == "__main__":
    pytest.main([__file__])
