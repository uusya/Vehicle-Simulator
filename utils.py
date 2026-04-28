# src/web_interface/utils.py
import numpy as np
import streamlit as st
from typing import Dict, Any

def calculate_performance_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Расчет метрик производительности маневра"""
    metrics = {}
    
    # Максимальные значения
    metrics['max_beta'] = float(np.max(np.abs(np.degrees(results['beta']))))
    metrics['max_r'] = float(np.max(np.abs(np.degrees(results['angular_velocity']))))
    metrics['max_ay'] = float(np.max(np.abs(results.get('lateral_acceleration', 0))))
    
    # Конечные значения
    metrics['final_y'] = float(results['Y'][-1])
    metrics['final_psi'] = float(np.degrees(results['yaw_angle'][-1]))
    
    # Время установления (для шага рулем)
    if len(results['beta']) > 10:
        final_beta = results['beta'][-1]
        settling_threshold = 0.02  # 2% от установившегося значения
        for i, beta in enumerate(results['beta']):
            if abs(beta - final_beta) < settling_threshold:
                metrics['settling_time'] = float(results['time'][i])
                break
        else:
            metrics['settling_time'] = float(results['time'][-1])
    
    return metrics

def save_simulation_results(results: Dict[str, np.ndarray], filename: str):
    """Сохранение результатов симуляции в файл"""
    import pandas as pd
    
    data = {
        'time': results['time'],
        'beta_deg': np.degrees(results['beta']),
        'r_deg_s': np.degrees(results['angular_velocity']),
        'psi_deg': np.degrees(results['yaw_angle']),
        'X': results['X'],
        'Y': results['Y']
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

def load_default_presets() -> Dict[str, Any]:
    """Загрузка预设ных настроек"""
    return {
        'sports_car': {
            'name': 'Спортивный автомобиль',
            'mass': 1200,
            'cf': 100000,
            'cr': 120000, 
            'cog_ratio': 0.48,
            'speed': 25
        },
        'family_car': {
            'name': 'Семейный автомобиль',
            'mass': 1500,
            'cf': 70000,
            'cr': 80000,
            'cog_ratio': 0.52, 
            'speed': 18
        },
        'race_car': {
            'name': 'Гоночный автомобиль',
            'mass': 800,
            'cf': 150000,
            'cr': 140000,
            'cog_ratio': 0.45,
            'speed': 30
        }
    }