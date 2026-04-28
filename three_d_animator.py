# src/visualization/three_d_animator.py
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

class ThreeDAnimator:
    """Класс для создания 3D анимации и визуализации"""
    
    def __init__(self, car_length=4.5, car_width=1.8, car_height=1.5):
        self.car_length = car_length
        self.car_width = car_width
        self.car_height = car_height
        
    def create_3d_trajectory(self, results, title="3D Траектория движения"):
        """
        Создание 3D траектории движения автомобиля
        """
        fig = go.Figure()
        
        # Траектория
        fig.add_trace(go.Scatter3d(
            x=results['X'],
            y=results['Y'], 
            z=np.zeros_like(results['X']),  # Пока движение в 2D
            mode='lines',
            name='Траектория',
            line=dict(color='blue', width=4),
            opacity=0.7
        ))
        
        # Начальная точка
        fig.add_trace(go.Scatter3d(
            x=[results['X'][0]],
            y=[results['Y'][0]],
            z=[0],
            mode='markers',
            name='Старт',
            marker=dict(size=8, color='green')
        ))
        
        # Конечная точка
        fig.add_trace(go.Scatter3d(
            x=[results['X'][-1]],
            y=[results['Y'][-1]], 
            z=[0],
            mode='markers',
            name='Финиш',
            marker=dict(size=8, color='red')
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X, м',
                yaxis_title='Y, м',
                zaxis_title='Z, м',
                aspectmode='data'
            ),
            height=600
        )
        
        return fig
    
    def create_animated_3d_car(self, results, frames_count=50):
        """
        Создание анимированного 3D автомобиля
        """
        # Упрощенная версия - точечная анимация
        times = np.linspace(0, len(results['X'])-1, frames_count, dtype=int)
        
        fig = go.Figure()
        
        # Полная траектория
        fig.add_trace(go.Scatter3d(
            x=results['X'],
            y=results['Y'],
            z=np.zeros_like(results['X']),
            mode='lines',
            name='Траектория',
            line=dict(color='gray', width=2),
            opacity=0.3
        ))
        
        # Анимированный автомобиль
        fig.add_trace(go.Scatter3d(
            x=[results['X'][0]],
            y=[results['Y'][0]], 
            z=[0],
            mode='markers',
            name='Автомобиль',
            marker=dict(size=15, color='red')
        ))
        
        # Создание кадров анимации
        frames = []
        for i, t in enumerate(times):
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=[results['X'][t]],
                    y=[results['Y'][t]],
                    z=[0]
                )],
                name=f'frame_{i}'
            ))
        
        fig.frames = frames
        
        # Настройки анимации
        fig.update_layout(
            title="Анимированное движение автомобиля",
            scene=dict(
                xaxis_title='X, м',
                yaxis_title='Y, м', 
                zaxis_title='Z, м'
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 100, "redraw": True},
                                       "fromcurrent": True}])]
            )],
            height=600
        )
        
        return fig
    
    def create_energy_analysis_plot(self, results, params):
        """
        Анализ энергетических характеристик
        """
        time = results['time']
        beta = results['beta']
        r = results['angular_velocity']
        V = getattr(params, 'speed', getattr(params, 'V', None))
        if V is None and isinstance(params, dict):
            V = params.get('speed', params.get('V'))
        if V is None:
            raise ValueError("Parameter object must provide speed via 'speed' or 'V'.")

        mass = getattr(params, 'm', None)
        inertia = getattr(params, 'J_z', None)
        if isinstance(params, dict):
            mass = params.get('m', mass)
            inertia = params.get('J_z', inertia)
        if mass is None or inertia is None:
            raise ValueError("Parameter object must provide 'm' and 'J_z'.")
        
        # Кинетическая энергия вращения
        rotational_energy = 0.5 * inertia * r**2
        
        # Энергия бокового движения (упрощенно)
        lateral_energy = 0.5 * mass * (V * beta)**2
        
        # Полная энергия
        total_energy = rotational_energy + lateral_energy
        
        # Мощность (производная энергии)
        power = np.gradient(total_energy, time)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Кинетическая энергия вращения',
                'Энергия бокового движения', 
                'Полная энергия системы',
                'Мощность'
            )
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=rotational_energy, name='E_вращ', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=lateral_energy, name='E_бок', line=dict(color='green')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=total_energy, name='E_полная', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=power, name='Мощность', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Энергетический анализ")
        
        return fig, {
            'max_rotational_energy': np.max(rotational_energy),
            'max_lateral_energy': np.max(lateral_energy),
            'max_total_energy': np.max(total_energy),
            'max_power': np.max(power),
            'energy_efficiency': np.trapezoid(total_energy, time) / (mass * V**2 * time[-1])
        }
