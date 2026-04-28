# src/visualization/animator.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

class VehicleAnimator:
    """Класс для создания анимации движения автомобиля"""
    
    def __init__(self, results, vehicle_params, delta_signal=None):
        self.results = results
        self.params = vehicle_params
        self.delta_signal = delta_signal
        
        # Параметры отображения автомобиля
        self.car_length = 4.5  # м
        self.car_width = 1.8   # м
        
    def create_animation(self, interval=50, save_path=None):
        """
        Создание анимации движения
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Настройка области траектории
        ax1.set_xlim(np.min(self.results['X']) - 10, np.max(self.results['X']) + 10)
        ax1.set_ylim(np.min(self.results['Y']) - 10, np.max(self.results['Y']) + 10)
        ax1.set_xlabel('X координата, м')
        ax1.set_ylabel('Y координата, м')
        ax1.set_title('Траектория движения автомобиля')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Настройка области параметров
        time = self.results['time']
        ax2.set_xlim(0, np.max(time))
        ax2.set_ylim(-10, 10)
        ax2.set_xlabel('Время, с')
        ax2.set_ylabel('Параметры, °')
        ax2.set_title('Параметры движения')
        ax2.grid(True, alpha=0.3)
        
        # Инициализация элементов анимации
        trajectory_line, = ax1.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Траектория')
        car_patch = ax1.add_patch(Rectangle((0, 0), self.car_length, self.car_width, 
                                          fill=False, edgecolor='red', linewidth=2))
        
        # Линии параметров
        beta_line, = ax2.plot([], [], 'g-', linewidth=2, label='Угол скольжения β')
        r_line, = ax2.plot([], [], 'm-', linewidth=2, label='Угловая скорость r')
        delta_line, = ax2.plot([], [], 'r-', linewidth=2, label='Угол руля δ')
        
        ax2.legend(loc='upper right')
        
        def init():
            trajectory_line.set_data([], [])
            car_patch.set_xy((0, 0))
            beta_line.set_data([], [])
            r_line.set_data([], [])
            delta_line.set_data([], [])
            return trajectory_line, car_patch, beta_line, r_line, delta_line
        
        def animate(i):
            # Обновление траектории
            trajectory_line.set_data(self.results['X'][:i+1], self.results['Y'][:i+1])
            
            # Обновление положения автомобиля
            x, y = self.results['X'][i], self.results['Y'][i]
            psi = self.results['yaw_angle'][i]
            
            # Создание трансформации для поворота автомобиля
            t = transforms.Affine2D().rotate_around(x, y, psi) + ax1.transData
            car_patch.set_transform(t)
            car_patch.set_xy((x - self.car_length/2, y - self.car_width/2))
            
            # Обновление графиков параметров
            current_time = self.results['time'][:i+1]
            beta_line.set_data(current_time, np.degrees(self.results['beta'][:i+1]))
            r_line.set_data(current_time, np.degrees(self.results['angular_velocity'][:i+1]))
            
            if self.delta_signal:
                delta_values = [np.degrees(self.delta_signal(t)) for t in current_time]
                delta_line.set_data(current_time, delta_values)
            
            return trajectory_line, car_patch, beta_line, r_line, delta_line
        
        # Создание анимации
        anim = FuncAnimation(fig, animate, frames=len(self.results['time']),
                           init_func=init, interval=interval, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000/interval)
        
        plt.tight_layout()
        return anim
    
    def create_simple_animation(self, interval=50):
        """Упрощенная анимация только траектории"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.set_xlim(np.min(self.results['X']) - 10, np.max(self.results['X']) + 10)
        ax.set_ylim(np.min(self.results['Y']) - 10, np.max(self.results['Y']) + 10)
        ax.set_xlabel('X координата, м')
        ax.set_ylabel('Y координата, м')
        ax.set_title('Анимация движения автомобиля')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        trajectory_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
        car_marker, = ax.plot([], [], 'ro', markersize=10)
        
        def animate(i):
            trajectory_line.set_data(self.results['X'][:i+1], self.results['Y'][:i+1])
            car_marker.set_data([self.results['X'][i]], [self.results['Y'][i]])
            return trajectory_line, car_marker
        
        anim = FuncAnimation(fig, animate, frames=len(self.results['time']),
                           interval=interval, blit=True, repeat=True)
        
        return anim