# src/visualization/plotter.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

class ResultPlotter:
    """Класс для построения статических графиков результатов симуляции"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def plot_comprehensive_results(self, results, delta_signal=None, title="Результаты симуляции"):
        """
        Комплексный график всех параметров движения
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(4, 2, figure=fig)
        
        # 1. Траектория движения
        ax1 = fig.add_subplot(gs[0:2, 0])
        self._plot_trajectory(ax1, results, title="Траектория движения")
        
        # 2. Угол руля
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_steering_angle(ax2, results, delta_signal)
        
        # 3. Угол бокового скольжения
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_sideslip_angle(ax3, results)
        
        # 4. Угловая скорость
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_angular_velocity(ax4, results)
        
        # 5. Курсовой угол
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_yaw_angle(ax5, results)
        
        # 6. Фазовый портрет
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_phase_portrait(ax6, results)
        
        # 7. Ускорения
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_accelerations(ax7, results)
        
        plt.tight_layout()
        return fig
    
    def _plot_trajectory(self, ax, results, title):
        """Построение траектории движения"""
        ax.plot(results['X'], results['Y'], 'b-', linewidth=2, label='Траектория')
        ax.plot(results['X'][0], results['Y'][0], 'go', markersize=8, label='Старт')
        ax.plot(results['X'][-1], results['Y'][-1], 'ro', markersize=8, label='Финиш')
        ax.set_xlabel('X координата, м')
        ax.set_ylabel('Y координата, м')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
    
    def _plot_steering_angle(self, ax, results, delta_signal):
        """График угла поворота руля"""
        if delta_signal is not None:
            time = results['time']
            delta = [delta_signal(t) for t in time]
            ax.plot(time, np.degrees(delta), 'r-', linewidth=2)
            ax.set_ylabel('Угол руля, °')
            ax.set_title('Угол поворота рулевого колеса')
            ax.grid(True, alpha=0.3)
    
    def _plot_sideslip_angle(self, ax, results):
        """График угла бокового скольжения"""
        ax.plot(results['time'], np.degrees(results['beta']), 'g-', linewidth=2)
        ax.set_ylabel('Угол скольжения β, °')
        ax.set_xlabel('Время, с')
        ax.set_title('Угол бокового скольжения')
        ax.grid(True, alpha=0.3)
    
    def _plot_angular_velocity(self, ax, results):
        """График угловой скорости"""
        ax.plot(results['time'], np.degrees(results['angular_velocity']), 'm-', linewidth=2)
        ax.set_ylabel('Угловая скорость r, °/с')
        ax.set_xlabel('Время, с')
        ax.set_title('Угловая скорость')
        ax.grid(True, alpha=0.3)
    
    def _plot_yaw_angle(self, ax, results):
        """График курсового угла"""
        ax.plot(results['time'], np.degrees(results['yaw_angle']), 'c-', linewidth=2)
        ax.set_ylabel('Курсовой угол ψ, °')
        ax.set_xlabel('Время, с')
        ax.set_title('Курсовой угол')
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_portrait(self, ax, results):
        """Фазовый портрет"""
        beta_deg = np.degrees(results['beta'])
        r_deg = np.degrees(results['angular_velocity'])
        ax.plot(beta_deg, r_deg, 'b-', linewidth=1, alpha=0.7)
        ax.plot(beta_deg[0], r_deg[0], 'go', markersize=6, label='Начало')
        ax.plot(beta_deg[-1], r_deg[-1], 'ro', markersize=6, label='Конец')
        ax.set_xlabel('Угол скольжения β, °')
        ax.set_ylabel('Угловая скорость r, °/с')
        ax.set_title('Фазовый портрет (β vs r)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_accelerations(self, ax, results):
        """График поперечного ускорения"""
        # Поперечное ускорение: ay = V * (dβ/dt + r)
        dt = np.diff(results['time'])
        dbeta_dt = np.diff(results['beta']) / dt
        # Добавляем нуль в конец для совпадения размерностей
        dbeta_dt = np.append(dbeta_dt, dbeta_dt[-1])
        
        V = 20.0  # Скорость из параметров
        ay = V * (dbeta_dt + results['angular_velocity'])
        
        ax.plot(results['time'], ay, 'orange', linewidth=2, label='Поперечное ускорение')
        ax.set_ylabel('Ускорение, м/с²')
        ax.set_xlabel('Время, с')
        ax.set_title('Поперечное ускорение')
        ax.grid(True, alpha=0.3)
        ax.legend()

def create_comparison_plot(results_list, labels, title="Сравнение маневров"):
    """Создание сравнительных графиков для нескольких симуляций"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['b', 'r', 'g', 'm', 'c']
    
    # Траектории
    for i, results in enumerate(results_list):
        axes[0,0].plot(results['X'], results['Y'], color=colors[i], linewidth=2, label=labels[i])
    axes[0,0].set_xlabel('X, м')
    axes[0,0].set_ylabel('Y, м')
    axes[0,0].set_title('Сравнение траекторий')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axis('equal')
    
    # Углы скольжения
    for i, results in enumerate(results_list):
        axes[0,1].plot(results['time'], np.degrees(results['beta']), 
                      color=colors[i], linewidth=2, label=labels[i])
    axes[0,1].set_xlabel('Время, с')
    axes[0,1].set_ylabel('β, °')
    axes[0,1].set_title('Углы бокового скольжения')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Угловые скорости
    for i, results in enumerate(results_list):
        axes[1,0].plot(results['time'], np.degrees(results['angular_velocity']), 
                      color=colors[i], linewidth=2, label=labels[i])
    axes[1,0].set_xlabel('Время, с')
    axes[1,0].set_ylabel('r, °/с')
    axes[1,0].set_title('Угловые скорости')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Фазовые портреты
    for i, results in enumerate(results_list):
        beta_deg = np.degrees(results['beta'])
        r_deg = np.degrees(results['angular_velocity'])
        axes[1,1].plot(beta_deg, r_deg, color=colors[i], linewidth=1, 
                      label=labels[i], alpha=0.7)
    axes[1,1].set_xlabel('β, °')
    axes[1,1].set_ylabel('r, °/с')
    axes[1,1].set_title('Фазовые портреты')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig