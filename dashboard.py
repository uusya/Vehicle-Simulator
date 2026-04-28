# src/visualization/dashboard.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

class AdvancedDashboard:
    """Продвинутые дашборды для анализа нелинейной динамики"""
    
    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize
    
    def create_tire_characteristics_dashboard(self, tire_models_dict, slip_range=(-0.3, 0.3)):
        """
        Дашборд характеристик шин
        """
        slip_angles = np.linspace(slip_range[0], slip_range[1], 100)
        
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Характеристики шин
        ax1 = fig.add_subplot(gs[0, 0])
        for name, model in tire_models_dict.items():
            forces = model.get_characteristics(slip_angles)
            ax1.plot(np.degrees(slip_angles), [f/1000 for f in forces], 
                    linewidth=2, label=name)
        ax1.set_xlabel('Угол увода, °')
        ax1.set_ylabel('Боковая сила, кН')
        ax1.set_title('Характеристики различных моделей шин')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Жесткость шин (производная dFy/dα)
        ax2 = fig.add_subplot(gs[0, 1])
        for name, model in tire_models_dict.items():
            forces = model.get_characteristics(slip_angles)
            stiffness = np.gradient(forces, slip_angles)
            ax2.plot(np.degrees(slip_angles), [s/1000 for s in stiffness], 
                    linewidth=2, label=name)
        ax2.set_xlabel('Угол увода, °')
        ax2.set_ylabel('Жесткость, кН/рад')
        ax2.set_title('Эффективная жесткость шин')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Нормированные характеристики
        ax3 = fig.add_subplot(gs[1, 0])
        for name, model in tire_models_dict.items():
            forces = model.get_characteristics(slip_angles)
            max_force = max(np.abs(forces))
            if max_force > 0:
                normalized_forces = [f/max_force for f in forces]
                ax3.plot(np.degrees(slip_angles), normalized_forces, 
                        linewidth=2, label=name)
        ax3.set_xlabel('Угол увода, °')
        ax3.set_ylabel('Нормированная сила')
        ax3.set_title('Нормированные характеристики шин')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Области работы шин
        ax4 = fig.add_subplot(gs[1, 1])
        typical_angles = {
            'Нормальная езда': (-0.03, 0.03),
            'Активное маневрирование': (-0.08, 0.08),
            'Предельные режимы': (-0.15, 0.15)
        }
        
        colors = ['green', 'orange', 'red']
        for (region, (min_angle, max_angle)), color in zip(typical_angles.items(), colors):
            ax4.axvspan(np.degrees(min_angle), np.degrees(max_angle), 
                       alpha=0.2, color=color, label=region)
        
        for name, model in tire_models_dict.items():
            forces = model.get_characteristics(slip_angles)
            ax4.plot(np.degrees(slip_angles), [f/1000 for f in forces], 
                    linewidth=2, alpha=0.7)
        
        ax4.set_xlabel('Угол увода, °')
        ax4.set_ylabel('Боковая сила, кН')
        ax4.set_title('Области работы шин')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        return fig