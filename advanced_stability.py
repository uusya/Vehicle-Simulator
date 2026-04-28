# src/analysis/advanced_stability.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AdvancedStabilityAnalysis:
    """Расширенный анализ устойчивости и управляемости"""
    
    def __init__(self):
        pass
    
    def analyze_stability_modes(self, vehicle_params, speed_range=(5, 40), num_points=50):
        """
        Анализ мод устойчивости в диапазоне скоростей
        """
        speeds = np.linspace(speed_range[0], speed_range[1], num_points)
        
        stability_data = {
            'speeds': speeds,
            'eigenvalues_real': [],
            'eigenvalues_imag': [],
            'natural_frequencies': [],
            'damping_ratios': [],
            'stability_modes': []
        }
        
        for V in speeds:
            # Расчет матрицы состояния
            m, J_z, a, b, C_f, C_r = (vehicle_params.m, vehicle_params.J_z, 
                                     vehicle_params.a, vehicle_params.b, 
                                     vehicle_params.C_f, vehicle_params.C_r)
            
            A11 = -(C_f + C_r) / (m * V)
            A12 = -1 - (C_f * a - C_r * b) / (m * V**2)
            A21 = -(C_f * a - C_r * b) / J_z
            A22 = -(C_f * a**2 + C_r * b**2) / (J_z * V)
            
            A = np.array([[A11, A12], [A21, A22]])
            eigenvalues = np.linalg.eigvals(A)
            
            stability_data['eigenvalues_real'].append(eigenvalues.real)
            stability_data['eigenvalues_imag'].append(eigenvalues.imag)
            
            # Расчет частот и коэффициентов демпфирования
            for eig in eigenvalues:
                wn = np.abs(eig)  # Собственная частота
                zeta = -eig.real / wn if wn > 0 else 0  # Коэффициент демпфирования
                
                stability_data['natural_frequencies'].append(wn)
                stability_data['damping_ratios'].append(zeta)
                
                # Определение типа моды
                if zeta > 1:
                    mode_type = "Апериодическая"
                elif zeta > 0:
                    mode_type = "Колебательная"
                else:
                    mode_type = "Неустойчивая"
                
                stability_data['stability_modes'].append(mode_type)
        
        return stability_data
    
    def create_stability_dashboard(self, stability_data):
        """
        Создание дашборда анализа устойчивости
        """
        speeds = stability_data['speeds']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Действительная часть собственных значений',
                'Мнимая часть собственных значений',
                'Коэффициенты демпфирования',
                'Карта устойчивости'
            )
        )
        
        # Действительные части
        for i in range(2):  # Для двух собственных значений
            real_parts = [eig[i].real for eig in stability_data['eigenvalues_real']]
            fig.add_trace(
                go.Scatter(x=speeds, y=real_parts, name=f'λ{i+1} реальная', 
                          line=dict(width=2)),
                row=1, col=1
            )
        
        # Мнимые части
        for i in range(2):
            imag_parts = [eig[i].imag for eig in stability_data['eigenvalues_imag']]
            fig.add_trace(
                go.Scatter(x=speeds, y=imag_parts, name=f'λ{i+1} мнимая',
                          line=dict(width=2)),
                row=1, col=2
            )
        
        # Коэффициенты демпфирования
        damping_ratios = np.array(stability_data['damping_ratios']).reshape(-1, 2)
        for i in range(2):
            fig.add_trace(
                go.Scatter(x=speeds, y=damping_ratios[:, i], name=f'ζ{i+1}',
                          line=dict(width=2)),
                row=2, col=1
            )
        
        # Карта устойчивости
        for i in range(2):
            real_parts = [eig[i].real for eig in stability_data['eigenvalues_real']]
            imag_parts = [eig[i].imag for eig in stability_data['eigenvalues_imag']]
            
            # Цвет в зависимости от устойчивости
            colors = ['red' if real > 0 else 'green' for real in real_parts]
            
            fig.add_trace(
                go.Scatter(x=real_parts, y=imag_parts, 
                          mode='markers+lines',
                          name=f'λ{i+1} траектория',
                          marker=dict(color=colors, size=6),
                          line=dict(color='gray', width=1)),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Анализ устойчивости по скоростям")
        fig.update_xaxes(title_text="Скорость, м/с", row=1, col=1)
        fig.update_xaxes(title_text="Скорость, м/с", row=1, col=2)
        fig.update_xaxes(title_text="Скорость, м/с", row=2, col=1)
        fig.update_xaxes(title_text="Re(λ)", row=2, col=2)
        fig.update_yaxes(title_text="Re(λ)", row=1, col=1)
        fig.update_yaxes(title_text="Im(λ)", row=1, col=2)
        fig.update_yaxes(title_text="ζ", row=2, col=1)
        fig.update_yaxes(title_text="Im(λ)", row=2, col=2)
        
        return fig
    
    def calculate_handling_indicators(self, vehicle_params, V):
        """
        Расчет индикаторов управляемости
        """
        m, L, a, b, C_f, C_r = (vehicle_params.m, vehicle_params.L,
                               vehicle_params.a, vehicle_params.b,
                               vehicle_params.C_f, vehicle_params.C_r)
        
        # Градиент недостаточной поворачиваемости
        K = (m / L) * (b / C_f - a / C_r)
        
        # Критическая скорость
        V_crit = np.sqrt(-L / K) if K < 0 else float('inf')
        
        # Характеристическая скорость
        V_char = np.sqrt(L / K) if K > 0 else float('inf')
        
        # Коэффициенты матрицы состояния
        A11 = -(C_f + C_r) / (m * V)
        A12 = -1 - (C_f * a - C_r * b) / (m * V**2)
        A21 = -(C_f * a - C_r * b) / vehicle_params.J_z
        A22 = -(C_f * a**2 + C_r * b**2) / (vehicle_params.J_z * V)
        
        # Собственные значения
        A = np.array([[A11, A12], [A21, A22]])
        eigenvalues = np.linalg.eigvals(A)
        
        indicators = {
            'understeer_gradient': K,
            'critical_speed': V_crit,
            'characteristic_speed': V_char,
            'eigenvalues': eigenvalues,
            'stability_margin': min(-eig.real for eig in eigenvalues if eig.real < 0),
            'natural_frequency': np.mean(np.abs(eigenvalues)),
            'damping_ratio': np.mean([-eig.real/np.abs(eig) for eig in eigenvalues if np.abs(eig) > 0])
        }
        
        return indicators