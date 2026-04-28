# src/tire_models.py (дополненная версия)
import numpy as np

class LinearTireModel:
    """Линейная модель шины (базовая)"""
    
    def __init__(self, stiffness):
        self.stiffness = stiffness
        
    def lateral_force(self, slip_angle):
        """Вычисление боковой силы (линейная модель)"""
        if abs(slip_angle) < 1e-12:
            return 0.0
        return self.stiffness * slip_angle
    
    def get_characteristics(self, slip_angles):
        """Возвращает характеристику шины для диапазона углов"""
        forces = [self.lateral_force(alpha) for alpha in slip_angles]
        return forces

class FialaTireModel:
    """
    Модель шины Фьяла - аналитическое приближение нелинейной характеристики
    """
    
    def __init__(self, stiffness, friction_coef=1.0, normal_load=4000):
        self.stiffness = stiffness
        self.friction_coef = friction_coef
        self.normal_load = normal_load
        
    def lateral_force(self, slip_angle):
        """Вычисление боковой силы по модели Фьяла"""
        alpha = slip_angle
        C_alpha = self.stiffness
        mu = self.friction_coef
        F_z = self.normal_load
        
        if abs(alpha) < 1e-12:
            return 0.0
            
        # Максимальная достижимая сила
        F_max = mu * F_z
        
        if F_max <= 0:
            return 0.0
            
        # Критический угол увода
        alpha_crit = np.arctan(3 * F_max / C_alpha)
        
        if abs(alpha) <= alpha_crit:
            # Упругая область - упрощенная формула
            tan_alpha = np.tan(alpha)
            Fy = -C_alpha * tan_alpha * (1 - abs(tan_alpha) / (3 * np.tan(alpha_crit)))
        else:
            # Область полного скольжения
            Fy = -F_max * np.sign(alpha)
            
        return Fy
    
    def get_characteristics(self, slip_angles):
        """Возвращает характеристику шины для диапазона углов"""
        forces = [self.lateral_force(alpha) for alpha in slip_angles]
        return forces

class SimplifiedFialaTireModel:
    """
    Упрощенная модель Фьяла для лучшего согласования с линейной моделью при малых углах
    """
    
    def __init__(self, stiffness, max_force=5000, transition_angle=0.1):
        self.stiffness = stiffness
        self.max_force = max_force
        self.transition_angle = transition_angle
        
    def lateral_force(self, slip_angle):
        """Упрощенная нелинейная модель с плавным переходом"""
        if abs(slip_angle) < 1e-12:
            return 0.0
            
        alpha = slip_angle
        
        # Линейная сила
        linear_force = self.stiffness * alpha
        
        # Коэффициент насыщения (0-1)
        saturation = min(1.0, abs(alpha) / self.transition_angle)
        
        # Плавный переход к насыщению
        if saturation <= 1.0:
            # Плавное уменьшение эффективной жесткости
            effective_force = linear_force * (1 - 0.5 * saturation**2)
        else:
            # Полное насыщение
            effective_force = np.sign(alpha) * self.max_force * (1 - 0.5 / saturation)
            
        return effective_force
    
    def get_characteristics(self, slip_angles):
        """Возвращает характеристику шины для диапазона углов"""
        forces = [self.lateral_force(alpha) for alpha in slip_angles]
        return forces

class PiecewiseLinearTireModel:
    """
    Кусочно-линейная модель шины с насыщением
    """
    
    def __init__(self, stiffness, saturation_angle=0.1, max_force=5000):
        self.stiffness = stiffness
        self.saturation_angle = saturation_angle
        self.max_force = max_force
        
    def lateral_force(self, slip_angle):
        """Вычисление боковой силы (кусочно-линейная модель)"""
        if abs(slip_angle) < 1e-12:
            return 0.0
            
        alpha = slip_angle
        
        # Линейная область
        linear_force = self.stiffness * alpha
        
        # Проверка насыщения
        if abs(linear_force) <= self.max_force:
            return linear_force
        else:
            return np.sign(alpha) * self.max_force
    
    def get_characteristics(self, slip_angles):
        """Возвращает характеристику шины для диапазона углов"""
        forces = [self.lateral_force(alpha) for alpha in slip_angles]
        return forces

class PacejkaTireModel:
    """
    Упрощенная версия 'Магической формулы' Пейджака
    Fy = D * sin(C * arctan(B * α - E*(B * α - arctan(B * α))))
    """
    
    def __init__(self, B=8.0, C=1.3, D=5000.0, E=0.5):
        self.B = B  # stiffness factor
        self.C = C  # shape factor  
        self.D = D  # peak value
        self.E = E  # curvature factor
        
    def lateral_force(self, slip_angle):
        """Упрощенная магическая формула"""
        alpha = slip_angle
        B, C, D, E = self.B, self.C, self.D, self.E
        
        x = B * alpha
        Fy = D * np.sin(C * np.arctan(x - E * (x - np.arctan(x))))
        
        return Fy
    
    def get_characteristics(self, slip_angles):
        """Возвращает характеристику шины для диапазона углов"""
        forces = [self.lateral_force(alpha) for alpha in slip_angles]
        return forces

class MagicFormulaTireModel:
    """
    Альтернативное название для Pacejka (для совместимости)
    """
    def __init__(self, B=8.0, C=1.3, D=5000.0, E=0.5):
        self.pacejka = PacejkaTireModel(B, C, D, E)
        
    def lateral_force(self, slip_angle):
        return self.pacejka.lateral_force(slip_angle)
        
    def get_characteristics(self, slip_angles):
        return self.pacejka.get_characteristics(slip_angles)

def compare_tire_models(slip_angles, models_dict, title="Сравнение моделей шин"):
    """
    Сравнительный анализ различных моделей шин
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    for name, model in models_dict.items():
        forces = model.get_characteristics(slip_angles)
        plt.plot(np.degrees(slip_angles), [f/1000 for f in forces], 
                linewidth=2, label=name)
    
    plt.xlabel('Угол увода, °')
    plt.ylabel('Боковая сила, кН')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def create_tire_model_comparison():
    """Создание сравнительного анализа моделей шин"""
    # Диапазон углов увода
    slip_angles = np.linspace(-0.3, 0.3, 100)
    
    # Создание различных моделей
    models = {
        'Линейная (C=80кН/рад)': LinearTireModel(80000),
        'Кусочно-линейная': PiecewiseLinearTireModel(80000, saturation_angle=0.08),
        'Упрощенная Фьяла': SimplifiedFialaTireModel(80000),
        'Магическая формула': PacejkaTireModel()
    }
    
    return compare_tire_models(slip_angles, models)