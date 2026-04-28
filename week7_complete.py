# week7_complete.py
"""
ПОЛНАЯ ВЕРСИЯ НЕДЕЛИ 7 - Машинное обучение и предиктивная аналитика
Запуск: streamlit run week7_complete.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== БАЗОВЫЕ КЛАССЫ ====================

class VehicleParameters:
    def __init__(self):
        self.m = 1200.0
        self.J_z = 1500.0
        self.a = 1.4
        self.b = 1.6
        self.L = self.a + self.b
        self.C_f = 80000.0
        self.C_r = 100000.0
        self.V = 20.0

class LinearTireModel:
    def __init__(self, stiffness):
        self.stiffness = stiffness
        
    def lateral_force(self, slip_angle):
        return self.stiffness * slip_angle

class PiecewiseLinearTireModel:
    def __init__(self, stiffness, max_force=5000):
        self.stiffness = stiffness
        self.max_force = max_force
        
    def lateral_force(self, slip_angle):
        linear_force = self.stiffness * slip_angle
        if abs(linear_force) <= self.max_force:
            return linear_force
        else:
            return np.sign(slip_angle) * self.max_force

def step_steer(t, amplitude=0.1, start_time=1.0):
    return amplitude if t >= start_time else 0.0

def sine_steer(t, amplitude=0.1, frequency=0.5, start_time=1.0):
    if t < start_time:
        return 0.0
    return amplitude * np.sin(2 * np.pi * frequency * (t - start_time))

def double_lane_change(t, amplitude=0.08, frequency=0.3):
    if t < 1.0:
        return 0.0
    elif t < 4.0:
        return amplitude * np.sin(2 * np.pi * frequency * (t - 1.0))
    elif t < 7.0:
        return -amplitude * np.sin(2 * np.pi * frequency * (t - 4.0))
    else:
        return 0.0

class NonlinearBicycleModel:
    def __init__(self, vehicle_params, front_tire_model, rear_tire_model):
        self.params = vehicle_params
        self.front_tire = front_tire_model
        self.rear_tire = rear_tire_model
        
    def equations_of_motion(self, t, state, delta_func, V):
        beta, r, psi, X, Y = state
        delta = delta_func(t)
        
        alpha_f = delta - (beta + self.params.a * r / V)
        alpha_r = -(beta - self.params.b * r / V)
        
        F_yf = self.front_tire.lateral_force(alpha_f)
        F_yr = self.rear_tire.lateral_force(alpha_r)
        
        dbeta_dt = (F_yf + F_yr) / (self.params.m * V) - r
        dr_dt = (F_yf * self.params.a - F_yr * self.params.b) / self.params.J_z
        dpsi_dt = r
        dX_dt = V * np.cos(psi - beta)
        dY_dt = V * np.sin(psi - beta)
        
        return [dbeta_dt, dr_dt, dpsi_dt, dX_dt, dY_dt]
    
    def simulate(self, delta_func, V, t_span):
        initial_state = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        solution = solve_ivp(
            fun=lambda t, y: self.equations_of_motion(t, y, delta_func, V),
            t_span=t_span,
            y0=initial_state,
            method='RK45'
        )
        
        return solution

def create_nonlinear_simulator(vehicle_params, front_stiffness, rear_stiffness, tire_model_type):
    if tire_model_type == 'linear':
        front_tire = LinearTireModel(front_stiffness)
        rear_tire = LinearTireModel(rear_stiffness)
    elif tire_model_type == 'piecewise':
        front_tire = PiecewiseLinearTireModel(front_stiffness)
        rear_tire = PiecewiseLinearTireModel(rear_stiffness)
    else:
        front_tire = LinearTireModel(front_stiffness)
        rear_tire = LinearTireModel(rear_stiffness)
    
    return NonlinearBicycleModel(vehicle_params, front_tire, rear_tire)

def get_simulation_results(solution):
    t = solution.t
    beta = solution.y[0, :]
    r = solution.y[1, :]
    psi = solution.y[2, :]
    X = solution.y[3, :]
    Y = solution.y[4, :]
    
    return {
        'time': t, 'beta': beta, 'angular_velocity': r,
        'yaw_angle': psi, 'X': X, 'Y': Y, 'success': solution.success
    }

# ==================== МАШИННОЕ ОБУЧЕНИЕ - НЕДЕЛЯ 7 ====================

class MLPredictor:
    """Класс для предсказания поведения автомобиля с помощью ML"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.training_data = None
        
    def generate_training_data(self, num_samples=1000):
        """Генерация данных для обучения"""
        st.info(f"🧠 Генерация {num_samples} образцов данных для обучения...")
        
        X_data = []
        y_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_samples):
            if i % 100 == 0:
                progress = (i + 1) / num_samples
                progress_bar.progress(progress)
                status_text.text(f"Сгенерировано {i+1}/{num_samples} образцов...")
            
            # Случайные параметры автомобиля
            params = VehicleParameters()
            params.m = np.random.uniform(800, 2000)
            params.C_f = np.random.uniform(40000, 150000)
            params.C_r = np.random.uniform(40000, 150000)
            params.a = np.random.uniform(1.0, 2.0)
            params.b = params.L - params.a
            params.V = np.random.uniform(5, 40)
            
            # Случайный маневр
            maneuver_type = np.random.choice(['step', 'sine'])
            if maneuver_type == 'step':
                amplitude = np.random.uniform(0.05, 0.15)
                maneuver_func = lambda t: step_steer(t, amplitude, 1.0)
            else:
                amplitude = np.random.uniform(0.03, 0.08)
                frequency = np.random.uniform(0.3, 0.8)
                maneuver_func = lambda t: sine_steer(t, amplitude, frequency, 1.0)
            
            # Запуск симуляции
            try:
                simulator = create_nonlinear_simulator(params, params.C_f, params.C_r, 'linear')
                solution = simulator.simulate(maneuver_func, params.V, (0, 8))
                
                if solution.success:
                    results = get_simulation_results(solution)
                    
                    # Входные признаки
                    features = [
                        params.m, params.C_f, params.C_r, 
                        params.a/params.L, params.V,
                        amplitude, 1 if maneuver_type == 'step' else 0
                    ]
                    
                    # Целевые переменные (метрики производительности)
                    targets = [
                        np.max(np.abs(results['beta'])),
                        np.max(np.abs(results['angular_velocity'])),
                        results['Y'][-1],
                        self._calculate_settling_time(results['beta'], results['time'])
                    ]
                    
                    X_data.append(features)
                    y_data.append(targets)
                    
            except:
                continue
        
        progress_bar.progress(1.0)
        status_text.text("✅ Данные сгенерированы!")
        
        self.training_data = {
            'X': np.array(X_data),
            'y': np.array(y_data),
            'feature_names': ['mass', 'C_f', 'C_r', 'cog_ratio', 'speed', 'amplitude', 'is_step'],
            'target_names': ['max_beta', 'max_r', 'final_y', 'settling_time']
        }
        
        return self.training_data
    
    def _calculate_settling_time(self, signal, time, threshold=0.02):
        """Расчет времени установления"""
        if len(signal) < 2:
            return time[-1]
        final_value = signal[-1]
        for i, value in enumerate(signal):
            if abs(value - final_value) < threshold:
                return time[i]
        return time[-1]
    
    def train_models(self):
        """Обучение нескольких ML моделей"""
        if self.training_data is None:
            st.error("❌ Сначала сгенерируйте данные для обучения")
            return False
        
        X = self.training_data['X']
        y = self.training_data['y']
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Масштабирование features
        self.scalers['X'] = StandardScaler()
        X_train_scaled = self.scalers['X'].fit_transform(X_train)
        X_test_scaled = self.scalers['X'].transform(X_test)
        
        # Масштабирование targets
        self.scalers['y'] = StandardScaler()
        y_train_scaled = self.scalers['y'].fit_transform(y_train)
        y_test_scaled = self.scalers['y'].transform(y_test)
        
        # Модели для обучения
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        training_results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models_to_train.items()):
            progress = (i + 1) / len(models_to_train)
            progress_bar.progress(progress)
            status_text.text(f"Обучение модели: {name}...")
            
            try:
                # Обучение модели
                if name == 'Neural Network':
                    model.fit(X_train_scaled, y_train_scaled)
                else:
                    model.fit(X_train, y_train)  # Деревья не требуют масштабирования
                
                # Предсказания
                if name == 'Neural Network':
                    y_pred_scaled = model.predict(X_test_scaled)
                    y_pred = self.scalers['y'].inverse_transform(y_pred_scaled)
                else:
                    y_pred = model.predict(X_test)
                
                # Метрики
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                training_results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                self.models[name] = model
                
            except Exception as e:
                st.warning(f"⚠️ Ошибка при обучении {name}: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Все модели обучены!")
        
        self.is_trained = True
        self.training_results = training_results
        
        return True
    
    def predict_performance(self, input_features):
        """Предсказание производительности для новых параметров"""
        if not self.is_trained:
            st.error("❌ Модели не обучены")
            return None
        
        # Преобразование входных данных
        X_input = np.array([input_features])
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'Neural Network':
                    X_scaled = self.scalers['X'].transform(X_input)
                    y_pred_scaled = model.predict(X_scaled)
                    y_pred = self.scalers['y'].inverse_transform(y_pred_scaled)[0]
                else:
                    y_pred = model.predict(X_input)[0]
                
                predictions[name] = {
                    'max_beta': y_pred[0],
                    'max_r': y_pred[1],
                    'final_y': y_pred[2],
                    'settling_time': y_pred[3]
                }
                
            except Exception as e:
                st.warning(f"⚠️ Ошибка предсказания для {name}: {str(e)}")
        
        return predictions
    
    def save_models(self, filepath="ml_models.pkl"):
        """Сохранение обученных моделей"""
        if not self.is_trained:
            st.error("❌ Нет обученных моделей для сохранения")
            return False
        
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'training_data': self.training_data,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            st.success(f"✅ Модели сохранены в {filepath}")
            return True
            
        except Exception as e:
            st.error(f"❌ Ошибка сохранения: {str(e)}")
            return False
    
    def load_models(self, filepath="ml_models.pkl"):
        """Загрузка обученных моделей"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.training_data = model_data['training_data']
            self.is_trained = model_data['is_trained']
            
            st.success(f"✅ Модели загружены из {filepath}")
            return True
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {str(e)}")
            return False

class RealTimeOptimizer:
    """Реалтайм оптимизатор с использованием ML"""
    
    def __init__(self, ml_predictor):
        self.ml_predictor = ml_predictor
        self.optimization_history = []
    
    def ml_objective_function(self, x, target_metric):
        """Целевая функция на основе ML предсказаний"""
        # x = [C_f, C_r, cog_ratio, speed]
        C_f, C_r, cog_ratio, speed = x
        
        # Базовые параметры
        base_params = [1200, C_f, C_r, cog_ratio, speed, 0.1, 1]  # step maneuver
        
        # Предсказание метрик
        predictions = self.ml_predictor.predict_performance(base_params)
        
        if not predictions:
            return 1000  # Штраф
        
        # Используем предсказания Random Forest (обычно наиболее надежные)
        if 'Random Forest' in predictions:
            pred = predictions['Random Forest']
        else:
            # Берем первую доступную модель
            pred = list(predictions.values())[0]
        
        # Целевая метрика
        if target_metric == 'stability':
            return pred['max_beta'] + 0.1 * pred['max_r']
        elif target_metric == 'responsiveness':
            return pred['settling_time']
        elif target_metric == 'safety':
            return pred['max_beta'] + 0.5 * pred['max_r']
        else:
            return pred['max_beta']
    
    def optimize_with_ml(self, target_metric, max_iter=20):
        """Быстрая оптимизация с использованием ML"""
        bounds = [
            (40000, 150000),  # C_f
            (40000, 150000),  # C_r
            (0.3, 0.7),       # cog_ratio
            (5, 40)           # speed
        ]
        
        x0 = [80000, 100000, 0.47, 20]
        
        self.optimization_history = []
        
        try:
            result = minimize(
                fun=lambda x: self.ml_objective_function(x, target_metric),
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
            
            return result
            
        except Exception as e:
            st.error(f"❌ Ошибка ML оптимизации: {str(e)}")
            return None

class PredictiveAnalytics:
    """Предиктивная аналитика и визуализация"""
    
    def __init__(self, ml_predictor):
        self.ml_predictor = ml_predictor
    
    def create_feature_importance_plot(self):
        """Анализ важности признаков"""
        if not self.ml_predictor.is_trained:
            return None
        
        # Для Random Forest можно получить важность признаков
        if 'Random Forest' in self.ml_predictor.models:
            rf_model = self.ml_predictor.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=self.ml_predictor.training_data['feature_names'],
                y=feature_importance,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Важность признаков (Random Forest)",
                xaxis_title="Признаки",
                yaxis_title="Важность",
                height=400
            )
            
            return fig
        
        return None
    
    def create_prediction_comparison(self, actual_params, ml_predictions):
        """Сравнение ML предсказаний с реальной симуляцией"""
        # Запуск реальной симуляции для сравнения
        params = VehicleParameters()
        params.m = actual_params[0]
        params.C_f = actual_params[1]
        params.C_r = actual_params[2]
        params.a = actual_params[3] * params.L
        params.b = params.L - params.a
        params.V = actual_params[4]
        
        simulator = create_nonlinear_simulator(params, params.C_f, params.C_r, 'linear')
        maneuver = lambda t: step_steer(t, 0.1, 1.0)
        solution = simulator.simulate(maneuver, params.V, (0, 8))
        
        if solution.success:
            results = get_simulation_results(solution)
            actual_metrics = {
                'max_beta': np.max(np.abs(results['beta'])),
                'max_r': np.max(np.abs(results['angular_velocity'])),
                'final_y': results['Y'][-1],
                'settling_time': self._calculate_settling_time(results['beta'], results['time'])
            }
            
            # Создание графика сравнения
            models = list(ml_predictions.keys())
            metrics = ['max_beta', 'max_r', 'final_y', 'settling_time']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Макс. угол скольжения', 'Макс. угловая скорость',
                              'Конечное смещение', 'Время установления')
            )
            
            for i, metric in enumerate(metrics):
                row = i // 2 + 1
                col = i % 2 + 1
                
                # ML предсказания
                ml_values = [ml_predictions[model][metric] for model in models]
                
                fig.add_trace(
                    go.Bar(x=models, y=ml_values, name='ML предсказания',
                          marker_color='lightblue'),
                    row=row, col=col
                )
                
                # Реальные значения
                fig.add_trace(
                    go.Scatter(x=models, y=[actual_metrics[metric]] * len(models),
                              mode='markers', name='Реальная симуляция',
                              marker=dict(size=10, color='red', symbol='x')),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="Сравнение ML предсказаний с реальной симуляцией")
            return fig, actual_metrics
        
        return None, None
    
    def _calculate_settling_time(self, signal, time, threshold=0.02):
        """Расчет времени установления"""
        if len(signal) < 2:
            return time[-1]
        final_value = signal[-1]
        for i, value in enumerate(signal):
            if abs(value - final_value) < threshold:
                return time[i]
        return time[-1]
    
    def create_performance_surface(self):
        """Создание 3D поверхности производительности"""
        if not self.ml_predictor.is_trained:
            return None
        
        # Создание сетки параметров
        C_f_range = np.linspace(50000, 120000, 20)
        C_r_range = np.linspace(50000, 120000, 20)
        C_f_grid, C_r_grid = np.meshgrid(C_f_range, C_r_range)
        
        # Предсказания для сетки
        Z_grid = np.zeros_like(C_f_grid)
        
        for i in range(C_f_grid.shape[0]):
            for j in range(C_f_grid.shape[1]):
                features = [1200, C_f_grid[i,j], C_r_grid[i,j], 0.47, 20, 0.1, 1]
                predictions = self.ml_predictor.predict_performance(features)
                
                if predictions and 'Random Forest' in predictions:
                    Z_grid[i,j] = predictions['Random Forest']['max_beta']
        
        # 3D поверхность
        fig = go.Figure(data=[
            go.Surface(
                x=C_f_grid, y=C_r_grid, z=Z_grid,
                colorscale='Viridis',
                opacity=0.8
            )
        ])
        
        fig.update_layout(
            title="3D поверхность производительности (Max Beta)",
            scene=dict(
                xaxis_title='C_f (Н/рад)',
                yaxis_title='C_r (Н/рад)',
                zaxis_title='Max Beta (рад)'
            ),
            height=600
        )
        
        return fig

# ==================== ПРИЛОЖЕНИЕ STREAMLIT ====================

def main():
    st.set_page_config(
        page_title="🏎️ Симулятор - Неделя 7",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title("🏎️ Симулятор управляемости - Неделя 7")
    st.markdown("### 🧠 Машинное обучение и предиктивная аналитика")
    
    # Инициализация компонентов
    if 'ml_predictor' not in st.session_state:
        st.session_state.ml_predictor = MLPredictor()
    if 'realtime_optimizer' not in st.session_state:
        st.session_state.realtime_optimizer = RealTimeOptimizer(st.session_state.ml_predictor)
    if 'predictive_analytics' not in st.session_state:
        st.session_state.predictive_analytics = PredictiveAnalytics(st.session_state.ml_predictor)
    
    # Навигация по вкладкам
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Обучение ML", 
        "⚡ ML Оптимизация",
        "📊 Аналитика",
        "🚗 Симуляция"
    ])
    
    with tab1:
        show_ml_training_tab()
    
    with tab2:
        show_ml_optimization_tab()
    
    with tab3:
        show_analytics_tab()
    
    with tab4:
        show_simulation_tab()

def show_ml_training_tab():
    st.header("🧠 Обучение моделей машинного обучения")
    
    st.markdown("""
    **Обучение ML моделей для предсказания поведения автомобиля**
    - Генерация тренировочных данных с помощью физической модели
    - Обучение нескольких алгоритмов ML
    - Сравнение точности предсказаний
    - Сохранение и загрузка моделей
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Генерация данных")
        num_samples = st.slider("Количество образцов:", 100, 5000, 1000, key="ml_samples")
        
        if st.button("🔄 Сгенерировать данные", key="generate_data_btn"):
            with st.spinner("Генерация данных..."):
                training_data = st.session_state.ml_predictor.generate_training_data(num_samples)
                
                if training_data:
                    st.success(f"✅ Сгенерировано {len(training_data['X'])} образцов")
                    
                    # Статистика данных
                    st.subheader("📊 Статистика данных")
                    df_stats = pd.DataFrame({
                        'Признак': training_data['feature_names'],
                        'Min': np.min(training_data['X'], axis=0),
                        'Max': np.max(training_data['X'], axis=0),
                        'Mean': np.mean(training_data['X'], axis=0)
                    })
                    st.dataframe(df_stats)
    
    with col2:
        st.subheader("Обучение моделей")
        
        if st.button("🎯 Обучить модели", key="train_models_btn"):
            if st.session_state.ml_predictor.training_data is None:
                st.error("❌ Сначала сгенерируйте данные")
            else:
                with st.spinner("Обучение моделей..."):
                    success = st.session_state.ml_predictor.train_models()
                    
                    if success:
                        st.success("✅ Модели успешно обучены!")
                        
                        # Результаты обучения
                        st.subheader("📈 Результаты обучения")
                        
                        results = st.session_state.ml_predictor.training_results
                        metrics_data = []
                        
                        for name, result in results.items():
                            metrics_data.append({
                                'Модель': name,
                                'MSE': f"{result['mse']:.6f}",
                                'R² Score': f"{result['r2']:.3f}"
                            })
                        
                        st.dataframe(pd.DataFrame(metrics_data))
    
    # Управление моделями
    st.subheader("💾 Управление моделями")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Сохранить модели", key="save_models_btn"):
            if st.session_state.ml_predictor.is_trained:
                st.session_state.ml_predictor.save_models()
            else:
                st.error("❌ Нет обученных моделей для сохранения")
    
    with col2:
        if st.button("📂 Загрузить модели", key="load_models_btn"):
            st.session_state.ml_predictor.load_models()
            
            if st.session_state.ml_predictor.is_trained:
                st.success("✅ Модели загружены!")
                
                # Информация о загруженных моделях
                st.subheader("📋 Загруженные модели")
                model_names = list(st.session_state.ml_predictor.models.keys())
                for name in model_names:
                    st.write(f"✅ {name}")

def show_ml_optimization_tab():
    st.header("⚡ ML Оптимизация в реальном времени")
    
    st.markdown("""
    **Быстрая оптимизация с использованием обученных ML моделей**
    - Предсказание производительности без запуска симуляции
    - Оптимизация в 10-100 раз быстрее физической модели
    - Идеально для интерактивной настройки параметров
    """)
    
    if not st.session_state.ml_predictor.is_trained:
        st.warning("⚠️ Сначала обучите ML модели во вкладке 'Обучение ML'")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Параметры оптимизации")
        
        target_metric = st.selectbox(
            "Целевая метрика:",
            ["Устойчивость", "Отзывчивость", "Безопасность"],
            key="ml_opt_target"
        )
        
        max_iter = st.slider(
            "Максимальное количество итераций:",
            5, 100, 30,
            key="ml_opt_iter"
        )
        
        # Текущие параметры для быстрого предсказания
        st.subheader("🔮 Быстрое предсказание")
        
        cf_current = st.slider("C_f (кН/рад)", 40, 150, 80, key="pred_cf") * 1000
        cr_current = st.slider("C_r (кН/рад)", 40, 150, 100, key="pred_cr") * 1000
        cog_current = st.slider("a/L", 0.3, 0.7, 0.47, key="pred_cog")
        speed_current = st.slider("Скорость (м/с)", 5, 40, 20, key="pred_speed")
        
        if st.button("🔮 Предсказать производительность", key="predict_btn"):
            features = [1200, cf_current, cr_current, cog_current, speed_current, 0.1, 1]
            predictions = st.session_state.ml_predictor.predict_performance(features)
            
            if predictions:
                st.success("✅ Предсказания получены!")
                
                # Отображение предсказаний
                for model_name, pred in predictions.items():
                    with st.expander(f"📊 {model_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Макс. β", f"{np.degrees(pred['max_beta']):.2f}°")
                            st.metric("Макс. r", f"{np.degrees(pred['max_r']):.2f}°/с")
                        with col2:
                            st.metric("Смещение Y", f"{pred['final_y']:.2f} м")
                            st.metric("Время уст.", f"{pred['settling_time']:.2f} с")
    
    with col2:
        st.subheader("ML Оптимизация")
        
        if st.button("⚡ Запустить ML оптимизацию", key="ml_opt_btn"):
            with st.spinner("ML оптимизация..."):
                metric_map = {
                    "Устойчивость": "stability",
                    "Отзывчивость": "responsiveness", 
                    "Безопасность": "safety"
                }
                
                result = st.session_state.realtime_optimizer.optimize_with_ml(
                    metric_map[target_metric], max_iter
                )
                
                if result and result.success:
                    st.success("✅ ML оптимизация завершена!")
                    
                    # Результаты
                    st.subheader("🎯 Результаты ML оптимизации")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("C_f", f"{result.x[0]:.0f} Н/рад", key="ml_opt_cf")
                    with col2:
                        st.metric("C_r", f"{result.x[1]:.0f} Н/рад", key="ml_opt_cr")
                    with col3:
                        st.metric("a/L", f"{result.x[2]:.3f}", key="ml_opt_cog")
                    with col4:
                        st.metric("Скорость", f"{result.x[3]:.1f} м/с", key="ml_opt_speed")
                    
                    st.metric("Целевая функция", f"{result.fun:.6f}", key="ml_opt_metric")
                    
                    # Сравнение с физической моделью
                    st.subheader("🔍 Проверка точности")
                    
                    features = [1200, result.x[0], result.x[1], result.x[2], result.x[3], 0.1, 1]
                    ml_predictions = st.session_state.ml_predictor.predict_performance(features)
                    
                    fig, actual_metrics = st.session_state.predictive_analytics.create_prediction_comparison(
                        features, ml_predictions
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="ml_accuracy_chart")
                        
                        if actual_metrics:
                            st.info("""
                            **Точность ML предсказаний:**
                            - ML модели обеспечивают быстрое предсказание
                            - Точность зависит от качества обучения
                            - Идеально для интерактивной оптимизации
                            """)

def show_analytics_tab():
    st.header("📊 Предиктивная аналитика")
    
    st.markdown("""
    **Углубленный анализ данных и предсказаний**
    - Анализ важности признаков
    - Визуализация предсказаний
    - 3D поверхности производительности
    - Статистический анализ
    """)
    
    if not st.session_state.ml_predictor.is_trained:
        st.warning("⚠️ Сначала обучите ML модели во вкладке 'Обучение ML'")
        return
    
    # Анализ важности признаков
    st.subheader("🔍 Важность признаков")
    
    importance_fig = st.session_state.predictive_analytics.create_feature_importance_plot()
    if importance_fig:
        st.plotly_chart(importance_fig, use_container_width=True, key="feature_importance_chart")
        
        st.info("""
        **Интерпретация важности признаков:**
        - **C_f, C_r**: Жесткости шин - наиболее важные параметры
        - **speed**: Скорость сильно влияет на динамику
        - **cog_ratio**: Положение ЦТ важно для баланса
        - **mass**: Масса автомобиля
        """)
    
    # 3D поверхности производительности
    st.subheader("🌐 3D поверхности производительности")
    
    if st.button("🔄 Сгенерировать 3D поверхность", key="3d_surface_btn"):
        with st.spinner("Создание 3D поверхности..."):
            surface_fig = st.session_state.predictive_analytics.create_performance_surface()
            
            if surface_fig:
                st.plotly_chart(surface_fig, use_container_width=True, key="3d_surface_chart")
                
                st.info("""
                **3D поверхность производительности:**
                - Показывает зависимость Max Beta от жесткостей шин
                - Помогает визуализировать оптимальные области
                - Полезно для понимания взаимодействия параметров
                """)
    
    # Статистический анализ
    st.subheader("📈 Статистический анализ")
    
    if st.session_state.ml_predictor.training_data is not None:
        # Корреляционная матрица
        st.write("**Корреляционная матрица признаков:**")
        
        df_features = pd.DataFrame(
            st.session_state.ml_predictor.training_data['X'],
            columns=st.session_state.ml_predictor.training_data['feature_names']
        )
        
        # Добавляем целевые переменные
        for i, target_name in enumerate(st.session_state.ml_predictor.training_data['target_names']):
            df_features[target_name] = st.session_state.ml_predictor.training_data['y'][:, i]
        
        corr_matrix = df_features.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title="Матрица корреляций",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True, key="correlation_chart")

def show_simulation_tab():
    st.header("🚗 Сравнительная симуляция")
    
    st.markdown("""
    **Сравнение ML предсказаний с физической симуляцией**
    - Проверка точности ML моделей
    - Визуализация различий в предсказаниях
    - Анализ ошибок предсказания
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Параметры симуляции")
        
        mass = st.slider("Масса (кг)", 800, 2000, 1200, key="sim_mass_7")
        cf = st.slider("C_f (кН/рад)", 40, 150, 80, key="sim_cf_7") * 1000
        cr = st.slider("C_r (кН/рад)", 40, 150, 100, key="sim_cr_7") * 1000
        cog_ratio = st.slider("a/L", 0.3, 0.7, 0.47, key="sim_cog_7")
        speed = st.slider("Скорость (м/с)", 5, 40, 20, key="sim_speed_7")
        
        maneuver = st.selectbox(
            "Маневр:",
            ["Шаг рулем 10°", "Переставка"],
            key="sim_maneuver_7"
        )
    
    with col2:
        st.subheader("ML Предсказания")
        
        if st.button("🔮 Сравнить с ML", key="compare_ml_btn"):
            # ML предсказания
            features = [mass, cf, cr, cog_ratio, speed, 0.1, 1]
            ml_predictions = st.session_state.ml_predictor.predict_performance(features)
            
            # Физическая симуляция
            params = VehicleParameters()
            params.m = mass
            params.C_f = cf
            params.C_r = cr
            params.a = cog_ratio * params.L
            params.b = params.L - params.a
            params.V = speed
            
            if maneuver == "Шаг рулем 10°":
                maneuver_func = lambda t: step_steer(t, np.radians(10), 1.0)
            else:
                maneuver_func = lambda t: sine_steer(t, 0.05, 0.5, 1.0)
            
            simulator = create_nonlinear_simulator(params, cf, cr, 'linear')
            solution = simulator.simulate(maneuver_func, speed, (0, 10))
            
            if solution.success:
                results = get_simulation_results(solution)
                actual_metrics = {
                    'max_beta': np.max(np.abs(results['beta'])),
                    'max_r': np.max(np.abs(results['angular_velocity'])),
                    'final_y': results['Y'][-1],
                    'settling_time': st.session_state.predictive_analytics._calculate_settling_time(
                        results['beta'], results['time']
                    )
                }
                
                # Отображение сравнения
                st.subheader("📊 Сравнение результатов")
                
                if ml_predictions:
                    # Таблица сравнения
                    comparison_data = []
                    for model_name, pred in ml_predictions.items():
                        for metric in ['max_beta', 'max_r', 'final_y', 'settling_time']:
                            ml_value = pred[metric]
                            actual_value = actual_metrics[metric]
                            error_pct = abs(ml_value - actual_value) / actual_value * 100
                            
                            comparison_data.append({
                                'Модель': model_name,
                                'Метрика': metric,
                                'ML Предсказание': ml_value,
                                'Реальное значение': actual_value,
                                'Ошибка %': error_pct
                            })
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison)
                    
                    # Визуализация траектории
                    st.subheader("📈 Визуализация симуляции")
                    
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Траектория', 'Угол скольжения'))
                    
                    fig.add_trace(
                        go.Scatter(x=results['X'], y=results['Y'], name='Траектория',
                                  line=dict(color='blue', width=3)),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=results['time'], y=np.degrees(results['beta']), name='β',
                                  line=dict(color='green', width=2)),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="simulation_viz_chart")
                
                else:
                    st.error("❌ Нет ML предсказаний для сравнения")
            
            else:
                st.error("❌ Ошибка физической симуляции")

if __name__ == "__main__":
    main()