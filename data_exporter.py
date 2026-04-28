import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.io as pio

class DataExporter:
    """Класс для экспорта результатов симуляции"""
    
    def __init__(self, export_dir="exports"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_to_csv(self, results, params, maneuver_name, filename=None):
        """
        Экспорт результатов в CSV
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{timestamp}.csv"
        
        # Создание DataFrame с результатами
        data = {
            'time': results['time'],
            'beta_deg': np.degrees(results['beta']),
            'angular_velocity_deg_s': np.degrees(results['angular_velocity']),
            'yaw_angle_deg': np.degrees(results['yaw_angle']),
            'X': results['X'],
            'Y': results['Y']
        }
        
        df = pd.DataFrame(data)
        
        # Добавление метаданных
        metadata = {
            'maneuver': maneuver_name,
            'mass_kg': params['mass'],
            'speed_m_s': params['speed'],
            'cog_ratio': params['cog_ratio'],
            'cf_N_rad': params['cf'],
            'cr_N_rad': params['cr'],
            'tire_model': params['tire_model'],
            'export_time': datetime.now().isoformat()
        }
        
        # Сохранение
        filepath = os.path.join(self.export_dir, filename)
        df.to_csv(filepath, index=False)
        
        # Сохранение метаданных в отдельный файл
        metadata_file = filepath.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def export_to_excel(self, results, params, maneuver_name, energy_metrics=None, filename=None):
        """
        Экспорт в Excel с несколькими листами
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{timestamp}.xlsx"
        
        filepath = os.path.join(self.export_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Лист с основными данными
            main_data = pd.DataFrame({
                'time': results['time'],
                'beta_deg': np.degrees(results['beta']),
                'angular_velocity_deg_s': np.degrees(results['angular_velocity']),
                'yaw_angle_deg': np.degrees(results['yaw_angle']),
                'X': results['X'],
                'Y': results['Y']
            })
            main_data.to_excel(writer, sheet_name='Основные данные', index=False)
            
            # Лист с параметрами
            params_data = pd.DataFrame([{
                'Маневр': maneuver_name,
                'Масса, кг': params['mass'],
                'Скорость, м/с': params['speed'],
                'Положение ЦТ': params['cog_ratio'],
                'Жесткость перед. шин, Н/рад': params['cf'],
                'Жесткость зад. шин, Н/рад': params['cr'],
                'Модель шин': params['tire_model']
            }])
            params_data.to_excel(writer, sheet_name='Параметры', index=False)
            
            # Лист с энергетическими метриками
            if energy_metrics:
                energy_data = pd.DataFrame([energy_metrics])
                energy_data.to_excel(writer, sheet_name='Энергетика', index=False)
        
        return filepath
    
    def export_plot(self, fig, plot_name, filename=None):
        """
        Экспорт графика в файл
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{plot_name}_{timestamp}.html"
        
        filepath = os.path.join(self.export_dir, filename)
        pio.write_html(fig, filepath)
        
        return filepath
    
    def generate_report(self, results, params, maneuver_name, energy_metrics, plots):
        """
        Генерация полного отчета
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.export_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        exported_files = []
        
        # Экспорт данных
        csv_file = self.export_to_csv(results, params, maneuver_name, 
                                    f"data_{timestamp}.csv")
        excel_file = self.export_to_excel(results, params, maneuver_name, 
                                        energy_metrics, f"report_{timestamp}.xlsx")
        
        exported_files.extend([csv_file, excel_file])
        
        # Экспорт графиков
        for plot_name, fig in plots.items():
            plot_file = self.export_plot(fig, plot_name, f"{plot_name}_{timestamp}.html")
            exported_files.append(plot_file)
        
        # Создание индексного файла
        self._create_index_file(report_dir, exported_files, params, maneuver_name)
        
        return report_dir, exported_files
    
    def _create_index_file(self, report_dir, files, params, maneuver_name):
        """Создание HTML индексного файла"""
        index_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Отчет симуляции</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .files {{ margin: 20px 0; }}
                .file-item {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Отчет симуляции управляемости</h1>
                <p><strong>Маневр:</strong> {maneuver_name}</p>
                <p><strong>Масса:</strong> {params['mass']} кг</p>
                <p><strong>Скорость:</strong> {params['speed']} м/с</p>
                <p><strong>Модель шин:</strong> {params['tire_model']}</p>
                <p><strong>Время генерации:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="files">
                <h2>📁 Экспортированные файлы:</h2>
                {"".join(f'<div class="file-item">📄 {os.path.basename(f)}</div>' for f in files)}
            </div>
        </body>
        </html>
        """
        
        index_file = os.path.join(report_dir, "index.html")
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        return index_file
