# examples/week4_demo.py
"""
Демонстрация веб-интерфейса для недели 4
Запуск: streamlit run week4_demo.py
"""

import sys
import os

# Добавляем путь к корневой директории проекта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Запуск веб-приложения через прямой импорт"""
    try:
        from src.web_interface.app import StreamlitSimulator
        
        print("🚀 Запуск веб-симулятора управляемости...")
        print("📱 Откройте браузер и перейдите по адресу: http://localhost:8501")
        
        app = StreamlitSimulator()
        app.run()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("📋 Проверьте структуру проекта и пути импорта")
        print("💡 Убедитесь, что все файлы находятся в правильных директориях")

if __name__ == "__main__":
    main()
    