# check_imports.py
"""
Скрипт для проверки импортов
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def check_imports():
    """Проверить все импорты"""
    print("🔍 Проверка импортов...")
    
    try:
        from src.vehicle_parameters import VehicleParameters
        print("✅ VehicleParameters - OK")
    except ImportError as e:
        print(f"❌ VehicleParameters - {e}")
    
    try:
        from src.tire_models import PacejkaTireModel
        print("✅ PacejkaTireModel - OK")
    except ImportError as e:
        print(f"❌ PacejkaTireModel - {e}")
    
    try:
        from src.nonlinear_bicycle_model import create_nonlinear_simulator
        print("✅ create_nonlinear_simulator - OK")
    except ImportError as e:
        print(f"❌ create_nonlinear_simulator - {e}")
    
    try:
        from src.maneuvers import StandardManeuvers
        print("✅ StandardManeuvers - OK")
    except ImportError as e:
        print(f"❌ StandardManeuvers - {e}")

if __name__ == "__main__":
    check_imports()
