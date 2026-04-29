# Vehicle Simulator

Интерактивный симулятор управляемости автомобиля на основе линейной и нелинейной модели "велосипед". Проект позволяет запускать стандартные маневры, сравнивать модели шин, анализировать устойчивость и визуализировать результаты в `matplotlib`, `plotly` и `streamlit`.

## Возможности

- линейная и нелинейная модели поперечной динамики автомобиля;
- несколько моделей шин: `Linear`, `PiecewiseLinear`, `SimplifiedFiala`, `Pacejka`;
- набор типовых маневров: шаг рулем, переставка, двойная переставка, постоянный радиус;
- интерактивный веб-интерфейс на `Streamlit`;
- экспорт результатов в `CSV`, `Excel` и `HTML`;
- базовые тесты на параметры, визуализацию, экспорт и анализ устойчивости.

## Структура проекта

```text
vehicle_simulator/
├── examples/              # демонстрационные сценарии
├── src/                   # исходный код симулятора
│   ├── analysis/          # анализ устойчивости
│   ├── export/            # экспорт данных
│   ├── visualization/     # графики и 3D-визуализация
│   └── web_interface/     # Streamlit-интерфейс
├── tests/                 # тесты
├── check_imports.py       # быстрая проверка импортов
├── pyproject.toml         # зависимости и конфигурация проекта
└── requirements.txt       # совместимый список зависимостей
```

## Требования

- Python `3.10+`
- `pip`

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Если нужен только минимальный запуск без dev-зависимостей:

```bash
pip install -r requirements.txt
```

## Быстрый старт

Проверка импортов:

```bash
python check_imports.py
```

Запуск веб-интерфейса:

```bash
streamlit run src/web_interface/app.py
```

Запуск примера:

```bash
python examples/week1_demo.py
```

Запуск тестов:

```bash
pytest
```

## Что внутри

Основные модули:

- [src/vehicle_parameters.py](/Users/anastasiakopaneva/Documents/Playground/vehicle_simulator/vehicle_simulator/src/vehicle_parameters.py) хранит параметры автомобиля;
- [src/linear_bicycle_model.py](/Users/anastasiakopaneva/Documents/Playground/vehicle_simulator/vehicle_simulator/src/linear_bicycle_model.py) реализует линейную модель;
- [src/nonlinear_bicycle_model.py](/Users/anastasiakopaneva/Documents/Playground/vehicle_simulator/vehicle_simulator/src/nonlinear_bicycle_model.py) содержит нелинейную модель и фабрику симулятора;
- [src/maneuvers.py](/Users/anastasiakopaneva/Documents/Playground/vehicle_simulator/vehicle_simulator/src/maneuvers.py) описывает стандартные тестовые маневры;
- [src/web_interface/app.py](/Users/anastasiakopaneva/Documents/Playground/vehicle_simulator/vehicle_simulator/src/web_interface/app.py) поднимает Streamlit-приложение.
