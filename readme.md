<h1>Веб-приложения на Python и фреймворке flask для ресчета результативности сотрудников с использованием машинного обучения</h1>

Модель машинного обучения: XGBRegressor (Extreme Gradient Boosting Regressor) - это модель машинного обучения, основанная на алгоритме градиентного бустинга. Она представляет собой регрессионную версию алгоритма XGBoost (Extreme Gradient Boosting), который является одним из наиболее эффективных ансамблевых методов машинного обучения.

<h2>Наборы данных используемые и измененные для работы программы:</h2>
https://www.kaggle.com/datasets/gauravduttakiit/employee-performance-prediction?select=test_dataset.csv
https://www.kaggle.com/datasets/gauravduttakiit/employee-performance-prediction?select=train_dataset.csv

<h2>Используемые технологии:</h2>

Flask: Фреймворк для создания веб-приложений на Python.
pandas: Библиотека для работы с данными в формате DataFrame.
xgboost: Библиотека для машинного обучения, использующая градиентный бустинг.
Bootstrap: Фреймворк для стилизации веб-приложений.

<h2>Краткое описание работы программы:</h2>

Обучение модели:

Обучается модель XGBoost на предварительно загруженном обучающем датасете (train_dataset.csv).
Вычисляются показатели точности модели (accuracy, R Squared, MSE) и время обучения.
Веб-интерфейс:

Разрабатывается веб-интерфейс с использованием HTML и Bootstrap, включая навигационное меню и страницу с показателями модели (model_metrics.html).
Реализованы маршруты для отображения основной страницы (index), скачивания примеров документов (download_example, download_example_test_generating_document), отображения показателей модели (model_metrics) и обработки загрузки файла для расчета результативности (calculate).
Вычисление результатов:

Пользователь может загрузить тестовый датасет через веб-интерфейс.
Модель используется для предсказания результативности на основе загруженных данных.
Вычисляется время выполнения расчета.
Отображение результатов:

Результаты предсказаний сохраняются в двух версиях: полной и короткой.
Пользователю предоставляется возможность скачать полные и сокращенные результаты в формате CSV.
Результаты отображаются на веб-странице (result.html), и пользователю предоставляется возможность сортировать их по разным параметрам.
Локальный запуск:

Flask-приложение запускается локально с поддержкой отладки (app.run(debug=True)).
Веб-приложение становится доступным по локальному адресу, и пользователь может взаимодействовать с интерфейсом через веб-браузер.
