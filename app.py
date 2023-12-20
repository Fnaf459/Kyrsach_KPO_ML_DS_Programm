from flask import Flask, render_template, request, send_file
import pandas as pd
import xgboost as xgb
from io import BytesIO
from sklearn.metrics import mean_squared_error, r2_score
import time

app = Flask(__name__)

# Загружаем обучающий датасет
df_train = pd.read_csv("D:/PythonProject/Kyrsach_Savenkov_KPO/train_dataset.csv")

# Заполняем пропущенные значения в столбце "wip" средними значениями
df_train["wip"] = df_train["wip"].fillna(df_train["wip"].mean())

# Выделяем целевую переменную
y_train = df_train["actual_productivity"]

# Выделяем признаки для обучения модели
X_train = df_train.drop(["actual_productivity", "name"], axis=1)

# Обучаем модель XGBoost и расчет времени обучения модели
start_time_train = time.time()
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
end_time_train = time.time()

# Вычисляем показатели точности модели на обучающем наборе
y_train_pred = model.predict(X_train)
accuracy = model.score(X_train, y_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Рассчитываем время обучения
training_time = end_time_train - start_time_train

# Добавим переменные для сохранения полной и короткой версий результата
result_csv_full = BytesIO()
result_csv_short = BytesIO()

result_full = pd.DataFrame()  # Инициализируем как пустой DataFrame
end_time_calculation = 0
start_time_calculation = 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download_example")
def download_example():
    return send_file("static/test_dataset.csv", as_attachment=True)

@app.route("/download_example_test_generating_document")
def download_example_test_generating_document():
    return send_file("static/Example_Test_Generating_Document.docx", as_attachment=True)

# Новый маршрут для страницы с показателями модели
@app.route("/model_metrics")
def model_metrics():
    return render_template("model_metrics.html", accuracy=accuracy, r2_train=r2_train, mse_train=mse_train, training_time=training_time)

@app.route("/calculate", methods=["POST"])
def calculate():
    global result_full, end_time_calculation, start_time_calculation
    if request.method == "POST":
        # Получаем загруженный файл
        uploaded_file = request.files["file"]

        # Читаем данные из CSV
        df_test = pd.read_csv(uploaded_file)

        # Засекаем время выполнения расчета результативности
        start_time_calculation = time.time()

        # Предсказываем значения
        predictions = model.predict(df_test.drop("name", axis=1))

        # Замеряем время выполнения расчета
        end_time_calculation = time.time()

        # Добавляем предсказания в новый сокращенный датафрейм
        result_df = pd.DataFrame(predictions, columns=["actual_productivity"])
        result_df.insert(loc=result_df.columns.get_loc("actual_productivity"), column="name", value=df_test["name"])

        # Сохраняем сокращенный результат в CSV-файл
        result_df.to_csv(result_csv_short, index=False, encoding='utf-8')
        result_csv_short.seek(0)

        # Добавляем предсказания в новый полный датафрейм
        result_full = pd.DataFrame(predictions, columns=["actual_productivity"])
        for col in df_test.columns:
            result_full.insert(loc=result_full.columns.get_loc("actual_productivity"), column=col, value=df_test[col])

        # Сохраняем полный результат в CSV-файл
        result_full.to_csv(result_csv_full, index=False, encoding='utf-8')
        result_csv_full.seek(0)

        # Передаем данные в шаблон result.html
        return render_template("result.html", result=result_full, calculation_time=end_time_calculation - start_time_calculation)

# Добавим два новых маршрута для скачивания полной и короткой версии файла
@app.route("/download_full_result")
def download_full_result():
    return send_file(result_csv_full, as_attachment=True, download_name="Ready_result_full.csv", mimetype='text/csv', last_modified=time.time())

@app.route("/download_short_result")
def download_short_result():
    return send_file(result_csv_short, as_attachment=True, download_name="Ready_result_short.csv", mimetype='text/csv', last_modified=time.time())

# Новый маршрут для сортировки результатов
@app.route("/sort_results")
def sort_results():
    global result_full, end_time_calculation, start_time_calculation
    # Получение параметров сортировки из URL
    column = request.args.get('column', 'name')
    order = request.args.get('order', 'asc')

    # Сортировка данных
    result_sorted = result_full.sort_values(by=column, ascending=(order.lower() == 'asc'))

    # Изменение порядка сортировки для следующего запроса
    next_order = 'desc' if order.lower() == 'asc' else 'asc'

    # Передача отсортированных данных в шаблон result.html
    return render_template("result.html", result=result_sorted, calculation_time=end_time_calculation - start_time_calculation, sort_order=next_order)

if __name__ == "__main__":
    app.run(debug=True)
