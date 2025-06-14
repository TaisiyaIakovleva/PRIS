# PRIS - Работа с Big Data

Это Flask-приложение, которое использует Apache Spark для обработки данных и построения модели логистической регрессии. Приложение позволяет загружать CSV-файлы, анализировать данные, строить модели машинного обучения и визуализировать результаты.

Основные компоненты
1. Импорты
```python
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from flask import Flask, request, render_template
from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType, StringType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```
3. Настройка Flask и Spark
2.1 Создается Flask-приложение
2.2 Определяются папки для загрузки файлов (uploads) и сохранения графиков (static/plots)
2.3 Инициализируется Spark-сессия с настройками:
```   
spark = SparkSession.builder \
    .appName("BigDataApp") \
    .config("spark.eventLog.gcMetrics.enabled", "false") \
    .getOrCreate()3. Загрузка и обработка данных
```
При получении CSV-файла:
```python
file = request.files['file']
file_path = os.path.join(UPLOAD_FOLDER, file.filename)
file.save(file_path)

# Чтение данных в Spark DataFrame
df = spark.read.csv(file_path, header=True, inferSchema=True)
df = df.dropna()  # Удаление пропущенных значений
```
4. Анализ структуры данных
```python
# Получение информации о типах столбцов
numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
categorical_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
# Определение целевой переменной
target_col = numeric_cols[-1] if numeric_cols else categorical_cols[-1]
feature_cols = [col for col in numeric_cols + categorical_cols if col != target_col]
# Генерация preview данных
preview_html = df.limit(10).toPandas().to_html(classes="data", index=False)
```
5. Построение модели
```python
# Преобразование категориальных признаков
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid='skip') 
    for col in categorical_cols
]

# Подготовка финальных признаков
final_features = [col + "_indexed" if col in categorical_cols else col for col in feature_cols]

# Создание pipeline
assembler = VectorAssembler(inputCols=final_features, outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=indexers + [assembler, lr])

# Обучение модели
model = pipeline.fit(df)
predictions = model.transform(df)

# Оценка точности
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
```
6. Визуализации

Матрица ошибок:
```python
preds = predictions.select("label", "prediction").dropna().toPandas()
cm = confusion_matrix(preds["label"], preds["prediction"])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(PLOT_FOLDER, 'confusion_matrix.png'))
plt.close()
```
Корреляционная матрица:
```python
numeric_df = df.select([f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)])
corr_pdf = numeric_df.toPandas().corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_pdf, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, 'correlation_matrix.png'))
plt.close()
```
7. Отображение результатов
```python
return render_template('index.html',
                     show_form=False,
                     schema_info=df._jdf.schema().treeString(),
                     preview_table=preview_html,
                     accuracy=round(accuracy, 4),
                     cm_plot_url='static/plots/confusion_matrix.png',
                     corr_plot_url='static/plots/correlation_matrix.png')
```

Полный цикл работы
1) Пользователь загружает CSV-файл через веб-интерфейс
2) Приложение:
- Сохраняет файл
- Загружает данные в Spark
- Анализирует структуру
- Строит модель
- Генерирует визуализации
4) Результаты отображаются на странице:
- Информация о данных
- Превью таблицы
- Точность модели
- Графики анализа

Пример вызова для тестирования:
```python
import requests

files = {'file': open('dataset.csv', 'rb')}
response = requests.post('http://localhost:5000', files=files)
print(response.text)
```
