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


# Создание Flask-приложения
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Создание Spark-сессии
spark = SparkSession.builder \
    .appName("BigDataApp") \
    .config("spark.eventLog.gcMetrics.enabled", "false") \
    .getOrCreate()


@app.route('/', methods=['GET', 'POST'])
def index():
    show_form = True
    cm_plot_path = None
    accuracy = None
    schema_info = None
    preview_html = None
    correlation_plot_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            df = spark.read.csv(file_path, header=True, inferSchema=True)
            df = df.dropna()

            # Разделяем признаки
            numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
            categorical_cols = [f.name for f in df.schema.fields if
                                isinstance(f.dataType, StringType)]

            if len(numeric_cols + categorical_cols) < 2:
                return "Недостаточно данных для обучения модели"

            target_col = numeric_cols[-1] if numeric_cols else categorical_cols[-1]
            feature_cols = [col for col in numeric_cols + categorical_cols if col != target_col]

            # Обзор данных
            schema_info = df._jdf.schema().treeString()

            preview_html = df.limit(10).toPandas().to_html(classes="data", index=False)

            # ======= ОБУЧЕНИЕ МОДЕЛИ ========
            # Indexers для категориальных признаков и целевой переменной
            indexers = [
                StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid='skip') for
                col in categorical_cols]
            if target_col in categorical_cols:
                indexers.append(
                    StringIndexer(inputCol=target_col, outputCol="label", handleInvalid='skip'))
            else:
                df = df.withColumnRenamed(target_col, "label")

            # Финальные признаки для VectorAssembler
            final_features = [col + "_indexed" if col in categorical_cols else col for col in
                              feature_cols]

            assembler = VectorAssembler(inputCols=final_features, outputCol="features")
            lr = LogisticRegression(featuresCol="features", labelCol="label")

            pipeline = Pipeline(stages=indexers + [assembler, lr])
            model = pipeline.fit(df)

            show_form = False

            # Оценка
            predictions = model.transform(df)
            evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                                          predictionCol="prediction",
                                                          metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)

            # Получаем предсказания и истинные значения
            preds = predictions.select("label", "prediction").dropna().toPandas()

            # Confusion matrix
            if preds["label"].nunique() <= 10:
                cm = confusion_matrix(preds["label"], preds["prediction"])

                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Предсказание")
                plt.ylabel("Истина")
                plt.title("Confusion Matrix")
                cm_plot_path = os.path.join(PLOT_FOLDER, 'confusion_matrix.png')
                plt.savefig(cm_plot_path)
                plt.close()
            else:
                cm_plot_path = None

            # Correlation matrix
            numeric_df = df.select(
                [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)])
            if len(numeric_df.columns) >= 2:
                corr_pdf = numeric_df.toPandas().corr()

                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_pdf, annot=True, fmt=".2f", cmap="coolwarm", square=True)
                plt.title("Корреляционная матрица")
                correlation_plot_path = os.path.join(PLOT_FOLDER, 'correlation_matrix.png')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(correlation_plot_path)
                plt.close()

    return render_template('index.html',
                           show_form=show_form,
                           schema_info=schema_info,
                           preview_table=preview_html,
                           accuracy=round(accuracy, 4) if accuracy else None,
                           cm_plot_url=cm_plot_path,
                           corr_plot_url=correlation_plot_path)


if __name__ == '__main__':
    app.run(debug=True)