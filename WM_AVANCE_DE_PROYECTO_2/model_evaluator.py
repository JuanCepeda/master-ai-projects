import json
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


class ModelEvaluator:
    """Módulo 6: Evaluación cuantitativa de modelos de clasificación y modelado de tópicos."""

    def __init__(self, models_folder: str = "models", reports_folder: str = "reports"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, models_folder)
        self.reports_dir = os.path.join(self.base_dir, reports_folder)
        self.figures_dir = os.path.join(self.reports_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_artifacts(self) -> tuple:
        """Carga los modelos y vectorizadores previamente entrenados con joblib."""
        tfidf_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        clf_path = os.path.join(self.models_dir, "sentiment_logreg.joblib")
        topic_vec_path = os.path.join(self.models_dir, "topic_vectorizer.joblib")
        lda_path = os.path.join(self.models_dir, "lda_topics.joblib")

        tfidf = joblib.load(tfidf_path)
        classifier = joblib.load(clf_path)

        topic_vec = joblib.load(topic_vec_path) if os.path.exists(topic_vec_path) else None
        lda = joblib.load(lda_path) if os.path.exists(lda_path) else None

        return tfidf, classifier, topic_vec, lda

    def evaluate_sentiment_classifier(self, df: pd.DataFrame) -> dict:
        """Evalúa el clasificador sobre el conjunto de prueba (20%) y guarda métricas + matriz de confusión."""
        tfidf, classifier, _, _ = self.load_artifacts()

        df_valid = df.dropna(subset=["clean_text", "rating_category"]).copy()
        df_valid = df_valid[df_valid["clean_text"].str.strip() != ""]

        X = df_valid["clean_text"]
        y = df_valid["rating_category"]

        # Reproducir exactamente la misma partición de prueba (random_state=42)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Transformar únicamente con el vocabulario entrenado
        X_test_vec = tfidf.transform(X_test)
        y_pred = classifier.predict(X_test_vec)

        # Cálculo de métricas principales
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "test_samples": int(len(y_test)),
        }

        print("\n" + "=" * 60)
        print("EVALUACIÓN DEL CLASIFICADOR DE SENTIMIENTO (MÓDULO 6)")
        print("=" * 60)
        print(f"Muestras evaluadas (Test 20%): {metrics['test_samples']}")
        print(f"Exactitud (Accuracy):          {metrics['accuracy']:.2%}")
        print(f"Precisión Ponderada:           {metrics['precision_weighted']:.2%}")
        print(f"Sensibilidad (Recall):         {metrics['recall_weighted']:.2%}")
        print(f"F1-Score Ponderado:            {metrics['f1_weighted']:.2%}")
        print("\nReporte detallado por categoría:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Guardar matriz de confusión
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        self._plot_confusion_matrix(cm, labels)

        # Guardar métricas en JSON para trazabilidad y uso en reportes
        metrics_path = os.path.join(self.reports_dir, "evaluation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"Métricas guardadas en: {metrics_path}")

        return metrics

    def _plot_confusion_matrix(self, cm, labels: list) -> str:
        """Genera y guarda la visualización de la matriz de confusión."""
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
        )
        plt.title("Matriz de Confusión - Clasificación de Sentimiento", fontsize=13, fontweight="bold")
        plt.xlabel("Predicción del Modelo", fontsize=11)
        plt.ylabel("Etiqueta Real (Estrellas)", fontsize=11)
        plt.xticks(rotation=15)
        plt.yticks(rotation=0)
        plt.tight_layout()

        file_path = os.path.join(self.figures_dir, "4_matriz_confusion.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Matriz de confusión guardada en: {file_path}")
        return file_path

    def evaluate_topic_coherence_proxy(self, df: pd.DataFrame) -> dict:
        """Evalúa la perplejidad del modelo LDA sobre las reseñas negativas."""
        _, _, topic_vec, lda = self.load_artifacts()
        if topic_vec is None or lda is None:
            return {}

        neg_reviews = df[df["rating"] <= 2]["clean_text"].dropna()
        neg_reviews = neg_reviews[neg_reviews.str.strip() != ""]
        dtm = topic_vec.transform(neg_reviews)

        perplexity = round(float(lda.perplexity(dtm)), 2)
        print(f"\nPerplejidad del modelo LDA (menor indica mejor ajuste): {perplexity}")
        return {"lda_perplexity": perplexity}


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_csv_path = os.path.join(current_dir, "data", "processed", "walmart_reviews_processed.csv")

    if os.path.exists(processed_csv_path):
        df_proc = pd.read_csv(processed_csv_path)
        evaluator = ModelEvaluator(models_folder="models", reports_folder="reports")

        # 1. Evaluar clasificador supervisado y generar matriz de confusión
        evaluator.evaluate_sentiment_classifier(df_proc)

        # 2. Evaluar ajuste del modelo no supervisado LDA
        evaluator.evaluate_topic_coherence_proxy(df_proc)
    else:
        print(f"No se encontró el archivo procesado en: {processed_csv_path}")
        print("Ejecuta primero text_processor.py y model_trainer.py.")