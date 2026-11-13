from datetime import datetime
import json
import os
import joblib
import pandas as pd


class ReportGenerator:
    """Módulo 7: Generación de reportes ejecutivos y salidas interpretables para toma de decisiones."""

    def __init__(self, models_folder: str = "models", reports_folder: str = "reports"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, models_folder)
        self.reports_dir = os.path.join(self.base_dir, reports_folder)
        os.makedirs(self.reports_dir, exist_ok=True)

    def _load_models_and_metrics(self) -> tuple:
        """Carga los modelos entrenados y el archivo de métricas del Módulo 6."""
        tfidf = joblib.load(os.path.join(self.models_dir, "tfidf_vectorizer.joblib"))
        classifier = joblib.load(os.path.join(self.models_dir, "sentiment_logreg.joblib"))

        topic_vec_path = os.path.join(self.models_dir, "topic_vectorizer.joblib")
        lda_path = os.path.join(self.models_dir, "lda_topics.joblib")
        topic_vec = joblib.load(topic_vec_path) if os.path.exists(topic_vec_path) else None
        lda = joblib.load(lda_path) if os.path.exists(lda_path) else None

        metrics_path = os.path.join(self.reports_dir, "evaluation_metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        return tfidf, classifier, topic_vec, lda, metrics

    def enrich_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica predicción de sentimiento y asignación de tópicos a todo el dataset."""
        tfidf, classifier, topic_vec, lda, _ = self._load_models_and_metrics()
        df_out = df.copy()

        valid_mask = df_out["clean_text"].notna() & (df_out["clean_text"].str.strip() != "")
        df_out["predicted_sentiment"] = "Sin clasificar"

        if valid_mask.any():
            X_vec = tfidf.transform(df_out.loc[valid_mask, "clean_text"])
            df_out.loc[valid_mask, "predicted_sentiment"] = classifier.predict(X_vec)

        # Detectar discrepancias: usuarios con 4-5 estrellas cuyo texto suena insatisfecho
        df_out["rating_text_mismatch"] = (df_out["rating"] >= 4) & (
            df_out["predicted_sentiment"] == "Insatisfecho (1-2)"
        )

        # Asignar tópico principal a reseñas negativas
        df_out["assigned_topic"] = "N/A (No negativa)"
        if topic_vec is not None and lda is not None:
            neg_mask = (df_out["rating"] <= 2) & valid_mask
            if neg_mask.any():
                dtm = topic_vec.transform(df_out.loc[neg_mask, "clean_text"])
                topic_probs = lda.transform(dtm)
                df_out.loc[neg_mask, "assigned_topic"] = [
                    f"Tópico {idx + 1}" for idx in topic_probs.argmax(axis=1)
                ]

        enriched_path = os.path.join(self.reports_dir, "walmart_reviews_enriched.csv")
        df_out.to_csv(enriched_path, index=False, encoding="utf-8-sig")
        print(f"Dataset enriquecido con predicciones guardado en: {enriched_path}")
        return df_out

    def generate_executive_report(self, df_enriched: pd.DataFrame) -> str:
        """Genera un reporte ejecutivo interpretable en formato Markdown para QA, UX y Producto."""
        _, _, topic_vec, lda, metrics = self._load_models_and_metrics()

        total_reviews = len(df_enriched)
        avg_rating = df_enriched["rating"].mean()
        neg_pct = (df_enriched["rating"] <= 2).mean() * 100
        mismatches = int(df_enriched["rating_text_mismatch"].sum())

        # Resumen de tópicos detectados
        topics_lines = []
        if topic_vec is not None and lda is not None:
            feature_names = topic_vec.get_feature_names_out()
            neg_df = df_enriched[df_enriched["rating"] <= 2]
            topic_counts = neg_df["assigned_topic"].value_counts()

            for idx, topic in enumerate(lda.components_):
                topic_label = f"Tópico {idx + 1}"
                top_words = ", ".join([feature_names[i] for i in topic.argsort()[:-7:-1]])
                count = int(topic_counts.get(topic_label, 0))
                share = (count / len(neg_df) * 100) if len(neg_df) > 0 else 0
                topics_lines.append(
                    f"| **{topic_label}** | {count} ({share:.1f}%) | `{top_words}` |"
                )

        # Alertas por versión más reciente (top 5 versiones con más reseñas)
        version_summary = []
        if "app_version" in df_enriched.columns:
            ver_stats = (
                df_enriched.groupby("app_version")
                .agg(total=("rating", "count"), avg_rating=("rating", "mean"))
                .sort_values("total", ascending=False)
                .head(5)
            )
            for ver, row in ver_stats.iterrows():
                status = "Alerta Crítica" if row["avg_rating"] < 2.5 else "Estable"
                version_summary.append(
                    f"| v{ver} | {int(row['total'])} | {row['avg_rating']:.2f} ★ | {status} |"
                )

        report_content = f"""# Reporte Ejecutivo de Experiencia de Usuario (UX/QA)
**Aplicación:** Walmart México (iOS - App ID: 823947897)  
**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  

---

## 1. Indicadores Clave de Desempeño (KPIs)
* **Total de reseñas analizadas:** {total_reviews}
* **Calificación promedio histórica:** {avg_rating:.2f} / 5.00 estrellas
* **Tasa de insatisfacción (1-2 estrellas):** {neg_pct:.1f}%
* **Alertas silenciosas (4-5 estrellas con texto de queja):** {mismatches} casos detectados

---

## 2. Desempeño del Modelo Predictivo (Validación Técnica)
* **Exactitud (Accuracy):** {metrics.get('accuracy', 0):.2%}
* **F1-Score Ponderado:** {metrics.get('f1_weighted', 0):.2%}
* **Muestras evaluadas en prueba (20%):** {metrics.get('test_samples', 0)}

---

## 3. Matriz de Priorización de Problemas (Tópicos LDA en Quejas)
Agrupamiento automático de reseñas insatisfechas para orientar acciones de Desarrollo y QA:

| Tópico Detectado | Volumen en Quejas | Palabras Clave Asociadas |
| :--- | :--- | :--- |
{chr(10).join(topics_lines) if topics_lines else "| Sin datos | 0 | N/A |"}

---

## 4. Monitoreo de Estabilidad por Versión de la App (Top 5)
| Versión | Reseñas | Calificación Promedio | Estado Operativo |
| :--- | :--- | :--- | :--- |
{chr(10).join(version_summary) if version_summary else "| N/A | 0 | N/A | Sin datos |"}

---

## 5. Recomendaciones Accionables para el Negocio
1. **Equipo de QA / Ingeniería:** Priorizar la revisión de los flujos asociados al **Tópico 1** y **Tópico 2**, ya que concentran el mayor volumen de fricción en usuarios de 1 y 2 estrellas.
2. **Equipo de Producto / UX:** Auditar los **{mismatches} casos de discrepancia** donde el usuario otorgó calificación alta pero el modelo detectó lenguaje de insatisfacción en el comentario.
3. **Monitoreo Continuo:** Utilizar este pipeline tras cada despliegue en App Store para detectar caídas de calificación por debajo de `2.5 ★` en las primeras 48 horas.
"""

        report_path = os.path.join(self.reports_dir, "executive_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"Reporte ejecutivo generado exitosamente en: {report_path}")
        return report_path


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_csv_path = os.path.join(current_dir, "data", "processed", "walmart_reviews_processed.csv")

    if os.path.exists(processed_csv_path):
        df_proc = pd.read_csv(processed_csv_path)
        reporter = ReportGenerator(models_folder="models", reports_folder="reports")

        # 1. Enriquecer dataset con predicciones y tópicos
        df_final = reporter.enrich_predictions(df_proc)

        # 2. Generar reporte ejecutivo interpretable en Markdown
        reporter.generate_executive_report(df_final)
    else:
        print(f"No se encontró el archivo procesado en: {processed_csv_path}")