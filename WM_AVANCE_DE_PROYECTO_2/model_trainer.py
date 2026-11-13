import os
import joblib 
#joblib is a Python library (included by default in Anaconda and used internally by scikit-learn)
#designed for fast serialization (saving and loading) of Python objects that contain large numerical arrays,
#such as trained Machine Learning models.
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """Módulo 5: Entrenamiento de modelos de clasificación de sentimiento y clustering de tópicos."""

    def __init__(self, models_folder: str = "models"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, models_folder)
        os.makedirs(self.models_dir, exist_ok=True)

        # Vectorizador TF-IDF con unigramas y bigramas (ej. "no deja", "cobro doble")
        self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    def train_sentiment_model(self, df: pd.DataFrame) -> dict:
        """Entrena un modelo supervisado (Logistic Regression) para clasificar sentimiento."""
        df_valid = df.dropna(subset=["clean_text", "rating_category"]).copy()
        df_valid = df_valid[df_valid["clean_text"].str.strip() != ""]

        X = df_valid["clean_text"]
        y = df_valid["rating_category"]

        # División estratificada: 80% entrenamiento, 20% prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Ajustar TF-IDF solo con datos de entrenamiento para evitar fuga de datos (data leakage)
        X_train_vec = self.tfidf.fit_transform(X_train)
        X_test_vec = self.tfidf.transform(X_test)

        # Entrenar clasificador
        self.classifier.fit(X_train_vec, y_train)

        # Guardar artefactos del modelo entrenado
        joblib.dump(self.tfidf, os.path.join(self.models_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(self.classifier, os.path.join(self.models_dir, "sentiment_logreg.joblib"))
        print(f"Modelo de sentimiento y vectorizador guardados en: {self.models_dir}")

        return {
            "X_test_text": X_test,
            "X_test_vec": X_test_vec,
            "y_test": y_test,
            "y_pred": self.classifier.predict(X_test_vec),
        }

    def train_topic_model(self, df: pd.DataFrame, n_topics: int = 4, n_top_words: int = 6) -> list:
        """Entrena un modelo no supervisado (LDA) sobre reseñas insatisfechas para descubrir tópicos."""
        neg_reviews = df[df["rating"] <= 2]["clean_text"].dropna()
        neg_reviews = neg_reviews[neg_reviews.str.strip() != ""]

        if len(neg_reviews) < 10:
            print("Advertencia: Muy pocas reseñas negativas para extraer tópicos.")
            return []

        # LDA trabaja mejor con conteos de palabras crudas (CountVectorizer)
        extra_stopwords = ["te", "me", "se", "le", "lo", "la", "walmart", "app", "aplicacion"]
        count_vec = CountVectorizer(max_features=800, max_df=0.90, min_df=2, stop_words = extra_stopwords,)
        dtm = count_vec.fit_transform(neg_reviews)

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)

        joblib.dump(count_vec, os.path.join(self.models_dir, "topic_vectorizer.joblib"))
        joblib.dump(lda, os.path.join(self.models_dir, "lda_topics.joblib"))

        feature_names = count_vec.get_feature_names_out()
        topics_summary = []

        print("\n--- TÓPICOS DETECTADOS EN QUEJAS (APRENDIZAJE NO SUPERVISADO - LDA) ---")
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[: -n_top_words - 1 : -1]
            top_words = [feature_names[i] for i in top_indices]
            topic_str = ", ".join(top_words)
            topics_summary.append({"topic_id": topic_idx + 1, "keywords": topic_str})
            print(f"Tópico {topic_idx + 1}: {topic_str}")

        return topics_summary


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_csv_path = os.path.join(current_dir, "data", "processed", "walmart_reviews_processed.csv")

    if os.path.exists(processed_csv_path):
        df_proc = pd.read_csv(processed_csv_path)
        trainer = ModelTrainer(models_folder="models")

        # 1. Entrenar modelo supervisado de sentimiento
        eval_bundle = trainer.train_sentiment_model(df_proc)

        # 2. Entrenar modelo no supervisado de tópicos sobre reseñas negativas
        trainer.train_topic_model(df_proc, n_topics=4, n_top_words=6)
    else:
        print(f"No se encontró el archivo procesado en: {processed_csv_path}")
        print("Ejecuta primero text_processor.py.")