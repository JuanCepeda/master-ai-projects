import os
import re
import unicodedata
import pandas as pd


class TextProcessor:
    """Módulo 3: Preprocesamiento y normalización de texto de reseñas en español."""

    def __init__(self):
        # Lista base de palabras vacías (stopwords) en español sin depender de descargas externas
        self.spanish_stopwords = {
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
            "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
            "mas", "pero", "sus", "le", "ya", "o", "este", "si", "porque", "esta",
            "entre", "cuando", "muy", "sin", "sobre", "tambien", "me", "hasta",
            "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
            "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
            "e", "esto", "mi", "antes", "algunos", "que", "unos", "yo", "otro",
            "otras", "otra", "el", "tanto", "esa", "estos", "mucho", "quienes",
            "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas",
            "algo", "nosotros", "es", "son", "fue", "era", "app", "aplicacion",
            "te", "me", "se", "le", "lo", "la", "walmart", "app", "aplicacion",
            
        }

    def normalize_text(self, text: str) -> str:
        """Limpia y estandariza una cadena de texto individual."""
        if not isinstance(text, str):
            return ""

        # 1. Convertir a minúsculas
        text = text.lower()

        # 2. Eliminar acentos y diacríticos (ej. 'aplicación' -> 'aplicacion')
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

        # 3. Eliminar URLs, menciones y caracteres especiales/emojis (dejar solo letras y espacios)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)

        # 4. Tokenización básica y eliminación de stopwords y palabras de 1 letra
        tokens = [
            word for word in text.split()
            if word not in self.spanish_stopwords and len(word) > 1
        ]

        return " ".join(tokens)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica el preprocesamiento textual y genera variables derivadas para análisis y ML."""
        if df.empty:
            print("Advertencia: El DataFrame de entrada está vacío.")
            return df

        df_proc = df.copy()
        print("Iniciando preprocesamiento de texto y extracción de características básicas...")

        # Combinar título y cuerpo de la reseña para tener el contexto completo
        df_proc["full_text"] = (
            df_proc["title"].fillna("") + " " + df_proc["review_text"].fillna("")
        ).str.strip()

        # Aplicar normalización
        df_proc["clean_text"] = df_proc["full_text"].apply(self.normalize_text)

        # Variables numéricas derivadas útiles para exploración y modelos futuros
        df_proc["char_count"] = df_proc["review_text"].astype(str).str.len()
        df_proc["word_count"] = df_proc["clean_text"].apply(lambda x: len(x.split()))

        # Etiqueta preliminar basada en calificación (útil como referencia para etapas futuras)
        df_proc["rating_category"] = df_proc["rating"].apply(
            lambda r: "Insatisfecho (1-2)" if r <= 2 else ("Neutral (3)" if r == 3 else "Satisfecho (4-5)")
        )

        print("Preprocesamiento completado exitosamente.")
        return df_proc

    def save_processed_data(
        self,
        df: pd.DataFrame,
        file_name: str = "walmart_reviews_processed.csv",
        folder: str = "data/processed",
    ) -> str:
        """Guarda el dataset preprocesado listo para visualización exploratoria o modelado."""
        if df.empty:
            print("No hay datos procesados para guardar.")
            return ""

        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(base_dir, folder)
        os.makedirs(target_dir, exist_ok=True)

        file_path = os.path.join(target_dir, file_name)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"Dataset preprocesado guardado en: {file_path}")
        return file_path


if __name__ == "__main__":
    # Prueba independiente leyendo el archivo limpio generado por data_cleaner.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clean_csv_path = os.path.join(current_dir, "data", "clean", "walmart_reviews_clean.csv")

    if os.path.exists(clean_csv_path):
        df_clean = pd.read_csv(clean_csv_path)
        processor = TextProcessor()
        df_processed = processor.process_dataframe(df_clean)
        processor.save_processed_data(df_processed, "walmart_reviews_processed.csv", folder="data/processed")

        print("\nComparación antes vs. después del preprocesamiento:")
        print(df_processed[["rating", "rating_category", "review_text", "clean_text"]].head())
    else:
        print(f"No se encontró el archivo limpio en: {clean_csv_path}")
        print("Asegúrate de haber ejecutado data_cleaner.py previamente.")