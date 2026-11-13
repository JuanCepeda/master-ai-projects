import os
import pandas as pd


class DataCleaner:
    """Módulo 2: Validación, limpieza y estandarización estructural de reseñas."""

    def __init__(self, target_country: str = "MX"):
        self.target_country = target_country.upper()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ejecuta el pipeline de validación y limpieza sobre el DataFrame crudo."""
        if df.empty:
            print("Advertencia: El DataFrame de entrada está vacío.")
            return df

        df_clean = df.copy()
        initial_rows = len(df_clean)
        print(f"Iniciando limpieza de datos ({initial_rows} registros crudos)...")

        # 1. Eliminación de duplicados por ID único de reseña
        df_clean = df_clean.drop_duplicates(subset=["review_id"])

        # 2. Filtrado por país objetivo (Mercado Mexicano)
        if "country" in df_clean.columns:
            df_clean["country"] = df_clean["country"].astype(str).str.upper().str.strip()
            df_clean = df_clean[df_clean["country"] == self.target_country]

        # 3. Manejo de valores nulos en columnas críticas
        df_clean = df_clean.dropna(subset=["review_text", "rating"])
        df_clean["title"] = df_clean["title"].fillna("Sin título")
        df_clean["app_version"] = df_clean["app_version"].fillna("Desconocida")

        # 4. Validación del rango de calificación (1 a 5 estrellas)
        df_clean["rating"] = pd.to_numeric(df_clean["rating"], errors="coerce")
        df_clean = df_clean[df_clean["rating"].between(1, 5)]
        df_clean["rating"] = df_clean["rating"].astype(int)

        # 5. Estandarización de fechas a formato datetime (ISO)
        if "date" in df_clean.columns:
            df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce", utc=True)
            # Eliminar zona horaria para facilitar agrupaciones y exportaciones
            df_clean["date"] = df_clean["date"].dt.tz_localize(None)
            # Crear columna auxiliar de año-mes para análisis de tendencias
            df_clean["year_month"] = df_clean["date"].dt.to_period("M").astype(str)

        # 6. Limpieza básica de espacios en blanco extra en textos
        df_clean["title"] = df_clean["title"].astype(str).str.strip()
        df_clean["review_text"] = df_clean["review_text"].astype(str).str.strip()

        # Eliminar reseñas que quedaron vacías tras quitar espacios
        df_clean = df_clean[df_clean["review_text"] != ""]

        # Reiniciar índice para mantener trazabilidad limpia
        df_clean = df_clean.reset_index(drop=True)
        final_rows = len(df_clean)

        print(f"Limpieza completada: {final_rows} registros válidos ({initial_rows - final_rows} descartados).")
        return df_clean

    def save_clean_data(
        self,
        df: pd.DataFrame,
        file_name: str = "walmart_reviews_clean.csv",
        folder: str = "data/clean",
    ) -> str:
        """Guarda el DataFrame limpio en una carpeta específica dentro del proyecto."""
        if df.empty:
            print("No hay datos limpios para guardar.")
            return ""

        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(base_dir, folder)

        # Crea la carpeta automáticamente si no existe
        os.makedirs(target_dir, exist_ok=True)

        file_path = os.path.join(target_dir, file_name)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"Archivo limpio guardado en: {file_path}")
        return file_path


if __name__ == "__main__":
    # Prueba independiente leyendo el CSV generado por data_loader.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv_path = os.path.join("data/loader", "walmart_reviews_raw.csv")

    if os.path.exists(raw_csv_path):
        df_raw = pd.read_csv(raw_csv_path)
        cleaner = DataCleaner(target_country="MX")
        df_cleaned = cleaner.clean_data(df_raw)
        cleaner.save_clean_data(df_cleaned, "walmart_reviews_clean.csv")

        print("\nResumen de tipos de datos y nulos después de limpiar:")
        print(df_cleaned.info())
    else:
        print("No se encontró 'walmart_reviews_raw.csv'. Ejecuta primero data_loader.py.")