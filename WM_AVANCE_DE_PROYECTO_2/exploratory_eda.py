import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ExploratoryEDA:
    """Módulo 4: Análisis Exploratorio de Datos y generación de visualizaciones."""

    def __init__(self, output_folder: str = "reports/figures"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, output_folder)
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def print_descriptive_stats(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("=" * 60)
        print(f"Total de reseñas válidas analizadas: {len(df)}")
        print(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
        print(f"Calificación promedio: {df['rating'].mean():.2f} / 5.00 estrellas")
        print(f"Mediana de calificación: {df['rating'].median():.1f} estrellas")

        print("\nDistribución por categoría de satisfacción:")
        print(df["rating_category"].value_counts())

        print("\nEstadísticas de longitud de texto (palabras clave por reseña):")
        print(df[["rating", "char_count", "word_count"]].describe().round(2))
        print("=" * 60 + "\n")

    def plot_rating_distribution(self, df: pd.DataFrame) -> str:
        """Gráfica 1: Distribución de calificaciones (1 a 5 estrellas)."""
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(
            data=df,
            x="rating",
            hue="rating",
            palette=["#d9534f", "#f0ad4e", "#ffd700", "#5bc0de", "#0fe90f"],
            legend=False,
        )

        plt.title("Distribución de Calificaciones - App Walmart México (iOS)", fontsize=13, fontweight="bold")
        plt.xlabel("Calificación (Estrellas)", fontsize=11)
        plt.ylabel("Cantidad de Reseñas", fontsize=11)

        # Añadir etiquetas con conteo y porcentaje encima de cada barra
        total = len(df)
        for p in ax.patches:
            count = int(p.get_height())
            pct = 100 * count / total
            ax.annotate(
                f"{count}\n({pct:.1f}%)",
                (p.get_x() + p.get_width() / 2.0, count),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.ylim(0, df["rating"].value_counts().max() * 1.18)
        plt.tight_layout()

        file_path = os.path.join(self.output_dir, "1_distribucion_calificaciones.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Gráfica 1 guardada en: {file_path}")
        return file_path

    def plot_temporal_trend(self, df: pd.DataFrame) -> str:
        """Gráfica 2: Evolución mensual del promedio de calificación y volumen de reseñas."""
        if "year_month" not in df.columns:
            print("Advertencia: No se encontró la columna 'year_month'.")
            return ""

        monthly = (
            df.groupby("year_month")
            .agg(avg_rating=("rating", "mean"), review_count=("review_id", "count"))
            .reset_index()
            .sort_values("year_month")
        )

        # Tomar los últimos 12 meses con actividad para mantener legibilidad
        monthly = monthly.tail(12)

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Barras: Volumen de reseñas
        ax1.bar(monthly["year_month"], monthly["review_count"], color="#b0c4de", alpha=0.6, label="Volumen de reseñas")
        ax1.set_xlabel("Mes de Publicación", fontsize=11)
        ax1.set_ylabel("Cantidad de Reseñas", color="#4682b4", fontsize=11)
        ax1.tick_params(axis="x", rotation=45)

        # Línea: Calificación promedio
        ax2 = ax1.twinx()
        ax2.plot(
            monthly["year_month"],
            monthly["avg_rating"],
            color="#d9534f",
            marker="o",
            linewidth=2.5,
            label="Calificación Promedio",
        )
        ax2.set_ylabel("Calificación Promedio (1-5)", color="#d9534f", fontsize=11)
        ax2.set_ylim(1, 5.3)

        plt.title("Tendencia Temporal de Calificaciones y Volumen de Reseñas", fontsize=13, fontweight="bold")
        fig.tight_layout()

        file_path = os.path.join(self.output_dir, "2_tendencia_temporal.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Gráfica 2 guardada en: {file_path}")
        return file_path

    def plot_top_complaint_words(self, df: pd.DataFrame, top_n: int = 12) -> str:
        """Gráfica 3: Palabras más frecuentes en reseñas de usuarios insatisfechos (1-2 estrellas)."""
        negative_reviews = df[df["rating"] <= 2]["clean_text"].dropna()
        all_words = " ".join(negative_reviews).split()
        word_counts = Counter(all_words).most_common(top_n)

        if not word_counts:
            print("No hay suficientes palabras en reseñas negativas para graficar.")
            return ""

        words_df = pd.DataFrame(word_counts, columns=["Palabra", "Frecuencia"])

        plt.figure(figsize=(9, 5))
        sns.barplot(data=words_df, x="Frecuencia", y="Palabra", hue="Palabra", palette="Reds_r", legend=False)
        plt.title(f"Top {top_n} Términos Recurrentes en Reseñas Insatisfechas (1-2 Estrellas)", fontsize=13, fontweight="bold")
        plt.xlabel("Frecuencia de Aparición", fontsize=11)
        plt.ylabel("Término Normalizado", fontsize=11)
        plt.tight_layout()

        file_path = os.path.join(self.output_dir, "3_top_palabras_quejas.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Gráfica 3 guardada en: {file_path}")
        return file_path


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_csv_path = os.path.join(current_dir, "data", "processed", "walmart_reviews_processed.csv")

    if os.path.exists(processed_csv_path):
        df_proc = pd.read_csv(processed_csv_path)
        eda = ExploratoryEDA(output_folder="reports/figures")

        # 1. Imprimir estadísticas para el Punto 8
        eda.print_descriptive_stats(df_proc)

        # 2. Generar las 3 gráficas para el Punto 9
        eda.plot_rating_distribution(df_proc)
        eda.plot_temporal_trend(df_proc)
        eda.plot_top_complaint_words(df_proc, top_n=12)
    else:
        print(f"No se encontró el archivo procesado en: {processed_csv_path}")
        print("Ejecuta primero text_processor.py.")