import os
import time
import pandas as pd

# Importar las clases de cada módulo del framework
from data_loader import DataLoader
from data_cleaner import DataCleaner
from text_processor import TextProcessor
from exploratory_eda import ExploratoryEDA
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from report_generator import ReportGenerator


def run_pipeline():
    """Ejecuta el flujo completo de extremo a extremo (Módulos 1 al 7)."""
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 65)
    print("INICIANDO PIPELINE DE ANÁLISIS DE RESEÑAS - WALMART MÉXICO (iOS)")
    print("=" * 65)

    # --- MÓDULO 1: Carga de datos ---
    print("\n[1/7] Ejecutando Módulo de Carga de Datos...")
    raw_path = os.path.join(base_dir, "data", "raw", "walmart_reviews_raw.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    # Si ya existe un CSV crudo local lo reutiliza; si no, consulta la API RSS
    loader = DataLoader(app_id="823947897", country="mx")
    df_raw = loader.fetch_reviews(max_pages=10)
    if df_raw.empty and os.path.exists(raw_path):
        print("Usando respaldo local existente en data/raw/...")
        df_raw = pd.read_csv(raw_path)
    else:
        df_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    if df_raw.empty:
        print("[ERROR CRÍTICO] No se obtuvieron datos. Abortando pipeline.")
        return

    # --- MÓDULO 2: Validación y limpieza ---
    print("\n[2/7] Ejecutando Módulo de Validación y Limpieza...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df_raw)
    interim_path = os.path.join(base_dir, "data", "interim", "walmart_reviews_cleaned.csv")
    os.makedirs(os.path.dirname(interim_path), exist_ok=True)
    df_clean.to_csv(interim_path, index=False, encoding="utf-8-sig")

    # --- MÓDULO 3: Preprocesamiento y NLP ---
    print("\n[3/7] Ejecutando Módulo de Preprocesamiento NLP...")
    processor = TextProcessor()
    df_processed = processor.process_dataframe(df_clean)
    processed_path = os.path.join(base_dir, "data", "processed", "walmart_reviews_processed.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False, encoding="utf-8-sig")

    # --- MÓDULO 4: Visualización exploratoria (EDA) ---
    print("\n[4/7] Ejecutando Módulo de Análisis Exploratorio y Gráficas...")
    eda = ExploratoryEDA(output_folder="reports/figures")
    eda.print_descriptive_stats(df_processed)
    eda.plot_rating_distribution(df_processed)
    eda.plot_temporal_trend(df_processed)
    eda.plot_top_complaint_words(df_processed, top_n=12)

    # --- MÓDULO 5: Entrenamiento de modelos ---
    print("\n[5/7] Ejecutando Módulo de Entrenamiento de Modelos (ML + LDA)...")
    trainer = ModelTrainer(models_folder="models")
    trainer.train_sentiment_model(df_processed)
    trainer.train_topic_model(df_processed, n_topics=4, n_top_words=6)

    # --- MÓDULO 6: Evaluación de resultados ---
    print("\n[6/7] Ejecutando Módulo de Evaluación de Resultados...")
    evaluator = ModelEvaluator(models_folder="models", reports_folder="reports")
    evaluator.evaluate_sentiment_classifier(df_processed)
    evaluator.evaluate_topic_coherence_proxy(df_processed)

    # --- MÓDULO 7: Generación de reportes ejecutivos ---
    print("\n[7/7] Ejecutando Módulo de Generación de Reportes...")
    reporter = ReportGenerator(models_folder="models", reports_folder="reports")
    df_final = reporter.enrich_predictions(df_processed)
    reporter.generate_executive_report(df_final)

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"PIPELINE COMPLETADO EXITOSAMENTE EN {elapsed:.2f} SEGUNDOS")
    print("Revisa las salidas en las carpetas: /data, /models y /reports")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()