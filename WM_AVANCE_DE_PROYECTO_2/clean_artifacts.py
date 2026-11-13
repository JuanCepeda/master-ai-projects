import os
import shutil


def clean_project_outputs(keep_raw_data: bool = False):
    """Elimina las carpetas y archivos generados durante la ejecución del pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Carpetas generadas por el framework
    folders_to_clean = ["models", "reports", "__pycache__"]

    if keep_raw_data:
        # Borra solo datos intermedios y procesados, conservando la descarga original
        folders_to_clean.extend([
            os.path.join("data", "interim"),
            os.path.join("data", "processed"),
        ])
    else:
        # Borra toda la carpeta data/ completa
        folders_to_clean.append("data")

    print("\n--- INICIANDO LIMPIEZA DE ARCHIVOS GENERADOS ---")
    deleted_count = 0

    for folder in folders_to_clean:
        target_path = os.path.join(base_dir, folder)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
            print(f"[ELIMINADO] {target_path}")
            deleted_count += 1
        else:
            print(f"[OMITIDO] No existe: {folder}/")

    print(f"\nLimpieza finalizada. Se eliminaron {deleted_count} directorios.")
    print("Tus archivos de código (.py) permanecen intactos.\n")


if __name__ == "__main__":
    # Cambia a True si deseas conservar data/raw/walmart_reviews_raw.csv sin volver a consultar la API
    clean_project_outputs(keep_raw_data=False)