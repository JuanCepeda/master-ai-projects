import json
import os
import time
import urllib.request
import pandas as pd


class DataLoader:
    """Módulo 1: Carga e Ingesta de datos de reseñas de la App Store de Walmart México (APP_ID: 823947897)."""

    def __init__(self, app_id: str = "823947897", country: str = "mx"):
        self.app_id = app_id
        self.country = country

    def fetch_reviews(self, max_pages: int = 10) -> pd.DataFrame:
        """Recopila reseñas iterando sobre la paginación oficial RSS de iTunes App Store (1 a 10 páginas)."""
        all_reviews = []
        print(
            f"Iniciando carga de reseñas para App ID {self.app_id} (Páginas 1 a {max_pages})..."
        )

        for page in range(1, max_pages + 1):
            url = f"https://itunes.apple.com/{self.country}/rss/customerreviews/page={page}/id={self.app_id}/sortby=mostrecent/json"
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode("utf-8"))

                entries = data.get("feed", {}).get("entry", [])
                if not entries:
                    break

                for entry in entries:
                    # Omitir el encabezado general del feed de iTunes
                    if "im:name" in entry:
                        continue

                    all_reviews.append({
                        "review_id": entry.get("id", {}).get("label"),
                        "user_name": entry.get("author", {})
                        .get("name", {})
                        .get("label"),
                        "title": entry.get("title", {}).get("label"),
                        "review_text": entry.get("content", {}).get("label"),
                        "rating": int(
                            entry.get("im:rating", {}).get("label", 0)
                        ),
                        "app_version": entry.get("im:version", {}).get(
                            "label"
                        ),
                        "date": entry.get("updated", {}).get("label"),
                        "country": self.country.upper(),
                    })

                print(
                    f"  Página {page} procesada. Total acumulado: {len(all_reviews)} reseñas."
                )
                time.sleep(0.2)  # Pausa de cortesía entre peticiones

            except Exception as e:
                print(f"  Error al leer la página {page}: {str(e)}")
                break

        df = pd.DataFrame(all_reviews)

        if not df.empty:
            df = df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)
            print(
                f"\nIngesta finalizada: {len(df)} reseñas únicas recopiladas."
            )
        else:
            print("\nNo se obtuvieron datos de la API.")

        return df

    def save_to_csv(
        self,
        df: pd.DataFrame,
        file_name: str = "walmart_reviews_raw.csv",
        folder: str = "data/loader",
    ) -> str:
        """Guarda el DataFrame crudo en una carpeta específica para trazabilidad y respaldo."""
        if df.empty:
            print("No hay datos para guardar.")
            return ""

        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(base_dir, folder)

        # 1. Primero creamos la carpeta si no existe
        os.makedirs(target_dir, exist_ok=True)

        # 2. Unimos la ruta de la carpeta con el nombre del archivo
        file_path = os.path.join(target_dir, file_name)

        # 3. Guardamos el archivo en la ruta completa
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"Respaldo guardado en: {file_path}")
        return file_path


if __name__ == "__main__":
    loader = DataLoader(app_id="823947897", country="mx")
    df_reviews = loader.fetch_reviews(max_pages=10)

    if not df_reviews.empty:
        # Solo pasamos el nombre del archivo y la carpeta destino por separado
        loader.save_to_csv(
            df_reviews, file_name="walmart_reviews_raw.csv", folder="data/loader"
        )
        print("\nVista previa de los datos cargados:")
        print(
            df_reviews[
                ["date", "rating", "app_version", "title", "review_text"]
            ].head()
        )