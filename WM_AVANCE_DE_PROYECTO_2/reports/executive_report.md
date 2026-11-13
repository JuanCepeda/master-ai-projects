# Reporte Ejecutivo de Experiencia de Usuario (UX/QA)
**Aplicación:** Walmart México (iOS - App ID: 823947897)  
**Fecha de generación:** 2026-11-13 13:42  

---

## 1. Indicadores Clave de Desempeño (KPIs)
* **Total de reseñas analizadas:** 500
* **Calificación promedio histórica:** 3.34 / 5.00 estrellas
* **Tasa de insatisfacción (1-2 estrellas):** 34.6%
* **Alertas silenciosas (4-5 estrellas con texto de queja):** 16 casos detectados

---

## 2. Desempeño del Modelo Predictivo (Validación Técnica)
* **Exactitud (Accuracy):** 65.00%
* **F1-Score Ponderado:** 67.33%
* **Muestras evaluadas en prueba (20%):** 100

---

## 3. Matriz de Priorización de Problemas (Tópicos LDA en Quejas)
Agrupamiento automático de reseñas insatisfechas para orientar acciones de Desarrollo y QA:

| Tópico Detectado | Volumen en Quejas | Palabras Clave Asociadas |
| :--- | :--- | :--- |
| **Tópico 1** | 46 (26.6%) | `pedido, tienda, linea, pagar, descuento, codigo` |
| **Tópico 2** | 57 (32.9%) | `envio, pedido, pedidos, entrega, servicio, productos` |
| **Tópico 3** | 38 (22.0%) | `servicio, pedido, cada, ia, cliente, pesimo` |
| **Tópico 4** | 32 (18.5%) | `comprar, pedido, solo, productos, bien, deja` |

---

## 4. Monitoreo de Estabilidad por Versión de la App (Top 5)
| Versión | Reseñas | Calificación Promedio | Estado Operativo |
| :--- | :--- | :--- | :--- |
| v26.34 | 92 | 3.08 ★ | Estable |
| v26.41.1 | 73 | 2.68 ★ | Estable |
| v26.36 | 55 | 3.55 ★ | Estable |
| v26.32 | 41 | 3.24 ★ | Estable |
| v26.39 | 41 | 3.22 ★ | Estable |

---

## 5. Recomendaciones Accionables para el Negocio
1. **Equipo de QA / Ingeniería:** Priorizar la revisión de los flujos asociados al **Tópico 1** y **Tópico 2**, ya que concentran el mayor volumen de fricción en usuarios de 1 y 2 estrellas.
2. **Equipo de Producto / UX:** Auditar los **16 casos de discrepancia** donde el usuario otorgó calificación alta pero el modelo detectó lenguaje de insatisfacción en el comentario.
3. **Monitoreo Continuo:** Utilizar este pipeline tras cada despliegue en App Store para detectar caídas de calificación por debajo de `2.5 ★` en las primeras 48 horas.
