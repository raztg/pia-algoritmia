# Parte 2 – Machine Learning (Hiper-heurística de selección para TSP)

> Versión actualizada con las etiquetas corregidas de la parte 1 (`PIAPT1`: SA con T0 = 0.5 y MST con refinamiento).

## Contenido
| Archivo | Qué es |
|---|---|
| `hh_ml.py` | Pipeline completo. `pip install -r requirements.txt` y luego `python hh_ml.py` (opciones: `--train`, `--test`, `--out`). Semilla fija (42). |
| `my_train.csv`, `my_test.csv` | Etiquetas generadas por el programa en C (`PIA/`), sin modificar. |
| `resultados/balance_clases.csv` | Balance de clases (total, train, test). |
| `resultados/resultados_test.csv` | Accuracy, macro-F1, regret medio/máximo y % de instancias con regret 0, por modelo y baseline. |
| `resultados/resumen.json` | Hiperparámetros, F1 de CV, matrices de confusión, reporte por clase, importancias, árbol (texto, profundidad ≤ 4). |
| `resultados/predicciones_test.csv` | Por instancia de test: costos, etiqueta real, predicción del modelo elegido y regret. |
| `resultados/patron_distribucion_tamano.csv` | Cuántas veces gana cada algoritmo por (distribución, n). |
| `resultados/costo_relativo_al_mejor.csv` | Costo medio de cada algoritmo / mejor costo, por (distribución, n). |
| `resultados/*.png` | `matrices_confusion`, `importancia_caracteristicas`, `costo_relativo_por_n`, `ganador_por_n`, `ganador_por_distribucion`. |

## Metodología
- Características: `n`, `mean_pairwise_dist`, `std_coords`, `mean_centroid_dist`, `mean_nn_dist` y `distribution` (one-hot). KNN usa estandarización; los árboles no.
- Modelos: Árbol de Decisión, KNN y Random Forest. Hiperparámetros por GridSearchCV con validación cruzada estratificada de 5 pliegues **solo sobre train**, métrica macro-F1.
- Modelo elegido (mejor macro-F1 en CV, sin mirar test): **Árbol de Decisión**, `max_depth=None`, `min_samples_leaf=1`, `class_weight=None`. KNN: k=3, pesos por distancia. Random Forest: 200 árboles, `min_samples_leaf=3`, `class_weight=balanced`.
- Baseline: predecir siempre la clase más frecuente (hill climbing; es la mayoritaria tanto en train como en las 500 instancias).
- Regret = costo del algoritmo elegido − costo del mejor algoritmo en esa instancia (promedio sobre las 149 instancias de test).
- Importancia: Gini del árbol e importancia por permutación (macro-F1) en test.

## Balance de clases (tus 500 instancias)
| | Total | Train | Test |
|---|---|---|---|
| Hill climbing (2-opt) | 341 (68.2%) | 240 | 101 |
| SA + 2-opt | 96 (19.2%) | 68 | 28 |
| MST + refinamiento | 63 (12.6%) | 43 | 20 |

El balance es muy cercano al de referencia del instructor (68% / 12% SA / 19% MST).

## Resultados (conjunto de prueba, 149 instancias)
| Modelo | Accuracy | Macro-F1 | Regret medio | Regret máx. | % con regret 0 |
|---|---|---|---|---|---|
| Baseline (siempre hill climbing) | **0.6779** | 0.269 | 0.0249 | 0.764 | 75.2 |
| **Árbol de Decisión (elegido)** | 0.6309 | 0.463 | 0.0232 | 0.469 | 79.2 |
| KNN (k=3) | 0.6577 | 0.428 | **0.0151** | 0.236 | 78.5 |
| Random Forest | 0.5839 | **0.482** | 0.0209 | 0.533 | 81.2 |

Matrices de confusión (filas = real, columnas = predicho; orden HC, SA, MST):
- Árbol: [[78, 14, 9], [8, 12, 8], [11, 5, 4]]
- KNN: [[87, 8, 6], [15, 9, 4], [11, 7, 2]]
- Random Forest: [[63, 27, 11], [4, 18, 6], [5, 9, 6]]

Recall por clase (HC / SA / MST): Árbol 0.77 / 0.43 / 0.20; KNN 0.86 / 0.32 / 0.10; Random Forest 0.62 / 0.64 / 0.30. Detalle completo en `resumen.json`.

## Lectura de los resultados
- **Accuracy:** ningún modelo supera el baseline (67.8%). El mejor es KNN con 65.8%. Es habitual cuando la clase mayoritaria es 68% y las características dan poca señal.
- **Macro-F1:** los tres modelos superan claramente al baseline (0.43–0.48 vs 0.27), es decir, sí detectan algo de SA y MST, aunque con precisión baja (precisión SA 0.33–0.39, MST 0.17–0.26).
- **Regret:** los tres modelos tienen regret medio menor que el baseline (0.015–0.023 vs 0.025); KNN es el mejor. Es la métrica donde el selector aporta valor práctico: aunque se equivoque de clase, elige algoritmos de costo cercano al mejor, y baja el peor caso (regret máx. de 0.76 a 0.24–0.53). La mejora es moderada, no grande.
- **¿Colapsa a predecir siempre el dominante?** No: Árbol y Random Forest predicen SA y MST con frecuencia (Árbol: 97 HC / 31 SA / 21 MST; RF: 72 / 54 / 23). KNN sí se inclina más a HC (113 de 149). Los modelos con `class_weight=balanced` detectan más SA/MST a costa de accuracy.
- **Importancia de características:** en el árbol (Gini) pesan `mean_pairwise_dist` (0.30), `mean_nn_dist` (0.23), `n` (0.17), `std_coords` (0.14), `mean_centroid_dist` (0.13) y `distribution` (0.04). Por permutación en test las más importantes son `n` (0.144) y `mean_nn_dist` (0.103); `distribution` es ≈ 0 o negativa. Conclusión: el **tamaño** y la **densidad local** (`mean_nn_dist`, que decrece con n) determinan el ganador; la distribución casi no aporta una vez conocidos esos valores.
- **Patrón por tamaño** (ver `ganador_por_n.png`, `costo_relativo_por_n.png`): hill climbing gana casi siempre con n ≥ 80 (n=80: 53 de 54; n=100: 54 de 54). Con n entre 15 y 40, SA y MST ganan con frecuencia (p. ej. n=15: HC 27, MST 8, SA 22; n=20: HC 23, MST 15, SA 19). Con n=60 vuelve a dominar HC (47 de 54). El costo relativo lo confirma: a n=100 SA cuesta en promedio 18% más que el mejor y MST 25% más, mientras que a n=15–30 hill climbing queda 1–2% por encima. La razón probable es el presupuesto fijo de 10 000 iteraciones de SA, que no alcanza para n grande.
- **Patrón por distribución:** diferencias pequeñas: hill climbing gana 80% de las instancias `border`, 63% de `clustered` y 62% de `uniform`; SA y MST ganan algo más en `clustered` y `uniform`.
- **Accuracy del árbol por tamaño en test:** alta con n=5 (0.78) y n ≥ 60 (≥ 0.73, 1.0 con n ≥ 80); mala en la zona intermedia n = 15–40 (0.29–0.50), donde el ganador es realmente incierto.

## ⚠️ Notas y limitaciones para el reporte
1. **Etiquetas ruidosas con n pequeño.** 174 de las 500 instancias tienen costos prácticamente idénticos en los tres algoritmos (diferencia < 1e-6), la mayoría (139) con n ≤ 15, y 40 de las instancias etiquetadas SA/MST ganan a hill climbing solo por ruido de punto flotante (< 1e-6). Para esas instancias la etiqueta es casi arbitraria, lo que limita cualquier clasificador y explica parte del error en n pequeño. El regret lo mide bien: elegir "mal" entre algoritmos empatados cuesta 0.
2. **Etiquetas no regenerables exactamente:** `trial.c` usa `srand(time(NULL))`, así que SA y MST (que ahora también usa un proceso estocástico) cambian entre corridas. En una corrida de prueba el balance fue ~253 HC / 68 SA / 30 MST en train, similar pero no idéntico. Los CSV incluidos son la versión oficial.
3. **Muestra pequeña:** 149 instancias de test (20 MST, 28 SA); un acierto más o menos mueve las métricas por clase varios puntos. Las importancias por permutación y la CV son ruidosas.
4. **Criterio de selección:** el modelo se eligió por macro-F1 en CV (no por accuracy), lo que favorece predecir SA y MST y explica por qué la accuracy queda bajo el baseline. Elegir por accuracy llevaría a "siempre hill climbing". Se ajustó la CV (3 → 5 pliegues) y la malla del árbol tras una primera corrida con las nuevas etiquetas; la selección sigue sin usar test.
5. **Árbol sin restricción de profundidad** (lo eligió la CV): es poco interpretable y propenso a sobreajuste. `resumen.json` guarda solo hasta profundidad 4.
6. **Características:** solo las provistas más `n` y `distribución`; no se calcularon otras desde las coordenadas.
7. **Versiones:** probado con Python 3.13 y scikit-learn 1.9.1 (`requirements.txt` no fija versiones).

## Reproducir
```
pip install -r requirements.txt
python hh_ml.py            # usa my_train.csv / my_test.csv y escribe en resultados/
```
