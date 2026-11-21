# Proyecto Final – Hiper-heurística de Selección para TSP

Dada una instancia de TSP, predecir cuál de tres algoritmos (2-opt hill climbing, recocido simulado + 2-opt, MST + 2-opt) obtiene el mejor tour.

## Estructura sugerida del repositorio
```
README.md
part1_tag/                 Parte 1: etiquetado (código en C y CSV etiquetados)
  trial.c
  my_train.csv
  my_test.csv
hh_tsp_dataset/      Dataset provisto por el curso (instancias y características, sin etiquetas)
part2_ml/                  Parte 2: Machine Learning (Python)
  hh_ml.py
  my_train.csv, my_test.csv   (copias de part1_tag/)
  requirements.txt
  LEEME.md                    notas, resultados y limitaciones
  resultados/                 tablas, figuras y predicciones
```

## Parte 1 – Generar las etiquetas (C)
`trial.c` corre los tres algoritmos sobre cada instancia e imprime en pantalla un CSV con los costos y `best_algorithm`.

```
cd part1_tag                      # hh_tsp_dataset/ debe estar en la misma carpeta que trial.c
gcc -O2 -o trial trial.c -lm
./trial > my_train.csv
```
- Las rutas son relativas: el programa lee `hh_tsp_dataset/data/train.csv` y `hh_tsp_dataset/data/instances/inst_NNNN.csv` desde el directorio donde se ejecuta.
- Procesa solo `train.csv`. Para el conjunto de prueba se cambia `data/train.csv` por `data/test.csv` en la línea 27, se recompila y se redirige a `my_test.csv`.
- Los CSV incluidos son las etiquetas oficiales. **No se pueden regenerar idénticas**: el programa usa `srand(time(NULL))` y SA y MST son estocásticos. Tarda menos de un segundo en el conjunto de entrenamiento.

### Parámetros de los algoritmos (parte de las etiquetas)
- **2-opt hill climbing:** parte del tour de vecino más cercano y aplica 2-opt hasta que no hay mejora.
- **SA + 2-opt:** parte del tour de vecino más cercano; movimientos 2-opt aleatorios; temperatura inicial **0.5**, enfriamiento geométrico **0.9995** por iteración, **10 000** iteraciones; devuelve el mejor tour visitado.
- **MST + 2-opt:** árbol de expansión mínima (Prim), recorrido en preorden y después el mismo procedimiento de refinamiento con movimientos 2-opt aleatorios, devolviendo el mejor tour visitado.
- **Ganador:** menor costo; en empate gana hill climbing, luego SA, luego MST.

## Parte 2 – Machine Learning (Python)
```
cd part2_ml
pip install -r requirements.txt
python hh_ml.py
```
Lee `my_train.csv` y `my_test.csv` y escribe en `resultados/`. Metodología, resultados completos y limitaciones en `part2_ml/LEEME.md`.

## Resumen de resultados
- Balance de clases (500 instancias): hill climbing 68.2%, SA 19.2%, MST 12.6%.
- Baseline en test (siempre hill climbing): accuracy 67.8%. Ningún modelo lo supera en accuracy (Árbol 63.1%, KNN 65.8%, Random Forest 58.4%).
- Sí lo superan en macro-F1 (0.43–0.48 vs 0.27) y en regret medio (0.015–0.023 vs 0.025); KNN tiene el menor regret.
- Hill climbing domina con n ≥ 80; con n entre 15 y 40, SA y MST ganan con frecuencia. El tamaño y `mean_nn_dist` son las características más importantes; la distribución aporta poco.
