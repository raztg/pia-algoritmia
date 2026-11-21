#!/usr/bin/env python3
"""Parte 2 – Machine Learning: hiper-heurística de selección para TSP.

Uso:  python hh_ml.py [--train my_train.csv] [--test my_test.csv] [--out resultados]
Lee los CSV etiquetados (salida del programa en C), entrena Árbol de Decisión,
KNN y Random Forest, evalúa en prueba (accuracy vs baseline, matriz de confusión,
regret) y guarda tablas, figuras y predicciones en --out.
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 42
NUM = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
CAT = ["distribution"]
FEATS = NUM + CAT
COSTS = {"hill_climbing_2opt": "cost_hill_climbing_2opt",
         "simulated_annealing_2opt": "cost_simulated_annealing_2opt",
         "mst_2opt": "cost_mst_2opt"}
ALGS = list(COSTS)

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="my_train.csv")
ap.add_argument("--test", default="my_test.csv")
ap.add_argument("--out", default="resultados")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

tr, te = pd.read_csv(args.train), pd.read_csv(args.test)
al = pd.concat([tr, te], ignore_index=True)
log = {}

# ---------- 1. Balance de clases ----------
bal = pd.DataFrame({"total": al.best_algorithm.value_counts(),
                    "train": tr.best_algorithm.value_counts(),
                    "test": te.best_algorithm.value_counts()}).reindex(ALGS).fillna(0).astype(int)
bal["total_%"] = (100 * bal.total / bal.total.sum()).round(2)
bal.to_csv(f"{args.out}/balance_clases.csv")
print("== Balance de clases ==\n", bal, "\n")
majority = tr.best_algorithm.value_counts().idxmax()  # baseline: moda del ENTRENAMIENTO
base_acc = accuracy_score(te.best_algorithm, [majority] * len(te))
print(f"Baseline (siempre '{majority}'): accuracy test = {base_acc:.4f}\n")
log["baseline"] = {"algoritmo": majority, "accuracy_test": base_acc}

# ---------- 2. Modelos ----------
def prep(scale):
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Árbol de Decisión": (Pipeline([("p", prep(False)), ("m", DecisionTreeClassifier(random_state=SEED))]),
        {"m__max_depth": [3, 4, 5, 8, None], "m__min_samples_leaf": [1, 3, 5, 10],
         "m__class_weight": [None, "balanced"]}),
    "KNN": (Pipeline([("p", prep(True)), ("m", KNeighborsClassifier())]),
        {"m__n_neighbors": [1, 3, 5, 9, 15], "m__weights": ["uniform", "distance"]}),
    "Random Forest": (Pipeline([("p", prep(False)), ("m", RandomForestClassifier(random_state=SEED))]),
        {"m__n_estimators": [200], "m__max_depth": [3, None], "m__min_samples_leaf": [1, 3],
         "m__class_weight": [None, "balanced"]}),
}
Xtr, ytr, Xte, yte = tr[FEATS], tr.best_algorithm, te[FEATS], te.best_algorithm

def regret_of(pred):
    c = te[list(COSTS.values())].to_numpy()
    chosen = np.array([c[i, ALGS.index(p)] for i, p in enumerate(pred)])
    return chosen - c.min(axis=1)

rows, fitted, preds = [], {}, {}
# referencias: baseline y oráculo
r = regret_of([majority] * len(te))
rows.append({"modelo": f"Baseline (siempre {majority})", "accuracy": base_acc,
             "macro_F1": f1_score(yte, [majority]*len(te), average="macro", zero_division=0),
             "regret_medio": r.mean(), "regret_max": r.max(), "%_regret_0": 100*(r < 1e-9).mean()})
for name, (pipe, grid) in models.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1).fit(Xtr, ytr)
    p = gs.predict(Xte); r = regret_of(p)
    fitted[name], preds[name] = gs, p
    rows.append({"modelo": name, "accuracy": accuracy_score(yte, p),
                 "macro_F1": f1_score(yte, p, average="macro", zero_division=0),
                 "regret_medio": r.mean(), "regret_max": r.max(), "%_regret_0": 100*(r < 1e-9).mean()})
    log[name] = {"mejores_hiperparametros": {k: str(v) for k, v in gs.best_params_.items()},
                 "cv_macroF1_train": gs.best_score_,
                 "cm_labels": ALGS, "matriz_confusion": confusion_matrix(yte, p, labels=ALGS).tolist(),
                 "pred_counts": pd.Series(p).value_counts().to_dict()}
    print(f"== {name} ==  params={gs.best_params_}\n   CV macroF1={gs.best_score_:.3f}  "
          f"test acc={rows[-1]['accuracy']:.4f}  regret medio={rows[-1]['regret_medio']:.5f}")
    print(pd.DataFrame(confusion_matrix(yte, p, labels=ALGS), index=[f"real:{a}" for a in ALGS],
                       columns=[f"pred:{a}" for a in ALGS]), "\n")

from sklearn.metrics import classification_report
for nm, p_ in preds.items():
    log[nm]["reporte_por_clase"] = classification_report(yte, p_, labels=ALGS, output_dict=True, zero_division=0)
    print(f"\n-- {nm}: reporte por clase --\n" + classification_report(yte, p_, labels=ALGS, zero_division=0))
res = pd.DataFrame(rows); res.to_csv(f"{args.out}/resultados_test.csv", index=False)
print(res.round(5).to_string(index=False))

# ---------- 3. Mejor modelo: por macro-F1 en CV (criterio decidido sin mirar test) ----------
best_name = max(fitted, key=lambda k: fitted[k].best_score_)
print(f"\nModelo elegido por CV en entrenamiento: {best_name}")
log["modelo_elegido"] = best_name
p = preds[best_name]
out = te[["instance_id", "n", "distribution"] + list(COSTS.values()) + ["best_algorithm"]].copy()
out["pred"] = p; out["regret"] = regret_of(p)
out.to_csv(f"{args.out}/predicciones_test.csv", index=False)

# ---------- 4. Importancia de características ----------
imp = {}
tree = fitted["Árbol de Decisión"].best_estimator_
names = tree.named_steps["p"].get_feature_names_out()
imp["tree_gini"] = dict(zip(names, tree.named_steps["m"].feature_importances_.round(4)))
for nm in ["Árbol de Decisión", best_name]:
    pi = permutation_importance(fitted[nm].best_estimator_, Xte, yte, n_repeats=30,
                                random_state=SEED, scoring="f1_macro")
    imp[f"permutacion_{nm}"] = dict(zip(FEATS, pi.importances_mean.round(4)))
log["importancias"] = imp
log["arbol_texto"] = export_text(tree.named_steps["m"], feature_names=list(names), max_depth=4)
print("\nImportancias:", json.dumps(imp, indent=1, ensure_ascii=False)); print(log["arbol_texto"])

# ---------- 5. Patrones por distribución / tamaño ----------
pat = al.groupby(["distribution", "n"]).best_algorithm.value_counts().unstack(fill_value=0)
pat.to_csv(f"{args.out}/patron_distribucion_tamano.csv")
ratios = pd.DataFrame({"n": al.n, "distribution": al.distribution})
for a in ALGS: ratios["ratio_"+a] = al[COSTS[a]] / al[list(COSTS.values())].min(axis=1)
ratios.groupby(["distribution", "n"]).mean().round(4).to_csv(f"{args.out}/costo_relativo_al_mejor.csv")

# ---------- 6. Figuras ----------
fig, ax = plt.subplots(1, len(fitted), figsize=(4.2*len(fitted), 3.8))
for a, nm in zip(np.atleast_1d(ax), fitted):
    cm = confusion_matrix(yte, preds[nm], labels=ALGS)
    a.imshow(cm, cmap="Blues"); a.set_title(nm, fontsize=10)
    short = ["HC", "SA", "MST"]
    a.set_xticks(range(3), short); a.set_yticks(range(3), short)
    a.set_xlabel("predicho"); a.set_ylabel("real")
    for i in range(3):
        for j in range(3): a.text(j, i, cm[i, j], ha="center", va="center",
                                   color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout(); plt.savefig(f"{args.out}/matrices_confusion.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.5))
s = pd.Series(imp["permutacion_" + best_name]).sort_values()
ax.barh(s.index, s.values); ax.set_title(f"Importancia por permutación (macro-F1) – {best_name}")
plt.tight_layout(); plt.savefig(f"{args.out}/importancia_caracteristicas.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 3.8))
g = ratios.groupby("n")[[ "ratio_"+a for a in ALGS]].mean()
for a, lab in zip(ALGS, ["2-opt HC", "SA+2-opt", "MST"]): ax.plot(g.index, g["ratio_"+a], marker="o", label=lab)
ax.set_xlabel("n"); ax.set_ylabel("costo / mejor costo"); ax.legend(); ax.set_title("Costo relativo al mejor por tamaño")
plt.tight_layout(); plt.savefig(f"{args.out}/costo_relativo_por_n.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ct = pd.crosstab(al.n, al.best_algorithm).reindex(columns=ALGS).fillna(0)
(ct.div(ct.sum(axis=1), axis=0)*100).plot(kind="bar", stacked=True, ax=ax, width=0.8)
ax.set_ylabel("% de instancias ganadas"); ax.set_xlabel("n"); ax.set_title("Algoritmo ganador por tamaño"); ax.legend(["2-opt HC","SA+2-opt","MST"], fontsize=8)
plt.tight_layout(); plt.savefig(f"{args.out}/ganador_por_n.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
cd_ = pd.crosstab(al.distribution, al.best_algorithm).reindex(columns=ALGS).fillna(0)
(cd_.div(cd_.sum(axis=1), axis=0)*100).plot(kind="bar", stacked=True, ax=ax, width=0.6, rot=0)
ax.set_ylabel("% de instancias ganadas"); ax.set_title("Algoritmo ganador por distribución"); ax.legend(["2-opt HC","SA+2-opt","MST"], fontsize=8)
plt.tight_layout(); plt.savefig(f"{args.out}/ganador_por_distribucion.png", dpi=150); plt.close()

json.dump(log, open(f"{args.out}/resumen.json", "w"), indent=1, ensure_ascii=False, default=str)
print(f"\nResultados guardados en {args.out}/")
