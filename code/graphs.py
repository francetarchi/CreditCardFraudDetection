import ast
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import paths


### UTILITY FUNCTIONS ###
def find_ROC_stats(fpr, tpr, thresholds):
    # Rimuovo la prima soglia infinita
    thresholds = thresholds[1:]
    tpr = tpr[1:]
    fpr = fpr[1:]

    index_tpr80 = -1
    for j in range (len(tpr)):
        if tpr[j] >= 0.8:
            if tpr[j] == 1.0:
                index_tpr80 = j - 1
            else:
                index_tpr80 = j
            break

    threshold80 = thresholds[index_tpr80]
    tpr80 = tpr[index_tpr80]
    fpr80 = fpr[index_tpr80]

    print(f"  --> {model_name} - Threshold for TPR 80%: {threshold80}, FPR: {fpr80}, index: {index_tpr80}")

    return fpr80, tpr80, threshold80


### RESOURCE LOADING ###
# Caricamento del test set preprocessato (sbilanciato)
print("Loading preprocessed testing set...")
test_set = pd.read_csv(paths.PREP_TEST_PATH)
y_test = test_set["isFraud"]
X_test = test_set.drop(columns=["isFraud"])

# Caricamento dei modelli salvati
print("Loading trained models...")
models = {}
model_paths = {
    "KNN": paths.KNN_PATH,
    "NB": paths.NB_PATH,
    "DT": paths.DT_PATH,
    "RF": paths.RF_PATH,
    "ADA": paths.ADA_PATH,
    "XGB": paths.XGB_PATH
}

for name, path in model_paths.items():
    models[name] = joblib.load(path)
tot = len(models)
print(f"  --> Models loaded: {list(models.keys())} (total: {tot})")

# Generazione e salvataggio delle predizioni per tutti i modelli
i = 1
y_pred = {}
print("Generating predictions for all models...")
for model_name, model in models.items():
    print(f"  --> Processing predictions for {model_name}...")

    # Probabilità della classe positiva
    y_pred[model_name] = model.predict_proba(X_test)[:, 1]
    print(f"  --> Predictions done for {model_name} ({i}/{tot}).")

    i += 1
print("Saving predictions to CSV...")
predictions_df = pd.DataFrame(y_pred)
predictions_df.to_csv(paths.PREDICTIONS_PATH, index=False)

# # Caricamento delle predizioni salvate
# print("Loading saved predictions...")
# y_pred = pd.read_csv(paths.PREDICTIONS_PATH).to_dict(orient='list')

# Caricamento delle statistiche dei modelli
print("Loading model statistics...")
model_stats = pd.read_csv("model_results/unique.csv")


### PLOTTING ###
# Plot confusion matrices
print("Plotting confusion matrices...")
for index, row in model_stats.iterrows():
    model_name = row['Model']
    cm = np.array(ast.literal_eval(row['Confusion Matrix']))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(cm, cmap='Blues', alpha=0.85, norm=LogNorm(vmin=2000, vmax=50000))

    # Colorbar personalizzata
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000])
    cbar.set_ticklabels(['2000', '3000', '5000', '10000', '20000', '30000', '40000', '50000'])
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f'Confusion Matrix - {model_name}', pad=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1], ['Non-Fraud', 'Fraud'])
    ax.set_yticks([0, 1], ['Non-Fraud', 'Fraud'])

    # Annotazioni dentro le celle
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value}', ha='center', va='center', fontsize=12)

    # Aggiusto margini manualmente per evitare tagli
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.18)

    fig.savefig(f'graphs/confusion_matrices/{model_name}_cm.png', dpi=150, bbox_inches='tight', pad_inches=0.3)
    fig.savefig(f'graphs_svg/confusion_matrices/{model_name}_cm.svg', bbox_inches='tight', pad_inches=0.3)

    plt.close(fig)
plt.clf()

# Plot ROC curves
i = 1
print("Plotting ROC curves...")
for model_name, model in models.items():
    print(f"  --> Processing ROC for {model_name}...")
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred[model_name])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    print(f"  --> ROC plotted for {model_name} ({i}/{tot}).")
    
    i += 1
# Aggiungo la linea diagonale
plt.plot([0,1], [0,1], "--", color="gray")

# Aggiungo una linea orizzontale all'altezza del TPR all'80%
plt.plot([0,1], [0.8,0.8], "--", color="black", alpha=0.6)

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves comparison")
plt.legend()
plt.savefig('graphs/ROC_comparison.png')
plt.savefig('graphs_svg/ROC_comparison.svg')
plt.clf()

#Plot thresholds vs TPR
i = 1
plt.figure(figsize=(10, 6))
print("Plotting thresholds vs TPR...")
for model_name, model in models.items():
    print(f"  --> Extracting thresholds for {model_name}...")

    # ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[model_name])

    # Elaboro i risultati del ROC
    fpr80, tpr80, threshold80 = find_ROC_stats(fpr, tpr, thresholds)

    # Plotto il grafico Thresholds vs TPR
    color = plt.get_cmap('tab10')( (i-1) % 10 )
    plt.plot(thresholds, tpr, label=model_name, color=color)
    plt.plot(threshold80, tpr80, marker='o', markersize=7, color=color)

    # Righe tratteggiate orizzontali
    if model_name in {"KNN", "RF", "XGB"}:
        plt.axhline(y=tpr80, xmax=threshold80+0.04, color=color, linestyle="--", alpha=0.6)
    elif model_name == "DT":
        plt.axhline(y=tpr80, xmax=threshold80+0.02, color=color, linestyle="--", alpha=0.6)
    else:
        plt.axhline(y=tpr80, xmax=threshold80+0.01, color=color, linestyle="--", alpha=0.6)
    
    # Righe tratteggiate verticali
    plt.axvline(x=threshold80, ymax=tpr80-0.02, color=color, linestyle="--", alpha=0.6)
    
    # Label dei valori dei pallini sull'asse orizzontale
    if model_name == "NB":
        plt.text(threshold80-0.01, -0.20, f"{threshold80:.3f}", rotation=45, color=color, ha="center", va="bottom", fontsize=9, fontweight="bold")
    elif model_name in {"DT", "RF", "ADA", "XGB"}:
        plt.text(threshold80-0.01, -0.15, f"{threshold80:.3f}", rotation=45, color=color, ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Label dei valori dei pallini sull'asse verticale
    if tpr80 < 0.79:
        plt.text(-0.105, tpr80+0.01, f"{tpr80:.2f}", rotation=-45, color=color, ha="left", va="center", fontsize=9, fontweight="bold")
    print(f"  --> Thresholds plotted for {model_name}.")

    i += 1
# Riga verticale della threshold == 0.5
plt.axvline(x=0.5, ymax = 1, color="black", linestyle="dotted", alpha=0.6)
plt.text(0.49, -0.15, f"{0.5:.3f}", rotation=45, color="black", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.xlabel("Thresholds", labelpad=20)
plt.ylabel("TPR", labelpad=15)
plt.title("Thresholds vs TPR")
plt.legend()
plt.savefig('graphs/Thresholds_TPR.png')
plt.savefig('graphs_svg/Thresholds_TPR.svg')
plt.clf()

# Plot thresholds vs FPR
i = 1
plt.figure(figsize=(10, 6))
print("Plotting thresholds vs FPR...")
for model_name, model in models.items():
    print(f"  --> Extracting thresholds for {model_name}...")

    # ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[model_name])

    # Elaboro i risultati del ROC
    fpr80, tpr80, threshold80 = find_ROC_stats(fpr, tpr, thresholds)

    # Plotto il grafico Thresholds vs FPR
    color = plt.get_cmap('tab10')( (i-1) % 10 )
    plt.plot(thresholds, fpr, label=model_name, color=color)
    plt.plot(threshold80, fpr80, marker='o', markersize=8, color=color)

    # Righe tratteggiate orizzontali
    if model_name == "ADA":
        plt.axhline(y=fpr80, xmax=threshold80, color=color, linestyle="--", alpha=0.6)
    else:
        plt.axhline(y=fpr80, xmax=threshold80+0.04, color=color, linestyle="--", alpha=0.6)

    # Righe tratteggiate verticali
    plt.axvline(x=threshold80, ymax=fpr80+0.025, color=color, linestyle="--", alpha=0.6)

    # Label dei valori dei pallini sull'asse orizzontale
    if model_name == "NB":
        plt.text(threshold80-0.01, -0.20, f"{threshold80:.3f}", rotation=45, color=color, ha="center", va="bottom", fontsize=9, fontweight="bold")
    elif model_name in {"DT", "RF", "ADA", "XGB"}:
        plt.text(threshold80-0.01, -0.15, f"{threshold80:.3f}", rotation=45, color=color, ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Label dei valori dei pallini sull'asse verticale
    if model_name == "RF":
        plt.text(-0.095, fpr80+0.02, f"{fpr80:.2f}", rotation=-45, color=color, ha="left", va="center", fontsize=9, fontweight="bold")
    elif model_name == "KNN":
        plt.text(-0.110, fpr80+0.02, f"{fpr80:.2f}", rotation=-45, color=color, ha="left", va="center", fontsize=9, fontweight="bold")
    elif model_name == "XGB":
        plt.text(-0.125, fpr80+0.02, f"{fpr80:.2f}", rotation=-45, color=color, ha="left", va="center", fontsize=9, fontweight="bold")
    elif model_name == "DT":
        plt.text(-0.140, fpr80+0.02, f"{fpr80:.2f}", rotation=-45, color=color, ha="left", va="center", fontsize=9, fontweight="bold")
    else:
        plt.text(-0.105, fpr80+0.01, f"{fpr80:.2f}", rotation=-45, color=color, ha="left", va="center", fontsize=9, fontweight="bold")
    print(f"  --> Thresholds plotted for {model_name}.")

    i += 1
# Riga verticale della threshold == 0.5
plt.axvline(x=0.5, ymax = 1, color="black", linestyle="dotted", alpha=0.6)
plt.text(0.49, -0.15, f"{0.5:.3f}", rotation=45, color="black", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.xlabel("Thresholds", labelpad=20)
plt.ylabel("FPR", labelpad=15)
plt.title("Thresholds vs FPR")
plt.legend()
plt.savefig('graphs/Thresholds_FPR.png')
plt.savefig('graphs_svg/Thresholds_FPR.svg')
plt.clf()

# Plot Precision-Recall curves
i = 1
print("Plotting Precision-Recall curves...")
for model_name, model in models.items():
    print(f"  --> Processing PR for {model_name}...")
    
    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_pred[model_name])
    pr_auc = average_precision_score(y_test, y_pred[model_name])

    plt.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.2f})")
    print(f"  --> PR plotted for {model_name} ({i}/{tot}).")
    
    i += 1
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves comparison")
plt.legend(loc='lower left')
plt.savefig('graphs/PR_comparison.png')
plt.savefig('graphs_svg/PR_comparison.svg')
plt.clf()

# Plot balanced accuracy bar chart
print("Plotting balanced accuracy bar chart...")
balanced_accuracies = model_stats.set_index('Model')['Balanced_Accuracy']
models_names = balanced_accuracies.index
values = balanced_accuracies.to_numpy(dtype=float)
plt.figure(figsize=(8, 5))
plt.bar(models_names, values, color='skyblue')
plt.ylim(0, 1.1)
plt.ylabel('Balanced Accuracy')
plt.xlabel('Models')
plt.title('Balanced Accuracy comparison')
plt.xticks(rotation=30, ha='right')
# Annotazioni sopra le barre
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('graphs/balanced_accuracy.png')
plt.savefig('graphs_svg/balanced_accuracy.svg')
plt.clf()

# Plot accuracy bar chart
print("Plotting accuracy bar chart...")
accuracies = model_stats.set_index('Model')['Accuracy']
models_names = accuracies.index
values = accuracies.to_numpy(dtype=float)
plt.figure(figsize=(8, 5))
plt.bar(models_names, values, color='skyblue')
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.title('Accuracy comparison')
plt.xticks(rotation=30, ha='right')
# Annotazioni sopra le barre
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('graphs/accuracy.png')
plt.savefig('graphs_svg/accuracy.svg')
plt.clf()

# Plot specificity bar chart
print("Plotting specificity bar chart...")
specificities = model_stats.set_index('Model')['Specificity']
models_names = specificities.index
values = specificities.to_numpy(dtype=float)
plt.figure(figsize=(8, 5))
plt.bar(models_names, values, color='skyblue')
plt.ylim(0, 1.1)
plt.ylabel('Specificity')
plt.xlabel('Models')
plt.title('Specificity comparison')
plt.xticks(rotation=30, ha='right')
# Annotazioni sopra le barre
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('graphs/specificity.png')
plt.savefig('graphs_svg/specificity.svg')
plt.clf()

# Plot precision and weighted precision bar chart (in the same graph)
print("Plotting precision and weighted precision bar chart...")
precisions = model_stats.set_index('Model')['Precision']
weighted_precisions = model_stats.set_index('Model')['Precision_weighted']
models_names = precisions.index.tolist()
values1 = precisions.to_numpy(dtype=float)
values2 = weighted_precisions.to_numpy(dtype=float)
x = np.arange(len(models_names))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values1, width, label='Precision', color='skyblue')
plt.bar(x + width/2, values2, width, label='Weighted Precision', color='lightgreen')
plt.ylim(0, 1.1)
plt.ylabel('Precision')
plt.xlabel('Models')
plt.title('Precision and Weighted Precision comparison')
plt.xticks(x, models_names, rotation=30, ha='right')
plt.legend()
# Annotazioni sopra le barre
for i in range(len(models_names)):
    plt.text(i - width/2, values1[i] + 0.01, f"{values1[i]:.3f}", ha='center', fontsize=9)
    plt.text(i + width/2, values2[i] + 0.01, f"{values2[i]:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('graphs/precision_weighted_precision.png')
plt.savefig('graphs_svg/precision_weighted_precision.svg')
plt.clf()

# Plot recall and weighted recall bar chart (in the same graph)
print("Plotting recall and weighted recall bar chart...")
recalls = model_stats.set_index('Model')['Recall']
weighted_recalls = model_stats.set_index('Model')['Recall_weighted']
models_names = recalls.index.tolist()
values1 = recalls.to_numpy(dtype=float)
values2 = weighted_recalls.to_numpy(dtype=float)
x = np.arange(len(models_names))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values1, width, label='Recall', color='skyblue')
plt.bar(x + width/2, values2, width, label='Weighted Recall', color='lightgreen')
plt.ylim(0, 1.1)
plt.ylabel('Recall')
plt.xlabel('Models')
plt.title('Recall and Weighted Recall comparison')
plt.xticks(x, models_names, rotation=30, ha='right')
plt.legend()
# Annotazioni sopra le barre
for i in range(len(models_names)):
    plt.text(i - width/2, values1[i] + 0.01, f"{values1[i]:.3f}", ha='center', fontsize=9)
    plt.text(i + width/2, values2[i] + 0.01, f"{values2[i]:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('graphs/recall_weighted_recall.png')
plt.savefig('graphs_svg/recall_weighted_recall.svg')
plt.clf()

# Plot F1 and Weighted F1 bar chart (in the same graph)
print("Plotting F1 and Weighted F1 bar chart...")
f1_scores = model_stats.set_index('Model')['F1']
weighted_f1_scores = model_stats.set_index('Model')['F1_weighted']
models_names = f1_scores.index.tolist()
values1 = f1_scores.to_numpy(dtype=float)
values2 = weighted_f1_scores.to_numpy(dtype=float)
x = np.arange(len(models_names))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values1, width, label='F1', color='skyblue')
plt.bar(x + width/2, values2, width, label='Weighted F1', color='lightgreen')
plt.ylim(0, 1.1)
plt.ylabel('F1')
plt.xlabel('Models')
plt.title('F1 and Weighted F1 comparison')
plt.xticks(x, models_names, rotation=30, ha='right')
plt.legend()
# Annotazioni sopra le barre
for i in range(len(models_names)):
    plt.text(i - width/2, values1[i] + 0.01, f"{values1[i]:.3f}", ha='center', fontsize=9)
    plt.text(i + width/2, values2[i] + 0.01, f"{values2[i]:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('graphs/f1_weighted_f1.png')
plt.savefig('graphs_svg/f1_weighted_f1.svg')
plt.clf()

print("All graphs have been generated and saved in the 'graphs' folder.")
