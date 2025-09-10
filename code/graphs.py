import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import paths

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
    cm = eval(row['Confusion Matrix'])

    plt.figure(figsize=(6, 4))
    plt.matshow(cm, cmap='Blues', alpha=0.7)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
    plt.yticks([0, 1], ['Non-Fraud', 'Fraud'])
    for (i, j), value in np.ndenumerate(cm):
        plt.text(j, i, f'{value}', ha='center', va='center', fontsize=12)
    
    plt.savefig(f'graphs/confusion_matrices/{model_name}_cm.png')
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
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves comparison")
plt.legend()
plt.savefig('graphs/ROC_comparison.png')
plt.clf()

#Plot thresholds vs TPR
i = 1
print("Plotting thresholds vs TPR...")
for model_name, model in models.items():
    print(f"  --> Extracting thresholds for {model_name}...")

    # ROC
    _, tpr, thresholds = roc_curve(y_test, y_pred[model_name])

    plt.plot(thresholds, tpr, label='TPR', color='blue') # type: ignore
    print(f"  --> Thresholds extracted for {model_name}.")

    i += 1
plt.figure(figsize=(10, 6))
plt.xlabel("Thresholds")
plt.ylabel("Rate")
plt.title("Thresholds vs TPR")
plt.legend()
plt.savefig('graphs/Thresholds_TPR.png')
plt.clf()

# Plot thresholds vs FPR
i = 1
print("Plotting thresholds vs FPR...")
for model_name, model in models.items():
    print(f"  --> Extracting thresholds for {model_name}...")

    # ROC
    fpr, _, thresholds = roc_curve(y_test, y_pred[model_name])

    plt.plot(thresholds, fpr, label='FPR', color='red') # type: ignore
    print(f"  --> Thresholds extracted for {model_name}.")

    i += 1
plt.figure(figsize=(10, 6))
plt.xlabel("Thresholds")
plt.ylabel("Rate")
plt.title("Thresholds vs FPR")
plt.legend()
plt.savefig('graphs/Thresholds_FPR.png')
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
plt.clf()

print("All graphs have been generated and saved in the 'graphs' folder.")
