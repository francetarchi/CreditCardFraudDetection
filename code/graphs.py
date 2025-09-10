import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


### RESOURCE LOADING ###
# Caricamento del test set preprocessato (sbilanciato)
print("Loading preprocessed testing set...")
test_set = pd.read_csv(
    # "C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\prep_test.csv"
    "C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Dataset\\prep_test.csv"
)
y_test = test_set["isFraud"]
X_test = test_set.drop(columns=["isFraud"])

# Caricamento dei modelli salvati
print("Loading trained models...")
model_paths = {
    "KNN": f"C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Trained models\\smote20.0\\KNN.pkl",
    "NB": f"C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Trained models\\smote20.0\\NaiveBayes.pkl",
    "DT": f"C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Trained models\\smote20.0\\DecisionTree.pkl",
    "RF": f"C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Trained models\\smote20.0\\RandomForest.pkl",
    "ADA": f"C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Trained models\\smote20.0\\AdaBoost.pkl",
    "XGB": f"C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Trained models\\smote20.0\\XGBoost.pkl"
}
models = {}
for name, path in model_paths.items():
    models[name] = joblib.load(path)

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
print("Plotting ROC curves...")
for model_name, model in models.items():
    # Probabilità della classe positiva
    y_pred = model.predict_proba(X_test)[:, 1]
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves comparison")
plt.legend()
plt.savefig('graphs/ROC_comparison.png')
plt.clf()

# Plot Precision-Recall curves
print("Plotting Precision-Recall curves...")
for model_name, model in models.items():
    # Probabilità della classe positiva
    y_pred = model.predict_proba(X_test)[:, 1]
    
    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    
    plt.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves comparison")
plt.legend()
plt.savefig('graphs/PR_comparison.png')
plt.clf()

# Plot balanced accuracy bar chart
print("Plotting balanced accuracy bar chart...")
balanced_accuracies = model_stats.set_index('Model')['Balanced Accuracy']
models_names = balanced_accuracies.index
values = balanced_accuracies.to_numpy(dtype=float)
plt.figure(figsize=(8, 5))
plt.bar(models_names, values, color='skyblue')
plt.ylim(0, 1)
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
plt.ylim(0, 1)
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
plt.ylim(0, 1)
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
weighted_precisions = model_stats.set_index('Model')['Weighted Precision']
models_names = precisions.index.tolist()
values1 = precisions.to_numpy(dtype=float)
values2 = weighted_precisions.to_numpy(dtype=float)
x = np.arange(len(models_names))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values1, width, label='Precision', color='skyblue')
plt.bar(x + width/2, values2, width, label='Weighted Precision', color='lightgreen')
plt.ylim(0, 1)
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
weighted_recalls = model_stats.set_index('Model')['Weighted Recall']
models_names = recalls.index.tolist()
values1 = recalls.to_numpy(dtype=float)
values2 = weighted_recalls.to_numpy(dtype=float)
x = np.arange(len(models_names))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values1, width, label='Recall', color='skyblue')
plt.bar(x + width/2, values2, width, label='Weighted Recall', color='lightgreen')
plt.ylim(0, 1)
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

# Plot F1-Score and Weighted F1-Score bar chart (in the same graph)
print("Plotting F1-Score and Weighted F1-Score bar chart...")
f1_scores = model_stats.set_index('Model')['F1-Score']
weighted_f1_scores = model_stats.set_index('Model')['Weighted F1-Score']
models_names = f1_scores.index.tolist()
values1 = f1_scores.to_numpy(dtype=float)
values2 = weighted_f1_scores.to_numpy(dtype=float)
x = np.arange(len(models_names))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values1, width, label='F1-Score', color='skyblue')
plt.bar(x + width/2, values2, width, label='Weighted F1-Score', color='lightgreen')
plt.ylim(0, 1)
plt.ylabel('F1-Score')
plt.xlabel('Models')
plt.title('F1-Score and Weighted F1-Score comparison')
plt.xticks(x, models_names, rotation=30, ha='right')
plt.legend()
# Annotazioni sopra le barre
for i in range(len(models_names)):
    plt.text(i - width/2, values1[i] + 0.01, f"{values1[i]:.3f}", ha='center', fontsize=9)
    plt.text(i + width/2, values2[i] + 0.01, f"{values2[i]:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('graphs/f1_score_weighted_f1_score.png')
plt.clf()

print("All graphs have been generated and saved in the 'graphs' folder.")
