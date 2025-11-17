import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Il test di Shapiro-Wilk verifica se un campione proviene da una distribuzione normale.
# H0: Il campione proviene da una distribuzione normale.
# Lo utilizziamo per verificare se i risultati delle cross-validation (10 fold) di ogni modello
# seguono una distribuzione normale, requisito necessario per l'applicazione del t-test.

# df = pd.read_csv("model_results/cv_results_f1.csv")
df = pd.read_csv("model_results/cv_results_roc_auc.csv")
models = df.columns
n_models = len(models)


n_rows = (n_models + 2) // 3 
plt.figure(figsize=(18, 5 * n_rows))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for i, model_name in enumerate(models):
    scores = df[model_name]
    
    # Shapiro Test sul singolo modello
    stat, p_value = stats.shapiro(scores)
    
    # Q-Q Plot
    ax = plt.subplot(n_rows, 3, i+1)
    stats.probplot(scores, dist="norm", plot=ax)
    
    # Titolo colorato in base alla normalità
    color = "green" if p_value > 0.05 else "red"
    ax.set_title(f"{model_name}\nShapiro p={p_value:.3f}", fontsize=11, color=color, fontweight='bold')
    ax.get_lines()[0].set_markerfacecolor('blue')
    ax.get_lines()[0].set_markersize(5.0)

plt.suptitle("Q-Q Plots: Points should follow the red line (ROC AUC Scores)", fontsize=16, y=1.02)
plt.savefig("model_results/shapiro_qq_plots_roc_auc.svg", bbox_inches='tight')
plt.show()


print("\n" + "="*60)
print("ANALYSIS OF DIFFERENCES on ROC AUC Scores (Requirement for T-test)")
print("="*60)

results_diff = []

for model_a, model_b in itertools.combinations(models, 2):
    # Calcolo la differenza (vettore di 10 numeri)
    diff = df[model_a] - df[model_b]
    
    # Test di Shapiro sulla differenza
    stat, p_value = stats.shapiro(diff)
    
    is_normal = p_value > 0.05
    
    results_diff.append({
        "Model A": model_a,
        "Model B": model_b,
        "Shapiro P-value": p_value,
        "Normal?": "YES" if is_normal else "NO"
    })

# Creo un DataFrame per visualizzare bene la tabella
df_results = pd.DataFrame(results_diff)

print("RESULTS (H0: The difference is Normal)")
df_results = df_results.sort_values(by="Shapiro P-value")
print(df_results.to_string(index=False))


matrix_p = pd.DataFrame(np.nan, index=models, columns=models)

for res in results_diff:
    p = res["Shapiro P-value"]
    
    matrix_p.loc[res["Model A"], res["Model B"]] = p
    matrix_p.loc[res["Model B"], res["Model A"]] = p

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(matrix_p, dtype=bool))

sns.heatmap(matrix_p, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=0.1, 
            cbar_kws={'label': 'Shapiro P-value (Green > 0.05 = Normal)'},
            mask=mask)

plt.title("P-values of Shapiro-Wilk test on Differences between Model Pairs on ROC AUC Scores", fontsize=16)
plt.savefig("model_results/shapiro_heatmap_differences_roc_auc.svg", bbox_inches='tight')
plt.show()
