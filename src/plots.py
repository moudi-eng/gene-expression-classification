"""All figures. Each function writes one PNG and closes its figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from evaluation import THRESHOLD_GRID, mcc_curve

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
CLASS_COLORS = ["#3B7DD8", "#D8553B"]


def class_balance(y, class_names, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    labels = [f"{class_names[0]} (0)", f"{class_names[1]} (1)"]
    ax.bar(labels, counts.values, color=CLASS_COLORS)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Class distribution (n={counts.sum()})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pca_overview(X, y, class_names, seed, path):
    """Exploratory, unsupervised PCA of the full matrix (labels only used for colour)."""
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    n_pc = min(10, *Xs.shape)
    pca = PCA(n_components=n_pc, random_state=seed)
    scores = pca.fit_transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for label in (0, 1):
        mask = y.values == label
        ax.scatter(scores[mask, 0], scores[mask, 1], s=18, alpha=0.7,
                   color=CLASS_COLORS[label], label=class_names[label], edgecolor="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA of full expression matrix")
    ax.legend(frameon=False)

    cum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, n_pc + 1), cum, marker="o", color=CLASS_COLORS[0])
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("PCA scree (cumulative)")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return pca.explained_variance_ratio_


def feature_stability(overlap, path):
    n = len(overlap)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(overlap, vmin=0, vmax=1, cmap="viridis")
    ticks = [f"Fold {i+1}" for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(ticks, fontsize=8)
    ax.set_title("Feature-set overlap across CV folds")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def roc_pr(curves, path_roc, path_pr):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, d in curves.items():
        ax.plot(d["fpr"], d["tpr"], label=f"{name} (AUC={d['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — held-out test set")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path_roc)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, d in curves.items():
        ax.plot(d["recall"], d["precision"], label=f"{name} (AP={d['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — held-out test set")
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path_pr)
    plt.close(fig)


def confusion_matrices(cms, class_names, path):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(class_names, fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def metric_bars(df, path):
    metrics = ["ROC_AUC", "PR_AUC", "MCC", "F1"]
    colors = ["#3B7DD8", "#D8553B", "#3BAA6E", "#C79A3B"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    width = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m], width, label=m, color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(min(0, df[metrics].min().min()), 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — held-out test set")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def threshold_curve(y_oof, p_oof, best_t, model_name, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(THRESHOLD_GRID, mcc_curve(y_oof, p_oof), color=CLASS_COLORS[0])
    ax.axvline(best_t, color=CLASS_COLORS[1], linestyle="--",
               label=f"Selected threshold = {best_t:.2f}")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("MCC (out-of-fold, training set)")
    ax.set_title(f"MCC-based threshold optimization — {model_name}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def subgroups(df, model_name, subgroup_column, path):
    metrics = ["MCC", "Sensitivity", "Specificity", "F1"]
    fig, ax = plt.subplots(figsize=(5.5 + 0.5 * len(df), 4))
    xpos = np.arange(len(metrics))
    width = 0.8 / len(df)
    for i, row in df.reset_index(drop=True).iterrows():
        ax.bar(xpos + i * width, row[metrics].astype(float), width,
               label=f"{row['Subgroup']} (n={row['N']})")
    ax.set_xticks(xpos + width * (len(df) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(min(0, df["MCC"].min()), 1)
    ax.set_title(f"{subgroup_column} subgroup performance — {model_name}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def top_features(top, model_name, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    top15 = top.head(15).iloc[::-1]
    ax.barh(top15["Label"], top15["ANOVA_F_score"], color=CLASS_COLORS[0])
    ax.set_xlabel("ANOVA F-score (training set)")
    ax.set_title(f"Top 15 selected features — {model_name}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
