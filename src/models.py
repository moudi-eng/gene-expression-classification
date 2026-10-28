"""
Model pipelines. Imputation, feature selection, scaling and PCA are steps
INSIDE each scikit-learn Pipeline, so cross-validation refits them on the
training fold only -- the leakage control described in the report.
"""
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(cfg, n_features, min_fold_train):
    """Six pipelines from the report; sizes are capped so small datasets still run."""
    seed = cfg.seed
    k = min(cfg.n_selected_features, n_features)
    k_pca = min(cfg.pca_prefilter_features, n_features)
    n_pc = min(cfg.pca_components, k_pca, min_fold_train - 1)
    n_nb = min(cfg.knn_neighbors, min_fold_train - 1)

    def head(k_sel, scale=True):
        steps = [("impute", SimpleImputer(strategy="median")),
                 ("select", SelectKBest(f_classif, k=k_sel))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return steps

    return {
        "Logistic Regression": Pipeline(head(k) + [
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced",
                                         random_state=seed)),
        ]),
        "LASSO": Pipeline(head(k) + [
            ("model", LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                         max_iter=5000, class_weight="balanced",
                                         random_state=seed)),
        ]),
        "Elastic Net": Pipeline(head(k) + [
            ("model", LogisticRegression(penalty="elasticnet", solver="saga",
                                         l1_ratio=0.5, C=0.1, max_iter=5000,
                                         class_weight="balanced", random_state=seed)),
        ]),
        "KNN": Pipeline(head(k) + [
            ("model", KNeighborsClassifier(n_neighbors=n_nb, weights="distance")),
        ]),
        "Random Forest": Pipeline(head(k, scale=False) + [
            ("model", RandomForestClassifier(n_estimators=cfg.rf_trees, class_weight="balanced",
                                             min_samples_leaf=2, n_jobs=-1,
                                             random_state=seed)),
        ]),
        "PCA + Logistic Regression": Pipeline(head(k_pca) + [
            ("pca", PCA(n_components=n_pc, random_state=seed)),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced",
                                         random_state=seed)),
        ]),
    }
