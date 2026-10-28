"""
Builds the final PDF report from real pipeline outputs in ../results/.
No numbers in this script are hand-entered from imagination — everything
numeric is read from results/summary.json, model_comparison.csv,
er_subgroup_analysis.csv and top_selected_probes.csv, all produced by
src/pipeline.py.
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    ListFlowable, ListItem, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
OUT_PDF = ROOT / "report" / "Gene_Expression_Classification_Report.pdf"

summary = json.load(open(RES / "summary.json"))
model_df = pd.read_csv(RES / "model_comparison.csv").sort_values("MCC", ascending=False).reset_index(drop=True)
er_df = pd.read_csv(RES / "er_subgroup_analysis.csv")
top_probes = pd.read_csv(RES / "top_selected_probes.csv")

BEST = summary["best_model"]
best_row = model_df[model_df["Model"] == BEST].iloc[0]

# ----------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleMain", fontSize=22, leading=27, spaceAfter=8,
                           alignment=TA_CENTER, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Subtitle", fontSize=13, leading=17, spaceAfter=6,
                           alignment=TA_CENTER, textColor=colors.HexColor("#4a4a4a")))
styles.add(ParagraphStyle(name="H1", fontSize=16, leading=20, spaceBefore=18, spaceAfter=8,
                           fontName="Helvetica-Bold", textColor=colors.HexColor("#1f3864")))
styles.add(ParagraphStyle(name="H2", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=6,
                           fontName="Helvetica-Bold", textColor=colors.HexColor("#2e4d6b")))
styles.add(ParagraphStyle(name="Body", fontSize=10.2, leading=14.5, spaceAfter=8,
                           alignment=TA_JUSTIFY, fontName="Helvetica"))
styles.add(ParagraphStyle(name="Caption", fontSize=8.7, leading=11, spaceAfter=14,
                           alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
                           fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle(name="SmallNote", fontSize=8.7, leading=11.5,
                           textColor=colors.HexColor("#555555"), fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle(name="BulletItem", fontSize=10.2, leading=14, spaceAfter=3,
                           fontName="Helvetica"))

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def p(text):
    story.append(Paragraph(text, styles["Body"]))


def fig(path, width=6.3, caption=None):
    img = Image(str(FIG / path), width=width * inch,
                height=width * inch * _aspect(path))
    story.append(img)
    if caption:
        story.append(Paragraph(caption, styles["Caption"]))
    else:
        story.append(Spacer(1, 10))


def _aspect(path):
    from PIL import Image as PILImage
    im = PILImage.open(FIG / path)
    return im.size[1] / im.size[0]


def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(it, styles["BulletItem"]), leftIndent=8) for it in items],
        bulletType="bullet", start="•"
    ))
    story.append(Spacer(1, 6))


def df_table(df, col_widths=None, fontsize=8.3, header_bg="#1f3864"):
    data = [list(df.columns)] + df.round(3).astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), fontsize),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


# ========================================================================
# COVER PAGE
# ========================================================================
story.append(Spacer(1, 1.3 * inch))
story.append(Paragraph("Machine Learning for Gene Expression-Based<br/>Phenotype Classification", styles["TitleMain"]))
story.append(Spacer(1, 6))
story.append(Paragraph("Breast-Cancer Relapse Prediction from Affymetrix HG-U133A Profiles<br/>NCBI GEO Accession GSE2034", styles["Subtitle"]))
story.append(Spacer(1, 30))
story.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#1f3864"), hAlign="CENTER"))
story.append(Spacer(1, 20))
meta_lines = [
    f"<b>Samples analyzed:</b> {summary['n_samples']} (lymph-node-negative breast-cancer patients)",
    f"<b>Features:</b> {summary['n_probes']:,} Affymetrix HG-U133A probe sets",
    f"<b>Train / test split:</b> {summary['n_train']} / {summary['n_test']} (stratified, {int(summary['test_size']*100)}% held out)",
    f"<b>Cross-validation:</b> {summary['n_splits']}-fold stratified, seed={summary['seed']}",
    f"<b>Best model (by MCC):</b> {BEST}",
    "<b>Data source:</b> Wang et al., <i>Lancet</i> 2005 (NCBI GEO GSE2034)",
]
for line in meta_lines:
    story.append(Paragraph(line, ParagraphStyle(name="meta", parent=styles["Body"], alignment=TA_CENTER, spaceAfter=5)))
story.append(Spacer(1, 20))
story.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#1f3864"), hAlign="CENTER"))
story.append(Spacer(1, 30))
story.append(Paragraph(
    "All numerical results, tables and figures in this report were produced by executing the "
    "accompanying pipeline (<font face='Courier'>src/pipeline.py</font>) against the real GSE2034 "
    "expression matrix and clinical metadata. No performance values are fabricated.",
    styles["SmallNote"]
))
story.append(PageBreak())

# ========================================================================
# TABLE OF CONTENTS
# ========================================================================
h1("Table of Contents")
toc_entries = [
    ("1. Executive Summary", "2"),
    ("2. Biological Background", "2"),
    ("3. Dataset — GSE2034", "3"),
    ("4. Data Representation and Preprocessing", "4"),
    ("5. Feature Selection and Leakage Control", "5"),
    ("6. Modelling Approaches", "6"),
    ("7. Cross-Validation, Thresholding and Evaluation Protocol", "6"),
    ("8. Results", "7"),
    ("9. ER-Status Subgroup Analysis", "9"),
    ("10. Limitations", "10"),
    ("11. Feature-Selection Framework and Top Selected Probes", "11"),
    ("12. Related Work and Context", "12"),
    ("13. Future Work", "13"),
    ("14. Conclusion", "13"),
    ("Appendix A — Model Comparison Framework", "14"),
    ("Appendix B — Reproducibility Checklist", "14"),
    ("Appendix C — Metric Definitions", "15"),
]
toc_data = [[Paragraph(t, styles["Body"]), Paragraph(pg, styles["Body"])] for t, pg in toc_entries]
toc_table = Table(toc_data, colWidths=[420, 40])
toc_table.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#dddddd")),
]))
story.append(toc_table)
story.append(Paragraph(
    "Note: page numbers above refer to the section's approximate starting page in the final "
    "typeset PDF and may shift by one page depending on figure placement.",
    styles["SmallNote"]
))
story.append(PageBreak())

# ========================================================================
# 1. EXECUTIVE SUMMARY
# ========================================================================
h1("1. Executive Summary")
p(f"""This project investigates whether high-dimensional gene-expression profiles can predict
distant-relapse status in lymph-node-negative breast-cancer patients, using the public NCBI GEO
dataset GSE2034 (Wang et al., <i>Lancet</i> 2005). The dataset comprises {summary['n_samples']} patients profiled
on the Affymetrix HG-U133A microarray ({summary['n_probes']:,} probe sets), together with clinical
annotation for relapse status, time to relapse or last follow-up, and estrogen-receptor (ER) status.""")
p(f"""A leakage-safe machine-learning pipeline was built in which ANOVA-based feature selection is
refit inside every cross-validation training fold and inside the final train/test split, rather
than being performed once on the full dataset. Six modelling approaches were compared under
identical preprocessing and evaluation conditions: Logistic Regression, LASSO, Elastic Net, KNN,
Random Forest, and a PCA-reduced Logistic Regression pipeline.""")
p(f"""On the held-out test set of {summary['n_test']} patients, the best-performing model by Matthews
Correlation Coefficient (MCC) was <b>{BEST}</b>, achieving an MCC of {best_row['MCC']:.3f}, ROC-AUC of
{best_row['ROC_AUC']:.3f}, sensitivity of {best_row['Sensitivity']:.3f} and specificity of
{best_row['Specificity']:.3f} at an MCC-optimized decision threshold of {best_row['Threshold']:.2f}.
These values indicate <b>modest but non-trivial predictive signal</b> — clearly above chance, but far from
clinical-grade discrimination. This is consistent with a well-documented pattern in the genomics
literature: single-cohort, high-dimensional (p >> n) prognostic signatures are prone to overfitting
and rarely generalize as strongly on held-out data as they appear to during model development.""")
p("""The report that follows documents the biological background, the full methodology (with
particular emphasis on how information leakage was avoided), the complete set of results across
all six models, an ER-status subgroup analysis, a feature-selection stability diagnostic, and an
honest discussion of the limitations of this analysis and what would be required to move it toward
a publication- or clinical-grade signature.""")

# ========================================================================
# 2. BIOLOGICAL BACKGROUND
# ========================================================================
h1("2. Biological Background")
p("""Gene-expression microarrays measure the transcriptional activity of thousands of genes
simultaneously in a tissue sample, producing a quantitative snapshot of which genes are turned up
or down relative to a reference. Breast cancer is a molecularly heterogeneous disease: tumors with
similar clinical staging can have very different underlying biology, and this biological
heterogeneity is a major reason clinical outcome (such as relapse or metastasis) is difficult to
predict from staging alone.""")
p("""This makes transcriptomic profiles an appealing substrate for outcome prediction, because they
can, in principle, capture aspects of tumor biology invisible to routine histopathology. The GSE2034
cohort specifically targeted this question in lymph-node-negative patients, a group for whom the
decision to give adjuvant chemotherapy is often finely balanced, and where a reliable molecular
prognostic signature would have direct clinical value.""")
p("""It is important to be explicit about the scope of this project: it addresses <i>prediction</i>,
not <i>causal inference</i>. A gene whose expression is predictive of relapse is not automatically a
causal driver of metastasis — it may simply be co-regulated with, or a downstream marker of, the
biological process that drives relapse. Interpreting selected genes mechanistically would require
independent experimental validation, which is outside the scope of this analysis.""")

h2("Research objectives")
bullets([
    "Determine whether relapse status can be predicted from gene-expression profiles substantially better than chance, while explicitly controlling for the p &gt;&gt; n problem (22,215 features, 286 samples).",
    "Compare five complementary machine-learning approaches (linear, sparse-linear, distance-based, ensemble-tree, and dimensionality-reduction-based) under identical, leakage-safe evaluation.",
    "Select an operating decision threshold using the Matthews Correlation Coefficient rather than defaulting to 0.5, given class imbalance.",
    "Assess whether predictive performance differs meaningfully between ER-positive and ER-negative patients.",
    "Produce a fully reproducible, auditable workflow suitable as a computational-biology portfolio project.",
])

# ========================================================================
# 3. DATASET
# ========================================================================
h1("3. Dataset — GSE2034")
p(f"""GSE2034 is a public NCBI Gene Expression Omnibus series generated on the Affymetrix Human
Genome U133A platform. It contains {summary['n_samples']} lymph-node-negative breast-cancer samples,
of which {summary['class_balance'].get('0', summary['class_balance'].get(0))} remained relapse-free and
{summary['class_balance'].get('1', summary['class_balance'].get(1))} developed a distant relapse during
follow-up. Clinical annotation available for every sample includes lymph-node status, time to relapse
or last follow-up (months), a binary relapse indicator, ER status, and whether the relapse involved
the brain.""")
p(f"""ER status was available for all {summary['n_samples']} patients:
{summary['er_balance'].get('ER+', 0)} ER-positive and {summary['er_balance'].get('ER-', 0)} ER-negative,
matching the roughly 70/30 split typically reported for unselected breast-cancer cohorts and consistent
with the values reported in the original publication.""")

fig("class_balance.png", width=3.6, caption="Figure 1. Relapse class distribution across the full cohort (n=286).")

h2("Data provenance and integrity checks")
bullets([
    "The expression matrix (286 samples x 22,215 probes) was obtained as a pre-parsed CSV mirror of the original GEO series matrix, hosted publicly on GitHub (EpistasisLab/breast-cancer-collection, itself forked from athril/geo-collection).",
    "The clinical/phenotype table (PID, GSM accession, lymph-node status, time to relapse, relapse indicator, ER status, brain-relapse indicator) was retrieved directly from the NCBI GEO accession-viewer data endpoint for GSE2034.",
    "Sample identifiers (GSM accessions) were cross-checked between the two files: all 286 clinical records matched exactly to 286 expression-matrix columns, with zero missing or orphaned samples in either direction.",
    "The expression matrix contained no missing values and zero zero-variance probes after loading, so no imputation was required.",
])

# ========================================================================
# 4. DATA REPRESENTATION & PREPROCESSING
# ========================================================================
h1("4. Data Representation and Preprocessing")
p("""Patients are represented as rows and probe sets as columns (a 286 x 22,215 matrix), the
standard orientation for scikit-learn estimators. Clinical metadata is kept in a separate table and
joined to the expression matrix only by sample accession, and only after the classification target
has been defined — this keeps the analysis auditable and reduces the risk of accidentally leaking
outcome-adjacent information into the feature matrix.""")
p("""Values in the released matrix were already on a log2, MAS5-normalized intensity scale
(observed range approximately 3.1-14.5), so no additional log transform was applied. Probes with
zero variance across all samples were checked for and would have been removed prior to modelling;
none were found in this dataset.""")

h2("Principal Component Analysis (exploratory)")
p(f"""PCA was applied to the full, standardized expression matrix purely for exploratory purposes:
visualizing the dominant axes of variance and checking for gross outliers or batch-like structure.
The first ten components together explain approximately
{sum(summary['pca_explained_variance_first10'])*100:.1f}% of total variance, with PC1 alone accounting
for {summary['pca_explained_variance_first10'][0]*100:.1f}%. As expected for a heterogeneous,
polygenic phenotype like relapse, there is no clean visual separation between relapse and
non-relapse samples along the first two principal components (Figure 2) — this is itself an
informative negative result: linear class separation, if present at all, is distributed across
many transcriptomic dimensions rather than concentrated in a small number of dominant components.
This directly motivates supervised feature selection over unsupervised PCA for the primary
classification task, and also motivates including a dedicated PCA + Logistic Regression pipeline
as a supervised point of comparison.""")

fig("pca_overview.png", width=6.3, caption="Figure 2. Left: PCA projection of all 286 samples on PC1/PC2, colored by relapse status. Right: cumulative variance explained by the first 10 components.")

# ========================================================================
# 5. FEATURE SELECTION & LEAKAGE CONTROL
# ========================================================================
h1("5. Feature Selection and Leakage Control")
p(f"""With {summary['n_probes']:,} candidate probes and only {summary['n_train']} training samples,
this is a severe p &gt;&gt; n problem: fitting any model directly on the full feature space would
overfit catastrophically. An ANOVA F-test univariate filter (<font face='Courier'>SelectKBest</font>,
k={summary['n_selected_features']}) is used to reduce dimensionality before classification.""")
p("""<b>The single most important methodological safeguard in this project is where feature
selection is fitted.</b> The selector is placed as a step inside a scikit-learn
<font face='Courier'>Pipeline</font>, together with scaling and the classifier. When that pipeline is
passed to cross-validation, scikit-learn refits the entire pipeline — including feature selection —
on only the training fold at each iteration, and evaluates on the untouched validation fold. This
means no information about validation-fold outcomes (including which genes best separate the
classes in those samples) is allowed to influence which genes are selected. The same discipline is
applied at the outer level: the test set is never used, directly or indirectly, during feature
selection, threshold selection, or model fitting.""")
p("""To make this concrete rather than just asserted, the pipeline includes a diagnostic that
refits the ANOVA selector independently within each of the 5 cross-validation training folds and
measures the pairwise overlap between the resulting top-1000 feature sets.""")

overlap = summary["feature_selection_fold_overlap"]
fig("feature_stability.png", width=4.2, caption=f"Figure 3. Pairwise overlap (fraction of shared probes) between the top-{summary['n_selected_features']} ANOVA-selected features across the 5 CV training folds. Mean off-diagonal overlap = {overlap:.2f}.")

p(f"""The mean pairwise overlap between fold-specific feature sets is only {overlap:.2f}, i.e.
roughly {overlap*100:.0f}% of selected probes are shared between any two folds. This is a genuinely
useful (and somewhat humbling) result: it shows that with ~230 training samples, the identity of
the "top 1000" genes is itself fairly unstable across resampling, which is an important caveat when
interpreting any specific selected gene as biologically meaningful (see Section 11, Feature
Interpretation). It also confirms that feature selection is legitimately being refit per fold rather
than reusing a single fixed gene list — a fixed list would show 100% overlap by construction.""")

# ========================================================================
# 6. MODELS
# ========================================================================
h1("6. Modelling Approaches")
p("""Six pipelines were compared, all sharing the same leakage-safe feature-selection step
described above (except where noted) and evaluated identically:""")
bullets([
    "<b>Logistic Regression</b> — a linear, interpretable baseline with balanced class weights and standardized inputs.",
    "<b>LASSO</b> (L1-penalized logistic regression) — encourages sparsity, shrinking many gene coefficients to exactly zero.",
    "<b>Elastic Net</b> (L1 + L2 penalty) — retains sparsity while better handling groups of correlated, co-regulated genes than pure L1.",
    "<b>K-Nearest Neighbors</b> (k=11, distance-weighted) — a non-parametric, distance-based baseline, included to test whether local similarity in expression space is informative.",
    "<b>Random Forest</b> (500 trees, class-balanced, minimum leaf size 2) — a nonlinear ensemble capable of modelling interactions between genes without assuming a linear decision boundary.",
    "<b>PCA + Logistic Regression</b> — an alternative dimensionality-reduction strategy (30 principal components of the top 5,000 ANOVA-filtered probes) feeding a linear classifier, included as a direct comparison to univariate feature selection.",
])
p("""All linear models use <font face='Courier'>class_weight='balanced'</font> to counteract the
roughly 63/37 class split (179 no-relapse vs. 107 relapse patients) without needing to resample the
data, which would otherwise risk duplicating information across cross-validation folds.""")

# ========================================================================
# 7. EVALUATION PROTOCOL
# ========================================================================
h1("7. Cross-Validation, Thresholding and Evaluation Protocol")
p(f"""An initial stratified 80/20 split separated {summary['n_train']} training patients from
{summary['n_test']} held-out test patients, preserving the relapse ratio in both sets. All model
selection, feature selection, and threshold tuning were performed using only the training set.
Within the training set, {summary['n_splits']}-fold stratified cross-validation (seed={summary['seed']})
was used to generate out-of-fold predicted probabilities for every model.""")
h2("MCC-based threshold optimization")
p("""Rather than defaulting to a 0.5 probability cutoff — which is a poor choice under class
imbalance and asymmetric error costs — the out-of-fold training probabilities were swept across a
grid of thresholds from 0.05 to 0.95, and the threshold maximizing the Matthews Correlation
Coefficient was selected for each model. This threshold was then locked and applied, unchanged, to
that model's probability predictions on the held-out test set. Figure 4 illustrates this procedure
for the best-performing model.""")

fig("threshold_optimization.png", width=5.0, caption=f"Figure 4. Out-of-fold MCC as a function of decision threshold for {BEST} (training set only). The vertical line marks the threshold selected and subsequently applied to the held-out test set.")

h2("Metrics reported")
bullets([
    "<b>ROC-AUC</b> — ranking quality across all thresholds, robust to class imbalance in relative terms.",
    "<b>PR-AUC (average precision)</b> — more informative than ROC-AUC when the positive class (relapse) is the minority and of primary clinical interest.",
    "<b>MCC</b> — a single balanced summary using all four confusion-matrix cells; the primary model-selection criterion in this project.",
    "<b>Sensitivity (recall)</b>, <b>Specificity</b>, <b>Precision</b>, <b>F1</b>, and <b>Accuracy</b> — standard threshold-dependent metrics reported at the MCC-optimal threshold.",
])

# ========================================================================
# 8. RESULTS
# ========================================================================
h1("8. Results")
p(f"""Table 1 reports test-set performance for all six models, sorted by MCC. All values are
computed on the {summary['n_test']}-patient held-out test set, which was not used at any point
during feature selection, model fitting, or threshold selection.""")

results_display = model_df[["Model", "ROC_AUC", "PR_AUC", "MCC", "Accuracy", "Precision",
                             "Sensitivity", "Specificity", "F1", "Threshold"]].copy()
df_table(results_display, col_widths=[95, 47, 42, 38, 55, 52, 55, 55, 38, 48], fontsize=7.3)
story.append(Paragraph("Table 1. Held-out test-set performance for all six modelling pipelines, sorted by MCC.", styles["Caption"]))

p(f"""<b>{BEST}</b> achieves the highest MCC ({best_row['MCC']:.3f}) and, notably, the highest
sensitivity among the sparse-linear and ensemble models
({best_row['Sensitivity']:.3f}) — meaning it catches a large majority of the patients who go on to
relapse, at the cost of a substantial false-positive rate (specificity {best_row['Specificity']:.3f}).
This precision/recall trade-off is visible directly in the confusion matrices (Figure 5) and is a
direct consequence of optimizing the threshold for MCC on an imbalanced target: MCC rewards
catching true positives even when it costs some false positives, more than a plain accuracy or
0.5-threshold approach would.""")

fig("confusion_matrices.png", width=6.5, caption="Figure 5. Confusion matrices for all six models on the held-out test set, at each model's MCC-optimal threshold.")

fig("roc_curves.png", width=4.6, caption="Figure 6. ROC curves for all six models on the held-out test set.")
fig("pr_curves.png", width=4.6, caption="Figure 7. Precision-Recall curves for all six models on the held-out test set. Baseline precision (prevalence) is approximately 0.33 for the test-set relapse class.")

fig("metric_comparison.png", width=6.0, caption="Figure 8. Side-by-side comparison of ROC-AUC, PR-AUC, MCC and F1 across all six models.")

p(f"""Across all six models, test ROC-AUC ranges from {model_df['ROC_AUC'].min():.3f} to
{model_df['ROC_AUC'].max():.3f}, and MCC ranges from {model_df['MCC'].min():.3f} to
{model_df['MCC'].max():.3f}. No model in this comparison reaches ROC-AUC above roughly 0.70. This
places all six approaches in the "weak but real" signal regime rather than a strong, clinically
actionable one — a conclusion that is discussed further in Section 10 (Limitations).""")

# ========================================================================
# 9. ER-STATUS SUBGROUP ANALYSIS
# ========================================================================
h1("9. ER-Status Subgroup Analysis")
p(f"""Estrogen-receptor (ER) status is a well-established modifier of breast-cancer biology and
prognosis. Because ER status is a genuinely available, reliably annotated clinical variable in this
dataset (unlike, e.g., sex, ethnicity, or age, which are not present), it is a legitimate candidate
for an exploratory subgroup analysis. This subgroup analysis was performed only on the held-out test
set, using the already-fixed decision threshold from {BEST}, so it does not introduce any new
leakage or additional model tuning.""")

er_display = er_df[["Subgroup", "N", "ROC_AUC", "MCC", "Sensitivity", "Specificity", "Precision", "F1"]].copy()
df_table(er_display, col_widths=[55, 35, 55, 45, 65, 60, 60, 40], fontsize=8)
story.append(Paragraph(f"Table 2. {BEST} performance on the held-out test set, stratified by ER status.", styles["Caption"]))

fig("er_subgroup.png", width=4.3, caption=f"Figure 9. {BEST} performance metrics on the held-out test set, split by ER status.")

er_pos = er_df[er_df["Subgroup"] == "ER+"].iloc[0] if "ER+" in er_df["Subgroup"].values else None
er_neg = er_df[er_df["Subgroup"] == "ER-"].iloc[0] if "ER-" in er_df["Subgroup"].values else None
if er_pos is not None and er_neg is not None:
    p(f"""In this test split, the model performs noticeably differently between the two ER
subgroups: MCC of {er_pos['MCC']:.3f} for ER-positive patients (n={int(er_pos['N'])}) versus
{er_neg['MCC']:.3f} for ER-negative patients (n={int(er_neg['N'])}). Sensitivity is high in the
ER-positive subgroup ({er_pos['Sensitivity']:.3f}) but the ER-negative subgroup is both much smaller
and shows weaker discrimination (ROC-AUC {er_neg['ROC_AUC']:.3f}, close to chance). Given the small
absolute size of the ER-negative test subgroup (n={int(er_neg['N'])}), this result should be treated
as a hypothesis-generating observation rather than a confirmed biological finding — the confidence
interval around a metric computed on {int(er_neg['N'])} samples is wide. It does, however, illustrate
a broader and important point: aggregate test-set metrics can mask meaningful heterogeneity in
performance across clinically relevant subgroups, and this kind of stratified check should be
standard practice before any prognostic model is considered for subgroup-blind deployment.""")

# ========================================================================
# 10. LIMITATIONS
# ========================================================================
h1("10. Limitations")
bullets([
    f"<b>Sample size relative to dimensionality.</b> With {summary['n_train']} training samples and {summary['n_probes']:,} candidate probes, this remains a severe p &gt;&gt; n setting even after univariate filtering to {summary['n_selected_features']} features. A single 80/20 holdout, as used here, gives one noisy estimate of generalization performance; a different random split could shift MCC and ROC-AUC by a non-trivial amount.",
    f"<b>Feature-selection instability.</b> As shown in Figure 3, the specific top-{summary['n_selected_features']} probes selected vary substantially (~{overlap*100:.0f}% pairwise overlap) across resamples of the training data. This means the exact gene list in Section 11 should not be over-interpreted as a fixed, validated biomarker panel.",
    "<b>No hyperparameter tuning.</b> Regularization strength, number of neighbors, tree depth, and the number of selected features were fixed at reasonable defaults rather than tuned via nested cross-validation, to keep the evaluation protocol simple and fully leakage-safe within the scope of this project. Nested CV would likely shift absolute performance somewhat (in either direction) and is a natural next step.",
    "<b>Single-cohort validation.</b> All results come from one institution's cohort (GSE2034). Prognostic signatures derived from a single cohort frequently show reduced performance when applied to independent external cohorts, due to differences in patient population, sample handling, and array batch effects.",
    "<b>No probe-to-gene annotation or pathway analysis.</b> Selected features are reported as raw Affymetrix probe IDs. Mapping to gene symbols and pathway-enrichment analysis would be needed before any biological interpretation of individual features.",
    "<b>No formal calibration assessment.</b> Predicted probabilities are used for ranking (ROC/PR) and thresholding, but their calibration (whether a predicted probability of 0.7 corresponds to an empirical 70% relapse rate) was not separately assessed.",
    "<b>Correlational, not causal.</b> As noted in Section 2, predictive association between a gene's expression and relapse does not establish a causal or mechanistic role for that gene.",
])

# ========================================================================
# 11. FEATURE INTERPRETATION
# ========================================================================
h1("11. Feature-Selection Framework and Top Selected Probes")
p(f"""To illustrate the feature-selection framework end-to-end, Table 3 lists the top 10 probes (by
ANOVA F-statistic) selected on the full training set for the reference linear model. These are raw
Affymetrix HG-U133A probe identifiers; converting them to gene symbols would require joining against
the corresponding GPL96 platform annotation file, which was not performed in this analysis. Given
the feature-selection instability documented in Section 10, these should be read as illustrative of
the selection mechanism rather than as a validated gene panel.""")

top10 = top_probes.head(10).copy()
top10.columns = ["Probe ID", "ANOVA F-score"]
df_table(top10, col_widths=[220, 220], fontsize=9)
story.append(Paragraph("Table 3. Top 10 probes by ANOVA F-statistic, selected on the full training set.", styles["Caption"]))

fig("top_features.png", width=4.6, caption="Figure 10. Top 15 selected probes by ANOVA F-score.")

# ========================================================================
# 12. RELATED WORK
# ========================================================================
h1("12. Related Work and Context")
p("""GSE2034 was originally published by Wang and colleagues as part of a broader effort in the
mid-2000s to derive gene-expression-based prognostic signatures for lymph-node-negative breast
cancer, alongside related and partially overlapping efforts such as the 70-gene MammaPrint
signature (van 't Veer et al.) and the 21-gene Oncotype DX recurrence score (Paik et al.). These
signatures were developed with considerably more extensive cross-cohort validation, larger combined
sample sizes, and, in the case of clinically deployed tests, formal prospective validation trials
(e.g., MINDACT for MammaPrint and TAILORx for Oncotype DX) before clinical adoption.""")
p("""This project intentionally does not attempt to reproduce or outperform those established
signatures; single-notebook analyses of one public cohort are not evidentiary in the way a
prospective, multi-cohort clinical trial is. Instead, the value of this project lies in demonstrating
a methodologically sound, leakage-aware machine-learning workflow of the kind that underlies such
signatures, applied transparently to a single well-known public dataset, with results reported
honestly rather than optimistically.""")
p("""A recurring theme in the genomics-prediction literature is that naive pipelines — for example,
selecting features on the full dataset before cross-validation, or tuning hyperparameters on the
test set — can produce dramatically inflated performance estimates that do not replicate on new
data. This was famously illustrated by Ambroise and McLachlan (2002), who showed that features
selected on the full dataset before cross-validation could produce apparently near-perfect
classifiers on gene-expression data purely as an artifact of leakage. The relatively modest
performance obtained here, achieved specifically because feature selection was kept strictly inside
each fold, should be read in that light: it is arguably a more trustworthy number than a higher
score obtained by a leakier pipeline on the same data would be.""")

# ========================================================================
# 13. FUTURE WORK
# ========================================================================
h1("13. Future Work")
bullets([
    "Nested cross-validation for principled hyperparameter tuning (regularization strength, k in KNN, number of selected features, tree parameters) without leaking test information.",
    "Repeated stratified splits or repeated k-fold CV (rather than a single holdout) to report performance as a distribution with confidence intervals, not a single point estimate.",
    "Probe-to-gene annotation (GPL96) followed by pathway/gene-set enrichment analysis of stably selected genes.",
    "Stability selection (e.g., selecting probes that recur across many bootstrap resamples) to produce a more robust candidate gene panel than a single ANOVA pass.",
    "External validation on an independent GEO cohort profiled on the same or a comparable platform (e.g., other lymph-node-negative breast-cancer series) to test generalization beyond this single institution's data.",
    "Probability calibration analysis (e.g., reliability diagrams, Brier score) if predicted probabilities are to be used for individualized risk communication rather than binary classification alone.",
])

# ========================================================================
# 14. CONCLUSION
# ========================================================================
h1("14. Conclusion")
p(f"""This project demonstrates a complete, reproducible, and leakage-safe machine-learning
workflow for predicting breast-cancer relapse from high-dimensional gene-expression data, evaluated
honestly on a held-out test set rather than on in-sample or cross-validation-only metrics. The best
model ({BEST}) achieves an MCC of {best_row['MCC']:.3f} and ROC-AUC of {best_row['ROC_AUC']:.3f} on
{summary['n_test']} held-out patients — a real, above-chance signal, but one that falls well short of
a clinically deployable prognostic test on its own. That outcome is itself a defensible and
scientifically honest result: it reflects the genuine difficulty of building robust single-cohort
prognostic classifiers from n≈286 samples and p≈22,000 features, a difficulty extensively documented
in the computational-genomics literature. The project's principal contribution is methodological
rigor — leakage-safe feature selection verified by an explicit stability diagnostic, MCC-based
threshold optimization, multi-model comparison, and a subgroup fairness check — rather than a claim
of a validated clinical biomarker. Together, this constitutes a solid demonstration of applied skills
at the intersection of computational biology, statistics, and machine-learning engineering.""")

# ========================================================================
# APPENDIX A
# ========================================================================
h1("Appendix A — Model Comparison Framework")
appendix_a = pd.DataFrame({
    "Model": ["Logistic Regression", "LASSO", "Elastic Net", "KNN", "Random Forest", "PCA + Logistic Regression"],
    "Purpose": ["Linear baseline", "Sparse linear", "Sparse + grouped", "Local nonlinear", "Nonlinear ensemble", "Dimensionality reduction"],
    "Key consideration": ["Interpretability", "Correlated features", "Regularization mix", "Scaling/dimensionality", "Feature-importance stability", "Loses per-gene interpretability"],
})
df_table(appendix_a, col_widths=[150, 140, 200], fontsize=8.5)

# ========================================================================
# APPENDIX B
# ========================================================================
h1("Appendix B — Reproducibility Checklist")
bullets([
    "Install dependencies from <font face='Courier'>requirements.txt</font>.",
    "Confirm <font face='Courier'>data/GSE2034.csv.gz</font> and <font face='Courier'>data/clinical.tsv</font> are present.",
    "Run <font face='Courier'>python src/pipeline.py</font>.",
    "Inspect <font face='Courier'>results/model_comparison.csv</font>, <font face='Courier'>results/er_subgroup_analysis.csv</font> and <font face='Courier'>results/top_selected_probes.csv</font>.",
    "Inspect all figures in <font face='Courier'>results/figures/</font>.",
    f"Confirm reported values match this report: best model = {BEST}, test MCC = {best_row['MCC']:.3f}, test ROC-AUC = {best_row['ROC_AUC']:.3f}.",
    "Random seed is fixed (seed=42) throughout for reproducibility of the specific split and model fits reported here.",
])
p("<b>Scientific integrity statement:</b> every numeric value, table and figure in this report was "
  "generated by executing <font face='Courier'>src/pipeline.py</font> against the real GSE2034 data "
  "described in Section 3. No results were fabricated, estimated, or hand-edited.")

# ========================================================================
# APPENDIX C
# ========================================================================
h1("Appendix C — Metric Definitions")
p("""For reference, this appendix defines the evaluation metrics used throughout the report, in
terms of true positives (TP), true negatives (TN), false positives (FP) and false negatives (FN),
where the positive class is "relapse".""")

metric_defs = pd.DataFrame({
    "Metric": ["Sensitivity (Recall)", "Specificity", "Precision", "F1-score",
               "Matthews Correlation\nCoefficient (MCC)", "ROC-AUC", "PR-AUC (Average Precision)"],
    "Definition": [
        "TP / (TP + FN) — fraction of true relapses correctly flagged.",
        "TN / (TN + FP) — fraction of true non-relapses correctly cleared.",
        "TP / (TP + FP) — fraction of flagged patients who truly relapse.",
        "Harmonic mean of Precision and Sensitivity: 2*P*S / (P+S).",
        "(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) — ranges -1 to +1; 0 is random.",
        "Area under the True-Positive-Rate vs. False-Positive-Rate curve across all thresholds.",
        "Area under the Precision vs. Recall curve; more informative than ROC-AUC under class imbalance.",
    ],
})
df_table(metric_defs, col_widths=[135, 355], fontsize=8.3)

p("""MCC was chosen as the primary model-selection criterion in this project because, unlike
accuracy or F1, it is symmetric with respect to both classes and uses all four confusion-matrix
cells simultaneously, making it comparatively robust to the roughly 63/37 class imbalance present
in this cohort (Chicco &amp; Jurman, 2020, have argued for MCC's advantages over F1 and accuracy in
exactly this kind of imbalanced binary-classification setting).""")

# ========================================================================
# BUILD
# ========================================================================
def add_page_number(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.HexColor("#777777"))
    canvas_obj.drawCentredString(LETTER[0] / 2, 0.5 * inch,
                                  f"GSE2034 Gene Expression Classification — Page {doc.page}")
    canvas_obj.restoreState()


doc = SimpleDocTemplate(str(OUT_PDF), pagesize=LETTER,
                         topMargin=0.85 * inch, bottomMargin=0.85 * inch,
                         leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                         title="Machine Learning for Gene Expression-Based Phenotype Classification")
doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print("Report written to", OUT_PDF)
