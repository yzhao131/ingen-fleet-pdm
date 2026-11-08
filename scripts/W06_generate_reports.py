
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(".")
W06 = ROOT / "reports/W06"
W03 = ROOT / "reports/W03"
W04 = ROOT / "reports/W04"
WEEKLY = ROOT / "weekly"

WEEKLY.mkdir(exist_ok=True)

def read_csv(folder, filename):
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return pd.read_csv(path)

clusters = read_csv(W06, "W06_cluster_risk_summary.csv")
profiles = read_csv(W06, "W06_cluster_profiles_k3.csv")
sites = read_csv(W06, "W06_site_summary.csv")
regression = read_csv(W06, "W06_site_regression_results.csv")
stability = read_csv(W06, "W06_resampling_stability.csv")
evaluation = read_csv(W06, "W06_cluster_evaluation.csv")
variance = read_csv(W06, "W06_pca_variance.csv")
w03 = read_csv(W03, "W03_signal_analysis.csv")
shap = read_csv(W04, "W04_SHAP_Importance.csv")
permutation = read_csv(W04, "W04_permutation_importance.csv")

def setup_doc(title, subtitle):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(0.75)
    sec.left_margin = Inches(0.85)
    sec.right_margin = Inches(0.85)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9)
    normal.paragraph_format.space_after = Pt(6)

    for name in ["Title", "Heading 1", "Heading 2"]:
        doc.styles[name].font.name = "Aptos"
        doc.styles[name].font.color.rgb = RGBColor(25, 55, 85)

    doc.add_heading(title, 0)
    doc.add_paragraph(subtitle)

    footer = sec.footer.paragraphs[0]
    footer.text = "InGen Dynamics | Data Science Internship | Week 06"
    footer.style = "Caption"
    return doc

def paragraph(doc, text):
    doc.add_paragraph(text)

def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Shading Accent 1"

    for cell, value in zip(t.rows[0].cells, headers):
        cell.text = str(value)

    for row in rows:
        cells = t.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
    return t

def figure(doc, filename, caption):
    path = W06 / filename
    if not path.exists():
        raise FileNotFoundError(f"Required figure missing: {path}")
    doc.add_picture(str(path), width=Inches(6.5))
    doc.add_paragraph(caption, style="Caption")

def fmt(x, digits=3):
    return f"{float(x):.{digits}f}"

# ==========================================================
# REPORT 1: RISK DRIVERS
# ==========================================================

doc = setup_doc(
    "Week 06 — Robot Segmentation and Risk Drivers",
    "Site & Unit Segmentation | Risk Driver Synthesis"
)

doc.add_heading("1. Executive Summary", 1)
paragraph(
    doc,
    "This analysis segments 50 robots using operational and "
    "condition-related features, compares historical fault "
    "frequencies across segments and sites, and synthesizes "
    "signals identified in Weeks 3 and 4. The results support "
    "exploratory maintenance prioritization rather than causal "
    "claims or validated prospective risk classification."
)

doc.add_heading("2. Data and Methodology", 1)
paragraph(
    doc,
    "The Week 6 analysis uses corrected robot, mission, "
    "telemetry, fault, and maintenance records from "
    "data/causal_fixed/raw/. Eight operational and condition "
    "features were standardized before PCA. The first four "
    "principal components were used for K-Means clustering. "
    "Fault counts, maintenance counts, and fault rates were "
    "excluded from clustering features and used for "
    "post-clustering comparisons."
)

pc4 = variance.iloc[:4]["explained_variance_ratio"].sum()
paragraph(
    doc,
    f"The first four principal components explained "
    f"{pc4:.1%} of feature variance."
)

figure(
    doc,
    "W06_pca_scree_plot.png",
    "Figure 1. PCA explained variance."
)

doc.add_heading("3. Cluster Selection and Profiles", 1)
paragraph(
    doc,
    "K values from 2 through 8 were evaluated using silhouette "
    "score, Davies–Bouldin index, and inertia. Although K=5 "
    "achieved the highest silhouette score, it produced two "
    "clusters containing only two robots each. K=3 was "
    "selected as an operationally interpretable alternative "
    "with more balanced cluster sizes."
)

table(
    doc,
    ["K", "Silhouette", "Davies–Bouldin"],
    [
        [
            int(r["k"]),
            fmt(r["silhouette_score"]),
            fmt(r["davies_bouldin_index"])
        ]
        for _, r in evaluation.iterrows()
    ]
)

figure(
    doc,
    "W06_silhouette_scores.png",
    "Figure 2. Silhouette scores across candidate K values."
)
figure(
    doc,
    "W06_final_pca_clusters_k3.png",
    "Figure 3. Final K=3 segmentation in PCA space."
)

table(
    doc,
    [
        "Cluster", "Robots", "Faults / 1k h",
        "Battery (%)", "Wheel Wear (%)"
    ],
    [
        [
            int(r["cluster"]),
            int(r["robot_count"]),
            fmt(r["pooled_faults_per_1000_hours"]),
            fmt(r["mean_battery_capacity"]),
            fmt(r["mean_wheel_wear"])
        ]
        for _, r in clusters.iterrows()
    ]
)

paragraph(
    doc,
    "Cluster 0 has the highest historical fault frequency and "
    "wheel wear, despite relatively strong battery capacity. "
    "Cluster 1 shows lower battery capacity and higher charge "
    "cycles but the lowest overall historical fault frequency. "
    "This suggests that fleet condition is multidimensional "
    "and cannot be summarized by battery health alone."
)

figure(
    doc,
    "W06_cluster_fault_frequency.png",
    "Figure 4. Historical fault frequency by cluster."
)

doc.add_heading("4. Clustering Stability", 1)
mean_ari = stability["ari"].mean()
median_ari = stability["ari"].median()
minimum_ari = stability["ari"].min()
q10_ari = stability["ari"].quantile(0.10)
small_runs = int((stability["smallest_cluster"] < 3).sum())

paragraph(
    doc,
    f"Across {len(stability)} iterations of 80% robot "
    f"subsampling, the mean ARI was {mean_ari:.3f}, "
    f"median ARI was {median_ari:.3f}, minimum ARI "
    f"was {minimum_ari:.3f}, and 10th percentile "
    f"was {q10_ari:.3f}. {small_runs} iterations "
    f"produced a cluster containing fewer than three robots. "
    f"Previous random-initialization testing yielded mean "
    f"ARI approximately 0.946 for K=3."
)
paragraph(
    doc,
    "The segmentation is reasonably consistent across random "
    "initializations but moderately sensitive to sample "
    "composition. Cluster labels should therefore be treated "
    "as exploratory operational groupings."
)

doc.add_heading("5. Site-Level Historical Risk", 1)
table(
    doc,
    ["Site", "Robots", "Faults", "Faults / 1k h"],
    [
        [
            r["site_id"],
            int(r["robot_count"]),
            int(r["total_faults"]),
            fmt(r["faults_per_1000_hours"])
        ]
        for _, r in sites.iterrows()
    ]
)

figure(
    doc,
    "W06_site_fault_frequency.png",
    "Figure 5. Unadjusted historical fault frequency by site."
)

paragraph(
    doc,
    "A Poisson count model with log operating hours as an "
    "exposure offset included site indicators, standardized "
    "average mission duration, and standardized average "
    "motor current. HC0 robust standard errors were used."
)

site7 = regression.loc[
    regression["term"] == "C(site_id)[T.SITE_07]"
].iloc[0]

paragraph(
    doc,
    f"Relative to SITE_01, SITE_07 had an estimated "
    f"adjusted incidence rate ratio of {site7['IRR']:.2f} "
    f"(95% CI {site7['CI95_low']:.2f}–"
    f"{site7['CI95_high']:.2f}; "
    f"p={site7['p_value']:.3f}). "
    "This exploratory comparison is borderline and should "
    "not be treated as a definitive site effect."
)

figure(
    doc,
    "W06_site_regression_irr.png",
    "Figure 6. Adjusted site fault-rate ratios relative to SITE_01."
)

paragraph(
    doc,
    "Terrain is completely nested within site: SITE_07 is "
    "the only rough-terrain site, while the remaining sites "
    "are paved or mixed. Consequently, the SITE_07 association "
    "cannot be separated from rough-terrain exposure. "
    "Pearson dispersion was approximately 0.558, suggesting "
    "underdispersion relative to the Poisson variance "
    "assumption. The sample contains only 50 robots."
)

doc.add_heading("6. Cross-Week Risk Driver Synthesis", 1)
paragraph(
    doc,
    "Week 3 used matched pre-fault signal comparisons; Week 4 "
    "reported predictive feature importance; Week 6 examines "
    "full-history operational segments and site associations. "
    "These analyses answer different questions and their "
    "effect sizes should not be interpreted interchangeably."
)

drivers = [
    (
        "Charge cycles",
        "charge_cycles_30d_mean",
        "num__charge_cycles_7d_mean",
        "Monitor charging intensity and battery degradation."
    ),
    (
        "Motor current",
        "motor_current_a_30d_mean",
        "num__motor_current_a_30d_mean",
        "Investigate sustained changes in motor load."
    ),
    (
        "Battery capacity",
        "battery_capacity_pct_7d_mean",
        "num__battery_capacity_pct_30d_std",
        "Review declining or unstable battery capacity."
    ),
    (
        "Wheel wear",
        "wheel_wear_pct_7d_mean",
        "num__wheel_wear_pct_30d_mean",
        "Prioritize inspections for increasing wheel wear."
    )
]

driver_rows = []
for label, signal_name, shap_name, action in drivers:
    s = w03.loc[w03["feature"] == signal_name]
    h = shap.loc[shap["transformed_feature"] == shap_name]

    d = (
        fmt(s.iloc[0]["paired_cohens_d"])
        if not s.empty else "Not found"
    )
    imp = (
        fmt(h.iloc[0]["mean_abs_shap"])
        if not h.empty else "Not found"
    )

    driver_rows.append([label, d, imp, action])

table(
    doc,
    [
        "Driver", "W03 Cohen's d",
        "W04 Mean |SHAP|", "Potential action"
    ],
    driver_rows
)

paragraph(
    doc,
    "Week 4 SHAP magnitudes describe predictive contribution, "
    "not the direction or causality of a relationship. "
    "Week 6's negative motor-current coefficient also differs "
    "from the positive pre-fault motor-current contrast in "
    "Week 3. Differences in time windows, aggregation, "
    "covariate adjustment, and data vintage may contribute "
    "to this discrepancy; its cause is not established."
)

doc.add_heading("7. Operational Recommendations", 1)
for item in [
    "Pilot condition-based inspections for high wheel-wear units.",
    "Track battery capacity jointly with charge-cycle history.",
    "Review motor-current trends rather than applying a single "
    "fleet-wide absolute threshold.",
    "Review SITE_07's terrain exposure, route design, and "
    "maintenance requirements without attributing its higher "
    "rate solely to site management.",
    "Validate whether segmentation improves decisions beyond "
    "the existing failure-prediction model before deployment."
]:
    doc.add_paragraph(item, style="List Bullet")

doc.add_heading("8. Limitations and Next Steps", 1)
paragraph(
    doc,
    "The analysis is retrospective and uses a small, "
    "synthetic fleet. Clustering stability is imperfect, "
    "and site effects are confounded with terrain. "
    "Weeks 3 and 4 were produced using an earlier data "
    "vintage, whereas Week 6 uses corrected records; "
    "cross-week synthesis is therefore qualitative until "
    "the earlier analyses are rerun on the corrected data. "
    "Future work should evaluate cluster reproducibility, "
    "fault-type-specific rates, and incremental decision "
    "value on a consistent temporal dataset."
)

risk_path = W06 / "W06_Risk_Drivers.docx"
doc.save(risk_path)

# ==========================================================
# REPORT 2: FARI EXTENSION
# ==========================================================

fari = setup_doc(
    "Week 06 — Fari Extension Assessment",
    "Transferability Assessment and Validation Plan"
)

fari.add_heading("1. Scope and Current Status", 1)
paragraph(
    fari,
    "This report evaluates the potential transfer of the "
    "Aido-oriented fleet analytics workflow to Fari. "
    "No Fari-specific dataset was identified in the current "
    "project's data, scripts, or reports directories by "
    "filename search. Consequently, no Fari-specific "
    "feature extraction, clustering, or model evaluation "
    "has been performed."
)

fari.add_heading("2. Potentially Transferable Methods", 1)
table(
    fari,
    ["Component", "Transferability", "Validation needed"],
    [
        [
            "Data-quality checks",
            "Conceptually transferable",
            "Schema, timestamps, missingness"
        ],
        [
            "Feature engineering",
            "Requires field mapping",
            "Telemetry and event definitions"
        ],
        [
            "StandardScaler + PCA",
            "Method transferable",
            "Feature comparability and loadings"
        ],
        [
            "K-Means clustering",
            "Method transferable",
            "Cluster selection and stability"
        ],
        [
            "Fault-rate analysis",
            "Requires consistent exposure",
            "Fault definitions and operating hours"
        ],
        [
            "Predictive modeling",
            "Not validated for Fari",
            "Labels, temporal split, calibration"
        ]
    ]
)

fari.add_heading("3. Platform-Specific Assumptions", 1)
paragraph(
    fari,
    "The current workflow assumes fields such as unit_id, "
    "mission start/end timestamps, operating hours, "
    "battery capacity, charge cycles, motor current, "
    "wheel wear, fault events, and maintenance events. "
    "Whether these fields exist with equivalent meanings "
    "in Fari is unknown. Robot hardware, deployment "
    "patterns, terrain exposure, and failure definitions "
    "may differ."
)

fari.add_heading("4. Proposed Validation Procedure", 1)
for item in [
    "Obtain a Fari data dictionary and representative raw data.",
    "Map identifiers, timestamps, operating exposure, "
    "telemetry, fault events, and maintenance records.",
    "Audit missingness, sampling rates, and label consistency.",
    "Recompute transferable features without using future data.",
    "Fit and evaluate PCA and clustering on Fari data, "
    "including resampling stability and cluster profiles.",
    "Compare fault rates and operational meaning across "
    "Fari clusters, with appropriate uncertainty intervals.",
    "Evaluate any predictive-model transfer on held-out "
    "Fari time periods before operational use."
]:
    fari.add_paragraph(item, style="List Number")

fari.add_heading("5. Explicitly Unverified Items", 1)
paragraph(
    fari,
    "Fari schema compatibility, feature distributions, "
    "PCA variance structure, cluster assignments, fault-rate "
    "comparisons, model calibration, and deployment "
    "performance remain unverified. No claim of successful "
    "Fari transfer is made."
)

fari_path = W06 / "W06_Fari_Extension.docx"
fari.save(fari_path)

# ==========================================================
# WEEKLY LOG
# ==========================================================

log = """# Week 06 — Site & Unit Segmentation, Risk Drivers & Fari Extension

## Work Completed

This week, I developed an exploratory robot-segmentation and site-risk analysis using the corrected fleet dataset. I aggregated mission, telemetry, fault, and maintenance records into a robot-level feature table covering 50 units. To avoid defining clusters directly by their outcomes, I excluded fault counts, maintenance counts, and fault rates from the clustering features. I standardized eight operational and condition features and used principal component analysis to reduce redundancy. The first four components explained approximately 81.4% of the variance.

I evaluated K-Means solutions using silhouette scores, the Davies–Bouldin index, inertia, and stability. Although K=5 achieved a stronger silhouette score, it created two very small groups. I selected K=3 for more balanced and interpretable operational segments of 10, 20, and 20 robots. Random-initialization stability was strong, but 80% subsampling produced a mean adjusted Rand index of 0.768, indicating sensitivity to changes in the robot sample.

I compared historical fault frequencies across clusters and sites. Cluster 0 showed the highest wheel wear and historical fault frequency despite relatively healthy battery capacity. This demonstrated why battery health alone is insufficient to summarize fleet condition. I also fitted a Poisson regression with operating-hours exposure and operational covariates. SITE_07 had an elevated estimated fault-rate ratio relative to SITE_01, but SITE_07 was also the only rough-terrain site, making site and terrain effects inseparable.

## Reflection: Did Segmentation Add Value Beyond the Failure Model?

Segmentation provided a different level of insight from the existing failure-prediction model. Weeks 3 and 4 emphasized pre-fault signals and predictive feature importance, whereas Week 6 identified operationally distinct groups and differences in historical site-level fault frequency. For example, the cluster with the lowest battery capacity did not have the highest overall fault rate. This finding suggests that maintenance planning should consider several condition dimensions rather than a single health indicator.

The segmentation is not a substitute for the failure model. Its sample sensitivity and retrospective design limit its use for fixed risk labels. Moreover, the earlier signal and model analyses used a different data vintage from the corrected Week 6 dataset, so cross-week conclusions remain qualitative until consistently rerun.

## Fari Extension and Next Steps

I assessed which workflow components could conceptually transfer to Fari, including data validation, feature standardization, PCA, clustering, and fault-rate analysis. No Fari-specific dataset was available in the current project directories, so I documented schema requirements and a validation plan rather than claiming completed transfer testing. Next steps include obtaining Fari data, verifying feature definitions, repeating stability tests, and measuring whether segmentation improves maintenance decisions beyond the existing predictive model.
"""

log_path = WEEKLY / "Wk-06-Log.md"
log_path.write_text(log, encoding="utf-8")

print("\nGenerated deliverables:")
for path in [risk_path, fari_path, log_path]:
    print(f" - {path}")

print("\nReport generation completed.")
