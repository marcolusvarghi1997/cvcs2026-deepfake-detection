import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Paths
clip_model_path = "/work/cvcs2026/resnet_gang/results/CLIP/classifiers/case1/logreg/model.joblib"
fusion_model_path = "/work/cvcs2026/resnet_gang/results/CoDE_CLIP/classifiers/case1/logreg/model.joblib"

clip_features_path = "/work/cvcs2026/resnet_gang/outputs/CLIP/features/case1/features_test.npy"
fusion_features_path = "/work/cvcs2026/resnet_gang/outputs/CoDE_CLIP/features/case1/features_test.npy"

metadata_path = "/work/cvcs2026/resnet_gang/outputs/CoDE_CLIP/features/case1/metadata_test.parquet"

out_path = "/homes/gdavena/resnet/results/CoDE_CLIP/visualizations/logreg/clip_vs_fusion_error_analysis_case1.csv"

clip_threshold = 0.5
fusion_threshold = 0.543




clip_bundle = joblib.load(clip_model_path)
fusion_bundle = joblib.load(fusion_model_path)

clip_model = clip_bundle["model"]
fusion_model = fusion_bundle["model"]

X_clip = np.load(clip_features_path)
X_fusion = np.load(fusion_features_path)
meta = pd.read_parquet(metadata_path)

label_col = "label"
y = meta[label_col].values.astype(int)


# Scores
clip_scores = clip_model.decision_function(X_clip)
fusion_scores = fusion_model.decision_function(X_fusion)

# Predictions
clip_pred = (clip_scores >= clip_threshold).astype(int)
fusion_pred = (fusion_scores >= fusion_threshold).astype(int)

clip_correct = clip_pred == y
fusion_correct = fusion_pred == y

# Categories
category = np.full(len(y), "both_wrong", dtype=object)
category[clip_correct & fusion_correct] = "both_correct"
category[~clip_correct & fusion_correct] = "fusion_fixes_clip"
category[clip_correct & ~fusion_correct] = "fusion_breaks_clip"

df = meta.copy()
df["y_true"] = y
df["clip_score"] = clip_scores
df["fusion_score"] = fusion_scores
df["clip_pred"] = clip_pred
df["fusion_pred"] = fusion_pred
df["clip_correct"] = clip_correct
df["fusion_correct"] = fusion_correct
df["category"] = category

# Summary
summary = df["category"].value_counts().to_frame("count")
summary["percentage"] = summary["count"] / len(df) * 100

print("\nCategory summary:")
print(summary)

print("\nCLIP metrics:")
print("Accuracy:", accuracy_score(y, clip_pred))
print("F1:", f1_score(y, clip_pred))
print("AUROC:", roc_auc_score(y, clip_scores))
print("AP:", average_precision_score(y, clip_scores))

print("\nFusion metrics:")
print("Accuracy:", accuracy_score(y, fusion_pred))
print("F1:", f1_score(y, fusion_pred))
print("AUROC:", roc_auc_score(y, fusion_scores))
print("AP:", average_precision_score(y, fusion_scores))

print(f"\nBreakdown by generator:")
print(pd.crosstab(df["generator"], df["category"], normalize="index"))


df.to_csv(out_path, index=False)
print(f"\nSaved detailed analysis to: {out_path}")