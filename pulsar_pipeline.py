"""Automated Pulsar Candidate Selection on HTRU2.

Implements the methodology from the presentation:
  RobustScaler -> SMOTE (train fold only) -> XGBoost / RBF-SVM, recall-oriented.
Run:  python pulsar_pipeline.py
"""
import json, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, precision_recall_curve,
                             roc_curve, recall_score, precision_score, f1_score,
                             matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
OUT = "results/"
COLS = ["ip_mean", "ip_std", "ip_kurtosis", "ip_skew",
        "dm_mean", "dm_std", "dm_kurtosis", "dm_skew", "label"]

df = pd.read_csv("data/HTRU_2.csv", header=None, names=COLS)
X, y = df.drop(columns="label"), df["label"]
print(f"Samples {len(df)} | pulsars {y.sum()} ({y.mean():.2%}) | RFI {(y==0).sum()}")

# Stratified 80/20 split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()  # ~9.9


def xgb(**kw):
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                         colsample_bytree=0.9, eval_metric="logloss", random_state=SEED,
                         n_jobs=-1, **kw)


models = {
    # Slide-07 pipeline: SMOTE balances data, so no extra class weight (avoids double-correcting)
    "XGBoost + SMOTE": Pipeline([("scaler", RobustScaler()), ("smote", SMOTE(random_state=SEED)),
                                 ("clf", xgb())]),
    # Cost-sensitive alternative: weighted log-loss, no resampling
    "XGBoost (weighted)": Pipeline([("scaler", RobustScaler()),
                                    ("clf", xgb(scale_pos_weight=spw))]),
    "RBF-SVM + SMOTE": Pipeline([("scaler", RobustScaler()), ("smote", SMOTE(random_state=SEED)),
                                 ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                                             probability=True, random_state=SEED))]),
    "RBF-SVM (weighted)": Pipeline([("scaler", RobustScaler()),
                                    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                                                class_weight="balanced", probability=True,
                                                random_state=SEED))]),
}

results, probs = {}, {}
for name, pipe in models.items():
    t = time.time(); pipe.fit(X_tr, y_tr); fit_s = time.time() - t
    t = time.time(); p = pipe.predict_proba(X_te)[:, 1]; pred_s = time.time() - t
    yp = (p >= 0.5).astype(int); probs[name] = p
    tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
    results[name] = dict(recall=recall_score(y_te, yp), precision=precision_score(y_te, yp),
                         f1=f1_score(y_te, yp), mcc=matthews_corrcoef(y_te, yp),
                         roc_auc=roc_auc_score(y_te, p), pr_auc=average_precision_score(y_te, p),
                         TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                         fit_s=fit_s, per_million_s=pred_s / len(X_te) * 1e6)

# Ensemble (slide 05): average XGB+SMOTE and SVM+SMOTE probabilities
ens = (probs["XGBoost + SMOTE"] + probs["RBF-SVM + SMOTE"]) / 2
probs["Ensemble (XGB+SVM)"] = ens
yp = (ens >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
results["Ensemble (XGB+SVM)"] = dict(
    recall=recall_score(y_te, yp), precision=precision_score(y_te, yp), f1=f1_score(y_te, yp),
    mcc=matthews_corrcoef(y_te, yp), roc_auc=roc_auc_score(y_te, ens),
    pr_auc=average_precision_score(y_te, ens), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
    fit_s=np.nan, per_million_s=np.nan)

res = pd.DataFrame(results).T
print("\nHold-out test results (threshold 0.5)\n", res.round(4).to_string())

# Recall-first operating point: highest threshold achieving >=95% recall for the best model
best = "XGBoost + SMOTE"
pr, rc, th = precision_recall_curve(y_te, probs[best])
ok = np.where(rc[:-1] >= 0.95)[0]
t95 = th[ok[-1]]
yp95 = (probs[best] >= t95).astype(int)
op95 = dict(threshold=float(t95), recall=recall_score(y_te, yp95),
            precision=precision_score(y_te, yp95),
            rfi_rejected=float(((yp95 == 0) & (y_te == 0)).sum() / (y_te == 0).sum()))
print("\n95%-recall operating point:", {k: round(v, 4) for k, v in op95.items()})

# 5-fold stratified CV (SMOTE only inside train folds thanks to imblearn Pipeline)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cvr = cross_validate(models[best], X, y, cv=cv, n_jobs=1,
                     scoring=["recall", "precision", "f1", "roc_auc", "average_precision"])
cv_summary = {k.replace("test_", ""): (float(v.mean()), float(v.std()))
              for k, v in cvr.items() if k.startswith("test_")}
print("\n5-fold CV (XGB+SMOTE):", {k: f"{m:.4f}±{s:.4f}" for k, (m, s) in cv_summary.items()})

# Feature importance (gain) from a fitted model
imp = pd.Series(models[best].named_steps["clf"].feature_importances_, index=COLS[:-1]).sort_values()

# ---------------- Plots ----------------
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
for n, p in probs.items():
    f, t_, _ = roc_curve(y_te, p); ax[0, 0].plot(f, t_, label=f"{n} ({results[n]['roc_auc']:.3f})")
ax[0, 0].plot([0, 1], [0, 1], "k:"); ax[0, 0].set(title="ROC", xlabel="FPR", ylabel="TPR"); ax[0, 0].legend(fontsize=7)
for n, p in probs.items():
    a, b, _ = precision_recall_curve(y_te, p); ax[0, 1].plot(b, a, label=f"{n} ({results[n]['pr_auc']:.3f})")
ax[0, 1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision"); ax[0, 1].legend(fontsize=7)
cm = confusion_matrix(y_te, (probs[best] >= .5).astype(int))
ax[1, 0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[1, 0].text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
ax[1, 0].set(title=f"Confusion matrix – {best}", xticks=[0, 1], yticks=[0, 1],
             xticklabels=["RFI", "Pulsar"], yticklabels=["RFI", "Pulsar"], xlabel="Predicted", ylabel="True")
imp.plot.barh(ax=ax[1, 1], color="#2563eb"); ax[1, 1].set_title("XGBoost feature importance")
plt.tight_layout(); plt.savefig(OUT + "pulsar_results.png", dpi=130); plt.close()

# Class-separation plot of the top features
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, c in zip(ax, ["ip_kurtosis", "dm_mean", "ip_skew"]):
    for lab, col, nm in [(0, "#d97706", "RFI"), (1, "#059669", "Pulsar")]:
        a.hist(df.loc[df.label == lab, c], bins=60, alpha=.6, color=col, label=nm, density=True)
    a.set_title(c); a.legend()
plt.tight_layout(); plt.savefig(OUT + "feature_distributions.png", dpi=130); plt.close()

res.round(4).to_csv(OUT + "model_comparison.csv")
json.dump(dict(test=results, op95=op95, cv=cv_summary, importance=imp.to_dict()),
          open(OUT + "metrics.json", "w"), indent=2, default=float)
print("\nfeature importance:\n", imp.sort_values(ascending=False).round(3).to_string())
