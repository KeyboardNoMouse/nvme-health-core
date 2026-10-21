"""Train the NVMe failure model and export it as JSON for in-browser inference."""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             confusion_matrix, precision_score, recall_score)
from features import build

OUT = Path("../web/public/model"); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("../data/NVMe_Drive_Failure_Dataset.csv")
X, y = build(df), df.Failure_Flag.values
mk = lambda: RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42)

# 1) honest holdout
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
m = mk().fit(Xtr, ytr)
p = m.predict_proba(Xte)[:, 1]

# 2) operating threshold from out-of-fold predictions on the training split only
oof = cross_val_predict(mk(), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_proba")[:, 1]
pr, rc, th = precision_recall_curve(ytr, oof)
f2 = 5 * pr * rc / (4 * pr + rc + 1e-9)          # recall-weighted: a missed failure costs more than a false alarm
thr = float(th[np.argmax(f2[:-1])])
pred = (p >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
metrics = dict(threshold=round(thr, 3), pr_auc=round(average_precision_score(yte, p), 3),
    roc_auc=round(roc_auc_score(yte, p), 3), precision=round(precision_score(yte, pred), 3),
    recall=round(recall_score(yte, pred), 3), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    n_train=len(ytr), n_test=len(yte), base_rate=round(float(y.mean()), 4))
print(metrics)

# 3) final model on all data -> export; fleet scores are out-of-fold so the dashboard isn't self-flattering
final = mk().fit(X, y)
fleet_p = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7), method="predict_proba")[:, 1]

def tree(t):
    s = t.tree_
    val = s.value[:, 0, :]; val = val[:, 1] / val.sum(1)
    return dict(f=s.feature.tolist(), t=[round(float(x), 5) for x in s.threshold],
                l=s.children_left.tolist(), r=s.children_right.tolist(), v=[round(float(x), 4) for x in val])

imp = sorted(zip(X.columns, final.feature_importances_), key=lambda a: -a[1])
print([(a, round(b, 3)) for a, b in imp[:8]])
(OUT / "model.json").write_text(json.dumps(dict(
    features=list(X.columns), threshold=thr, metrics=metrics,
    medians={c: float(X[c].median()) for c in X.columns},
    importances={a: round(float(b), 4) for a, b in imp},
    trees=[tree(e) for e in final.estimators_]), separators=(",", ":")))

fleet = df[["Drive_ID", "Vendor", "Model", "Firmware_Version", "Power_On_Hours", "Percent_Life_Used",
            "Temperature_C", "Media_Errors", "Read_Error_Rate", "Failure_Flag"]].copy()
fleet["risk"] = np.round(fleet_p, 4)
fleet.to_csv(OUT / "fleet_scored.csv", index=False)
df.drop(columns=["SMART_Warning_Flag", "Failure_Mode", "Failure_Flag"]).sample(300, random_state=3).to_csv(
    OUT / "sample_upload.csv", index=False)
print("model.json KB:", round((OUT / "model.json").stat().st_size / 1024))
