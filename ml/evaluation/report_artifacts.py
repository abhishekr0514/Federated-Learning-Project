"""Plots and HTML/CSV/JSON artifacts for model evaluation reports."""

from __future__ import annotations

import base64
import csv
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.evaluation.classification_metrics import (
    CLASS_LABELS,
    absolute_metric_differences,
    metrics_comparison_row,
)
from ml.evaluation.gat_bundle_evaluator import GATCheckpointEvaluation

RESEARCH_DISCLAIMER = (
    "PTB-XL proxy classification (NORM vs MI/STTC) is a research demonstration only. "
    "It is not a clinically validated heart-disease diagnosis. "
    "Softmax outputs are uncalibrated and must not be interpreted as clinical probabilities."
)


def _save_confusion_matrix_plot(
    cm: List[List[int]],
    *,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    arr = np.array(cm, dtype=int)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_roc_plot(fpr: List[float], tpr: List[float], *, title: str, out_path: Path, auroc: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pr_plot(recall: List[float], precision: List[float], *, title: str, out_path: Path, ap: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_comparison_plot(rows: List[Dict[str, Any]], out_path: Path) -> None:
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "mcc"]
    labels = [r["model_id"] for r in rows if r.get("status") != "unavailable"]
    if len(labels) < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Comparison plot requires two evaluated models.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, row in enumerate(rows):
        if row.get("status") == "unavailable":
            continue
        vals = [float(row.get(k) or 0) if not (isinstance(row.get(k), float) and math.isnan(row.get(k))) else 0 for k in keys]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=row["model_id"])
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (selected metrics)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _img_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def write_evaluation_artifacts(
    output_dir: Path,
    *,
    centralized: Optional[GATCheckpointEvaluation],
    federated: Optional[GATCheckpointEvaluation],
    manifest: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    def _plots_for(ev: GATCheckpointEvaluation, prefix: str) -> None:
        m = ev.metrics
        if m.get("n_samples", 0) == 0:
            return
        cm_path = output_dir / f"{prefix}_confusion_matrix.png"
        _save_confusion_matrix_plot(
            m["confusion_matrix_raw"]["matrix"],
            title=f"{ev.model_role} — confusion matrix (counts)",
            out_path=cm_path,
        )
        paths[f"{prefix}_confusion_matrix"] = str(cm_path.name)

        if "roc_curve" in m and math.isfinite(m.get("auroc", float("nan"))):
            roc_path = output_dir / f"{prefix}_roc_curve.png"
            _save_roc_plot(
                m["roc_curve"]["fpr"],
                m["roc_curve"]["tpr"],
                title=f"{ev.model_role} ROC (positive = MI/STTC)",
                out_path=roc_path,
                auroc=float(m["auroc"]),
            )
            paths[f"{prefix}_roc_curve"] = str(roc_path.name)

        if "pr_curve" in m and math.isfinite(m.get("average_precision", float("nan"))):
            pr_path = output_dir / f"{prefix}_pr_curve.png"
            _save_pr_plot(
                m["pr_curve"]["recall"],
                m["pr_curve"]["precision"],
                title=f"{ev.model_role} precision–recall",
                out_path=pr_path,
                ap=float(m["average_precision"]),
            )
            paths[f"{prefix}_pr_curve"] = str(pr_path.name)

    if centralized:
        _plots_for(centralized, "gat_fusion")
    if federated:
        _plots_for(federated, "fl_model")

    rows: List[Dict[str, Any]] = []
    if centralized:
        rows.append(metrics_comparison_row("gat_centralized", centralized.metrics))
    if federated:
        rows.append(metrics_comparison_row("gat_federated_global", federated.metrics))

    cmp_path = output_dir / "model_comparison.png"
    _save_comparison_plot(rows, cmp_path)
    paths["model_comparison"] = cmp_path.name

    # JSON metrics (strip large curves for separate storage optional — keep in json)
    payload = {
        "manifest": manifest,
        "models": {},
        "comparison": {},
    }
    if centralized:
        payload["models"]["gat_centralized"] = _json_safe_metrics(centralized.metrics)
    if federated:
        payload["models"]["gat_federated_global"] = _json_safe_metrics(federated.metrics)
    if len(rows) == 2 and rows[0].get("status") != "unavailable":
        payload["comparison"]["metric_differences_federated_minus_centralized"] = absolute_metric_differences(
            rows[0], rows[1]
        )
        payload["comparison"]["same_test_examples"] = manifest.get("comparable_examples", False)

    json_path = output_dir / "model_evaluation_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["metrics_json"] = json_path.name

    csv_path = output_dir / "model_evaluation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    paths["metrics_csv"] = csv_path.name

    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["evaluation_manifest"] = manifest_path.name

    html_path = output_dir / "model_evaluation_report.html"
    html_path.write_text(
        _render_html(
            output_dir,
            centralized=centralized,
            federated=federated,
            manifest=manifest,
            rows=rows,
            artifact_names=paths,
        ),
        encoding="utf-8",
    )
    paths["html_report"] = html_path.name
    return paths


def _json_safe_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def _render_html(
    output_dir: Path,
    *,
    centralized: Optional[GATCheckpointEvaluation],
    federated: Optional[GATCheckpointEvaluation],
    manifest: Dict[str, Any],
    rows: List[Dict[str, Any]],
    artifact_names: Dict[str, str],
) -> str:
    def img_tag(key: str, alt: str) -> str:
        name = artifact_names.get(key)
        if not name:
            return f"<p><em>{alt} not available.</em></p>"
        p = output_dir / name
        if not p.is_file():
            return f"<p><em>{alt} missing on disk.</em></p>"
        return f'<img alt="{alt}" src="{_img_data_uri(p)}" style="max-width:100%;height:auto;" />'

    def metrics_section(ev: Optional[GATCheckpointEvaluation], heading: str) -> str:
        if not ev:
            return f"<h2>{heading}</h2><p>Not evaluated (checkpoint or bundle unavailable).</p>"
        m = ev.metrics
        if m.get("n_samples", 0) == 0:
            return f"<h2>{heading}</h2><p>No metrics ({m.get('reason', 'unknown')}).</p>"
        lines = [
            f"<h2>{heading}</h2>",
            f"<p><strong>Checkpoint:</strong> <code>{ev.checkpoint_path}</code></p>",
            f"<p><strong>Test samples:</strong> {m['n_samples']} "
            f"(correct {m['n_correct']}, incorrect {m['n_incorrect']})</p>",
            f"<p><strong>Accuracy:</strong> {m['accuracy']:.4f} &nbsp; "
            f"<strong>Balanced accuracy:</strong> {m['balanced_accuracy']:.4f}</p>",
            f"<p><strong>Macro F1:</strong> {m['macro_f1']:.4f} &nbsp; "
            f"<strong>Weighted F1:</strong> {m['weighted_f1']:.4f}</p>",
            f"<p><strong>AUROC (class 1):</strong> {_fmt(m.get('auroc'))} &nbsp; "
            f"<strong>Average precision:</strong> {_fmt(m.get('average_precision'))}</p>",
            f"<p><strong>MCC:</strong> {_fmt(m.get('mcc'))} &nbsp; "
            f"<strong>Cohen κ:</strong> {_fmt(m.get('cohen_kappa'))}</p>",
            f"<pre>{m.get('classification_report_text', '')}</pre>",
        ]
        return "\n".join(lines)

    ds = manifest.get("dataset") or {}
    smoke = ds.get("smoke_caps") or {}
    split_counts = ds.get("split_counts") or {}
    limitations = []
    if ds.get("test_split_mode") == "capped" and smoke.get("max_test"):
        limitations.append(
            f"Test split capped at max_test={smoke['max_test']} rows in the embedding bundle (smoke/dev cap). "
            f"Full eligible fold-10 test count was {split_counts.get('test_eligible_full_split', 'unknown')}. "
            "Metrics are on the capped subset, not the full PTB-XL test fold population."
        )
    elif split_counts.get("test_in_bundle") and split_counts.get("test_eligible_full_split"):
        if split_counts["test_in_bundle"] == split_counts["test_eligible_full_split"]:
            limitations.append(
                f"Test evaluation uses the full eligible fold-10 split "
                f"({split_counts['test_in_bundle']} records) from the bundle."
            )
    if not manifest.get("comparable_examples"):
        limitations.append("Models were not both evaluated on the same examples.")

    comp_html = "<table border='1' cellpadding='6'><tr>"
    if rows:
        for h in rows[0].keys():
            comp_html += f"<th>{h}</th>"
        comp_html += "</tr>"
        for r in rows:
            comp_html += "<tr>" + "".join(f"<td>{_fmt(r.get(k))}</td>" for k in r.keys()) + "</tr>"
    comp_html += "</table>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>MedShield-FL Model Evaluation Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
    .disclaimer {{ background: #fff8e6; border: 1px solid #e6c200; padding: 1rem; margin: 1rem 0; }}
    pre {{ background: #f8f8f8; padding: 1rem; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>MedShield-FL Model Evaluation Report</h1>
  <p><strong>Generated (UTC):</strong> {manifest.get('generated_at_utc')}</p>
  <div class="disclaimer"><strong>Research only:</strong> {RESEARCH_DISCLAIMER}</div>

  <h2>Methodology</h2>
  <p>Both models use <code>MedShieldGATECGReportNet</code> on precomputed 128-d ECG and clinical-text embeddings
  from the federated FL bundle test partition. Predicted class = <code>argmax(softmax(logits))</code>.
  AUROC / average precision use <code>softmax[:, 1]</code> (P(MI/STTC)), not max-softmax confidence.</p>
  <p><strong>Bundle:</strong> <code>{ds.get('bundle_path', '')}</code></p>
  <p><strong>Target:</strong> {ds.get('target_definition', '')}</p>
  <p><strong>Class mapping:</strong> 0 = NORM, 1 = MI/STTC (positive class for AUROC).</p>
  <p><strong>Same held-out examples for comparison:</strong> {manifest.get('comparable_examples')}</p>

  {metrics_section(centralized, "GAT ECG + Clinical Report Fusion (centralized checkpoint)")}
  {img_tag('gat_fusion_confusion_matrix', 'GAT confusion matrix')}
  {img_tag('gat_fusion_roc_curve', 'GAT ROC')}
  {img_tag('gat_fusion_pr_curve', 'GAT PR')}

  {metrics_section(federated, "Federated learning global model (FedAvg)")}
  {img_tag('fl_model_confusion_matrix', 'FL confusion matrix')}
  {img_tag('fl_model_roc_curve', 'FL ROC')}
  {img_tag('fl_model_pr_curve', 'FL PR')}

  <h2>Model comparison</h2>
  {comp_html}
  {img_tag('model_comparison', 'Comparison chart')}

  <h2>Limitations</h2>
  <ul>{''.join(f'<li>{x}</li>' for x in limitations) or '<li>See evaluation_manifest.json for configuration.</li>'}</ul>
</body>
</html>"""


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
