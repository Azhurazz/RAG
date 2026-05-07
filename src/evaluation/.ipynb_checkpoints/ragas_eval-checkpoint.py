"""
src/evaluation/ragas_eval.py
RAGAS metrics + local heuristics + report export
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()

@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

@dataclass
class EvalReport:
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    num_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        return sum(self.metrics.values()) / len(self.metrics) if self.metrics else 0.0

class RAGEvaluator:
    DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def __init__(self, metrics: List[str] | None = None):
        self.metrics_to_run = metrics or self.DEFAULT_METRICS

    def evaluate(self, samples: List[EvalSample]) -> EvalReport:
        try:
            from ragas import evaluate
            from ragas.metrics import (faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness)
            from datasets import Dataset
        except ImportError as e:
            raise ImportError("pip install ragas datasets") from e

        metric_map = {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
                      "context_precision": context_precision, "context_recall": context_recall,
                      "answer_correctness": answer_correctness}
        selected = [metric_map[m] for m in self.metrics_to_run if m in metric_map]
        data = {"question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "contexts": [s.contexts for s in samples]}
        if any(s.ground_truth for s in samples):
            data["ground_truth"] = [s.ground_truth or "" for s in samples]
        result = evalaute(dataset=Dataset.from_dict(data), metrics=selected)
        return EvalReport(num_samples=len(samples),
                          metrics={k: round(float(v),4) for k, v in result.items()},
                          config={"metrics": self.metrics_to_run})

    def faithfulness_heuristic(self, sample: EvalSample) -> float:
        sents = [s.strip() for s in sample.answer.split(".") if s.strip()]
        if not sents: return 0.0
        ctx = " ".join(sample.contexts).lower()
        return round(sum(1 for s in sents
                         if any(w.lower() in ctx for w in s.split() if len(w) > 4)) / len(sents), 3)

    def print_report(self, report: EvalReport) -> None:
        t = Table(title=f"Eval Report — {report.timestamp}")
        t.add_column("Metric", style="cyan")
        t.add_column("Score",  style="green")
        t.add_column("Rating", style="yellow")
        for m, s in report.metrics.items():
            t.add_row(m, f"{s:.3f}",
                      "Good" if s >= 0.8 else "Fair" if s >= 0.6 else "Poor")
        t.add_row("OVERALL", f"{report.overall_score:.3f}", "")
        console.print(t)

    def save_report(self, report: EvalReport, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"timestamp": report.timestamp, "overall": report.overall_score,
                       "metrics": report.metrics, "num_samples": report.num_samples}, f, indent=2)
        print(f"[eval] saved -> {path}")