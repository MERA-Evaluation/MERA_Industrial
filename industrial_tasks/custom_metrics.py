from typing import Dict, List

from lm_eval.api.metrics import metric_max_over_ground_truths
from transformers.evaluation import squad_metrics


def compute_max_em_f1_over_target(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Compute maximum EM and F1 scores over a semicolon-separated list of ground truth answers.

    The lm-evaluation-harness uses a list of generation candidates ``results``. In
    our tasks we only ever examine the first candidate.  When a document
    contains answers (``outputs``) they are separated by semicolons.  We
    compute the maximum of the token-level F1 and exact match scores over
    all provided answers and return them in a dictionary.  If no answer is
    present (for example when test answers are hidden), both metrics are
    zero.

    Parameters
    ----------
    doc : Dict
        The dictionary representing one dataset entry.  Must contain a key
        ``"outputs"`` whose value is either an empty string or a
        semicolon-separated list of acceptable answers.
    results : List[str]
        A list of model generations.  Only the first element is used.

    Returns
    -------
    Dict[str, float]
        A dictionary with keys ``"f1"`` and ``"em"`` containing the best
        F1 and exact match scores, respectively.
    """

    # If the ground truth answers are hidden (empty string), return zeros.
    if not doc.get("outputs"):
        return {"f1": 0.0, "em": 0.0}

    # Split multiple answers on semicolons and strip whitespace.
    gold_label_set = [ans.strip() for ans in doc["outputs"].split(";") if ans.strip()]
    # Use the first model generation as the prediction.
    pred = results[0] if results else ""

    # Compute F1 and EM using lm-evaluation-harness helpers.
    f1 = metric_max_over_ground_truths(squad_metrics.compute_f1, pred, gold_label_set)
    em = metric_max_over_ground_truths(
        squad_metrics.compute_exact, pred, gold_label_set
    )

    return {"f1": f1, "em": em}
