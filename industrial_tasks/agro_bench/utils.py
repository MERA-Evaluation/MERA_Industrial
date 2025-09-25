from typing import Dict, List, Any

from transformers.data.metrics import squad_metrics


def doc_to_text(doc: Dict[str, Any]) -> str:

    return doc["instruction"].format(**dict(**doc["inputs"], **doc["meta"]))

def process_results(doc: Dict, results: List[str]) -> Dict:
    if len(doc["outputs"]) > 0:
        gold_label = doc["outputs"]
        pred_label = results[0]

        f1 = squad_metrics.compute_f1(gold_label, pred_label)
        em = squad_metrics.compute_exact(gold_label, pred_label)

        return {"f1": f1, "em": em}
    return {"f1": 0, "em": 0}  # if no label provided (test answers are secret)
