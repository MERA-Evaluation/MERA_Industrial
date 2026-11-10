import re
from typing import Dict, List, Any
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
from lm_eval.api.registry import FILTER_REGISTRY
from transformers.data.metrics import squad_metrics


def doc_to_text(doc: Dict[str, Any]) -> str:
    inputs = {
        key: "" if value is None else str(value) for key, value in dict(doc["inputs"]).items()
    }
    for letter in "abcdefgh":
        inputs.setdefault(f"option_{letter}", "")
    meta = {key: "" if value is None else value for key, value in dict(doc["meta"]).items()}
    rendered = str(doc["instruction"]).format(**inputs, **meta)
    lines = [line for line in rendered.splitlines() if not re.fullmatch(r"[A-Z]\.\s*", line.strip())]
    return "\n".join(lines)


def process_results(doc: Dict, results: List[str]) -> Dict:
    if len(doc["outputs"]) > 0:
        gold_label = doc["outputs"]
        pred_label = results[0]

        f1 = squad_metrics.compute_f1(gold_label, pred_label)
        em = squad_metrics.compute_exact(gold_label, pred_label)

        return {"f1": f1, "em": em}
    return {"f1": 0, "em": 0}  # if no label provided (test answers are secret)


if not FILTER_REGISTRY.get("remove_whitespace_and_nones", None):

    @register_filter("remove_whitespace_and_nones")
    class RemoveWhitespaceAndNones(Filter):

        def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
            def filter_set(inst):
                filtered_resp = []
                for resp in inst:
                    if not resp:
                        resp = ""
                    else:
                        resp = resp.lstrip()
                    filtered_resp.append(resp)
                return filtered_resp

            filtered_resps = [filter_set(resp) for resp in resps]

            return filtered_resps
