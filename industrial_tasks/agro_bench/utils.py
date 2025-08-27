import re
from typing import Dict, Iterable, List


def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _exact_match(pred: str, gold: str) -> float:
    """EM по нормализованным строкам."""
    return float(_normalize(pred) == _normalize(gold))


def _token_f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    # считаем пересечение мультимножеств
    from collections import Counter

    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def _to_golds(doc) -> List[str]:
    """
    Приводим таргеты к списку строк: поддержка 'outputs' (строка с ';' или список), а также распространённые ключи 'answers'/'targets'/'target'.

    Унифицируем gold-ответы:
    - если doc["outputs"] строка — делим по ';'
    - если список — приводим к списку строк
    - если ключ другой (например, 'answers'/'target') — пробуем его
    """
    for key in ("outputs", "answers", "targets", "target"):
        if key in doc and doc[key] is not None:
            v = doc[key]
            if isinstance(v, str):
                return [x.strip() for x in v.split(";") if x.strip()]
            if isinstance(v, Iterable):
                return [str(x).strip() for x in v if str(x).strip()]
    return []


def compute_max_em_f1_over_target(doc: Dict, results: List[str]) -> Dict[str, float]:
    golds = _to_golds(doc)
    pred = results[0] if results else ""
    if not golds:
        return {"f1": 0.0, "em": 0.0}
    em = max(_exact_match(pred, g) for g in golds)
    f1 = max(_token_f1(pred, g) for g in golds)
    return {"f1": f1, "em": em}


def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    # Сюда ссылается YAML: !function utils.process_results
    return compute_max_em_f1_over_target(doc, results)
