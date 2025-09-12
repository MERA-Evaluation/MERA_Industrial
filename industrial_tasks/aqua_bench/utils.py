from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)


def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower().replace("ё", "е")
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _letters_only(s: str) -> bool:
    return bool(s) and all(ch.isalpha() for ch in s.replace(" ", ""))


def _mc_letters(s: Any) -> str:
    s_norm = _norm(s)
    letters = [ch for ch in s_norm if ch.isalpha()]
    return (
        "".join(sorted(letters))
        if letters and all(ch.isalpha() for ch in letters)
        else s_norm
    )


def _em(pred: Any, gold: Any) -> float:
    p_mc, g_mc = _mc_letters(pred), _mc_letters(gold)
    if _letters_only(p_mc) and _letters_only(g_mc):
        return 1.0 if p_mc == g_mc else 0.0
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def _f1(pred: Any, gold: Any) -> float:
    """
    Token-level F1 (SQuAD-style) with MC special-case:
    - If both look like MC letter sets, compute set-F1 on letters.
    - Else compute word-token F1 after normalization.
    """
    p_mc, g_mc = _mc_letters(pred), _mc_letters(gold)
    if _letters_only(p_mc) and _letters_only(g_mc):
        P = set(p_mc)
        G = set(g_mc)
        tp = len(P & G)
        if tp == 0:
            return 0.0
        prec = tp / max(len(P), 1)
        rec = tp / max(len(G), 1)
        return 2 * prec * rec / (prec + rec)

    p_tokens = _norm(pred).split()
    g_tokens = _norm(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    # multiset overlap
    pc, gc = Counter(p_tokens), Counter(g_tokens)
    tp = sum(min(pc[t], gc[t]) for t in pc.keys() | gc.keys())
    prec = tp / max(len(p_tokens), 1)
    rec = tp / max(len(g_tokens), 1)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(len(xs), 1)


# Harness hooks
def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    pred = results[0] if results else ""
    gold = doc.get("outputs", "")
    return {
        "em": _em(pred, gold),
        "f1": _f1(pred, gold),
    }


def aggregation() -> Dict[str, Any]:
    return {"em": _mean, "f1": _mean}


def higher_is_better() -> Dict[str, bool]:
    return {"em": True, "f1": True}
