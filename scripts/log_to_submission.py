from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from lm_eval.utils import sanitize_model_name
except Exception:

    def sanitize_model_name(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


ROOT = Path(__file__).resolve().parents[1]  # repo root
TASKS_DIR = ROOT / "industrial_tasks"


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def discover_task_ids(tasks_dir: Path) -> Dict[str, str]:
    """
    Build a map of {task_dir_name -> task_id} from task.yaml files.
    """
    mapping: Dict[str, str] = {}
    for task_yaml in tasks_dir.glob("*/task.yaml"):
        try:
            raw = task_yaml.read_text(encoding="utf-8")
            m = re.search(r"(?m)^\s*task:\s*(?P<tid>[A-Za-z0-9_\-]+)\s*$", raw)
            if not m:
                continue
            task_id = m.group("tid").strip()
            task_dir = task_yaml.parent.name
            mapping[task_dir] = task_id
        except Exception:
            continue
    return mapping


_TEXT_KEYS = (
    "response",
    "decoded",
    "text",
    "completion",
    "generated",
    "output",
    "prediction",
    "pred",
)
_ID_KEYS = ("id", "doc_id", "sample_id", "uuid", "qid", "question_id")


def _pick_first(d: dict, keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            # unwrap lists like ["text"] or [{"text": "..."}]
            if isinstance(v, list) and v:
                v = v[0]
            if isinstance(v, dict):
                # grab common inner fields
                for kk in ("text", "response", "decoded"):
                    if kk in v and v[kk] is not None:
                        return str(v[kk])
                # or fallback to first non-null scalar
                for kk, vv in v.items():
                    if isinstance(vv, (str, int, float)) and vv is not None:
                        return str(vv)
                continue
            return str(v)
    return None


def normalize_samples(samples_path: Path) -> List[dict]:
    """
    Convert any lm-eval samples.jsonl schema into lines:
      {"id": <id>, "response": <string>}
    """
    rows = []
    for obj in read_jsonl(samples_path):
        sid = _pick_first(obj, _ID_KEYS)
        txt = _pick_first(obj, _TEXT_KEYS)

        if txt is None:
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                ch = choices[0]
                if isinstance(ch, dict) and "text" in ch:
                    txt = str(ch["text"])

        if txt is None and "decoded" in obj:
            txt = str(obj["decoded"])

        if sid is None:
            seed = json.dumps(obj.get("input", obj), ensure_ascii=False, sort_keys=True)
            sid = hashlib.md5(seed.encode("utf-8")).hexdigest()

        if txt is None:
            txt = ""

        rows.append({"id": sid, "response": txt})
    return rows


@dataclass
class FoundSamples:
    task_dir: str
    task_id: str
    samples_path: Path


def find_all_samples(outputs_dir: Path, task_map: Dict[str, str]) -> List[FoundSamples]:
    """
    Locate every lm-eval samples file under outputs_dir.

    Supports both:
      - <...>/samples.jsonl
      - <...>/samples_<task>_<timestamp>.jsonl
    """
    found: List[FoundSamples] = []

    # 1) Collect all plausible sample files
    candidate_paths = list(outputs_dir.rglob("samples.jsonl"))
    candidate_paths += list(outputs_dir.rglob("samples_*.jsonl"))

    if not candidate_paths:
        return found

    # 2) For each candidate, try to infer the task
    for samples_path in candidate_paths:
        fname = samples_path.name

        # (a) Filename-based match: samples_{task}_<timestamp>.jsonl
        task_dir: Optional[str] = None
        if fname.startswith("samples_") and fname.endswith(".jsonl"):
            base = fname[
                len("samples_") : -len(".jsonl")
            ]  # "<task>_<timestamp>" OR maybe just "<task>"
            for tdir in task_map.keys():
                prefix = f"{tdir}_"
                if base == tdir or base.startswith(prefix):
                    task_dir = tdir
                    break

        # (b) Path contains a known task folder
        if task_dir is None:
            for part in samples_path.parts:
                if part in task_map:
                    task_dir = part
                    break

        # (c) Fallback: parent-of-parent (usual harness layout)
        if task_dir is None:
            try:
                candidate = samples_path.parent.parent.name
                if candidate in task_map:
                    task_dir = candidate
            except Exception:
                pass

        if task_dir is None:
            continue

        task_id = task_map[task_dir]
        found.append(
            FoundSamples(task_dir=task_dir, task_id=task_id, samples_path=samples_path)
        )

    latest_by_task: Dict[str, FoundSamples] = {}
    for fs in found:
        key = fs.task_id
        cur = latest_by_task.get(key)
        if (
            cur is None
            or fs.samples_path.stat().st_mtime > cur.samples_path.stat().st_mtime
        ):
            latest_by_task[key] = fs

    return list(latest_by_task.values())


def make_submission_zip(
    outputs_dir: Path,
    dst_dir: Path,
    model_args: str,
) -> Path:
    task_map = discover_task_ids(TASKS_DIR)
    samples = find_all_samples(outputs_dir, task_map)
    if not samples:
        raise RuntimeError(
            f"No samples.jsonl found under {outputs_dir}. "
            "Did you run lm_eval with --log_samples and correct --output_path?"
        )

    model_tag = sanitize_model_name(model_args or "unknown_model")
    stamp = now_stamp()
    work_dir = Path(tempfile.mkdtemp(prefix="mera_submit_"))
    submit_dir = work_dir / f"submission_{model_tag}_{stamp}"
    submit_dir.mkdir(parents=True, exist_ok=True)

    # 1) Normalize and write per-task files
    normalized_files: List[Path] = []
    for s in samples:
        out_path = submit_dir / f"{s.task_id}.jsonl"
        rows = normalize_samples(s.samples_path)
        write_jsonl(rows, out_path)
        normalized_files.append(out_path)

    # 2) Meta
    meta = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_args": model_args,
        "outputs_dir": str(outputs_dir),
        "tasks": sorted({s.task_id for s in samples}),
        "files": {p.name: file_sha256(p) for p in normalized_files},
        "tool": "log_to_submission.py",
        "tool_version": "1.0.0",
    }
    (submit_dir / "submission_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) Zip
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dst_dir / f"{submit_dir.name}.zip"
    shutil.make_archive(
        zip_path.with_suffix(""), "zip", root_dir=work_dir, base_dir=submit_dir.name
    )

    # Cleanup temp folder
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    return zip_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack MERA submission archive from lm-eval logs."
    )
    p.add_argument(
        "--outputs_dir", required=True, help="Path passed as --output_path to lm_eval"
    )
    p.add_argument("--dst_dir", required=True, help="Where to put the ZIP")
    p.add_argument(
        "--model_args", default="", help="Exact --model_args string you used"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()
    zip_path = make_submission_zip(outputs_dir, dst_dir, args.model_args)
    print(f"[OK] Submission ZIP created: {zip_path}")


if __name__ == "__main__":
    main()
