import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
from lm_eval.api.task import Task
Task.supports_multimodal = True
from lm_eval.api.instance import Instance

try:                                      # ≥ v0.4.0
    from lm_eval.api.registry import register_task
except ImportError:                       # ≤ v0.3.x
    from lm_eval.tasks.registry import register_task

try:
    from lm_eval.api.metrics import mean
except ImportError:
    from lm_eval.metrics import mean

import json
import re



def parse_answer(text: str) -> str:
    """
    Parses a raw model prediction string to extract one or more 
    answer letters (A–E), preserving order and uniqueness.
    """
    # Split off any leading context up to the last "Правильный ответ"
    parts = re.split(r'Правильный ответ[:\-\s]*', text, flags=re.IGNORECASE)
    candidate = parts[-1]
    # Find all standalone letters A–E
    letters = re.findall(r'\b[A-E]\b', candidate)
    # Deduplicate while preserving order
    seen = []
    for l in letters:
        if l not in seen:
            seen.append(l)
    return ", ".join(seen)


@register_task("med_mc")
class MedMC(Task):
    VERSION = 0
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/ssu_medQ_fundamental.jsonl")
    DATASET_NAME = "med_mc"
    OUTPUT_TYPE = "generate_until"
    task_name = "med_mc"

    def download(self, *args, **kwargs):
        # no-op: we load directly from DATA_PATH below
        return
        
    # ------------------------- docs -----------------------------------------
    def __init__(self):
        super().__init__()           # download() is now a no-op
        with open(self.DATA_PATH, "r", encoding="utf-8") as f:
            self._docs = [json.loads(line) for line in f]

    def has_training_docs(self):   return False
    def has_validation_docs(self): return False
    def has_test_docs(self):       return True
    def test_docs(self):           return self._docs

    # ------------------------ prompting -------------------------------------
    def doc_to_text(self, doc):
        # join all message strings exactly as authored
        prompt = "\n".join(m["content"].rstrip() for m in doc["context"])
        # be sure it ends with “Правильный ответ:”
        if not prompt.endswith("Правильный ответ:"):
            prompt += " Правильный ответ:"
        return prompt

    def doc_to_target(self, doc):
        return doc["answer"].strip()

    def doc_to_metadata(self, doc):
        return {"subset": doc["subset"]}
    
    def construct_requests(self, doc, ctx, **kwargs):
        """
        Build one `generate_until` request.  Pop out harness-only kwargs,
        and pass only metadata into the Instance.
        """
        from copy import deepcopy

        # Extract and remove the harness-only kwargs:
        metadata = kwargs.pop("metadata")
        kwargs.pop("apply_chat_template", None)
        kwargs.pop("chat_template", None)

        # Clone the generation settings (from your YAML + CLI)
        gen_kwargs = deepcopy(self.config.generation_kwargs)
        # Ensure your default "until" is set if none provided
        gen_kwargs.setdefault("until", ["\n\n"])

        # Wrap in a list so build_all_requests sees it as a group
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, gen_kwargs),
                idx=0,
                metadata=metadata,
            )
        ]

    def process_results(self, doc, results):
        raw_pred = results[0]  # **don’t** .lower() or strip off punctuation yet
        pred     = parse_answer(raw_pred)           # e.g. "C" or "A, B, D"
        gold     = parse_answer(doc["answer"])      # also e.g. "C" or "A, B, D"
        return {"acc": 1.0 if pred == gold else 0.0}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


    def fewshot_context(self, *args, **kwargs):
        doc, num_fewshot = args[0], args[1]
        rnd         = kwargs.get("rnd", None)
        description = kwargs.get("description", None)
        return super().fewshot_context(
            doc,
            num_fewshot,
            rnd=rnd,
            description=description,
        )