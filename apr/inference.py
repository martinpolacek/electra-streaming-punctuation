import json
import torch
from .artifacts import load_bundle
from .cache import KVCache, confidence_score
from .calibration import load_gate, gate_scores, check_identity
from .data import windows
from .model import target_logits

class Predictor:
    """Single-recording streaming windows, batch size one, four future words.

    Each predict_words call resets history. No labels or gate statistics survive
    into another recording. Input tokenization is offline; encoder decisions see
    exactly the current window, including at most four following words.
    """
    def __init__(self, model_path, mode="full", policy_path=None, gate_path=None, tau=None, window=64, device="cpu"):
        if mode not in ("full", "exit", "fast", "gate", "margin", "entropy"):
            raise ValueError("Unknown inference mode")
        self.model, self.tokenizer, self.document = load_bundle(model_path, device, "punctuation")
        self.device, self.mode, self.window = device, mode, window
        if window not in (32, 64, 128):
            raise ValueError("Paper windows are 32, 64 and 128 subwords")
        if mode in ("fast", "gate", "margin", "entropy") and window != 64:
            raise ValueError("Published KV-cache policies require a 64-subword window")
        if tau is not None and (mode != "exit" or policy_path is not None or not 0 <= tau <= 1):
            raise ValueError("Use --tau only for uncalibrated exit, within [0,1]")
        if mode == "exit" and len(self.model.exits) < 2:
            raise ValueError("Train intermediate exit heads before selecting exit mode")
        self.threshold = 0.99 if mode == "exit" and tau is None else tau
        self.gate_state = None
        if mode in ("gate", "margin", "entropy") and not policy_path:
            raise ValueError("A frozen calibration policy is required")
        if policy_path:
            if mode in ("full", "fast"):
                raise ValueError("Full and fast modes do not use calibrated policies")
            with open(policy_path, encoding="utf-8") as f:
                policy = json.load(f)
            identity = {"model_sha256": self.document["weights_sha256"], "tokenizer_sha256": self.document["tokenizer_sha256"],
                        "architecture_sha256": self.document["architecture_sha256"], "window": window, "lookahead": 4, "anchor_limit": 128}
            check_identity(identity, policy["metadata"])
            if policy.get("format_version") != 1 or mode not in policy["policies"]:
                raise ValueError("Policy does not contain this inference mode")
            selected = policy["policies"][mode]
            if not selected.get("feasible"):
                raise ValueError("Policy did not satisfy calibration constraints")
            self.threshold = selected["threshold"]
            if mode == "gate":
                if not gate_path:
                    raise ValueError("Learned gate inference requires a gate directory")
                gate, mean, scale, gate_doc = load_gate(gate_path, device)
                check_identity(identity, gate_doc)
                if gate_doc["weights_sha256"] != policy["gate_sha256"]:
                    raise ValueError("Calibrated gate does not match supplied gate weights")
                self.gate_state = gate, mean, scale
        if gate_path and mode != "gate":
            raise ValueError("--gate is only used by learned gate mode")

    @torch.inference_mode()
    def predict_words(self, words):
        cache = KVCache(self.model) if self.mode in ("fast", "gate", "margin", "entropy") else None
        for index, item in enumerate(windows(words, self.tokenizer, self.window, device=self.device)):
            ids, a, b, word_start, target, ends = item
            mask = torch.ones_like(ids, dtype=torch.bool)
            targets = torch.tensor([target-a], dtype=torch.long, device=self.device)
            repair = False
            score = None
            if cache is None:
                logits, depth = target_logits(self.model, ids, mask, targets, self.threshold if self.mode == "exit" else None)
                depth = int(depth[0])
            else:
                logits, features = cache.step(*item)
                depth = self.model.config.num_layers
                if self.mode != "fast":
                    score = gate_scores(features, *self.gate_state) if self.mode == "gate" else confidence_score(features, self.mode)
                    repair = bool(score >= self.threshold)
                    if repair:
                        # Fresh positions 0..window_length-1. No cache mutation.
                        logits, _ = target_logits(self.model, ids, mask, targets)
            result = {"word_index": index, "word": words[index], "label": int(logits.argmax(-1).item()),
                      "depth": depth, "repair": repair}
            if score is not None:
                result["score"] = float(score)
            if cache is not None:
                result.update(cache.last_stats)
            yield result
