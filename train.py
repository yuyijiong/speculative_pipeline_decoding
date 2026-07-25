"""
Train the speculation head for pipelined speculative decoding (v11).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import setproctitle
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import LengthGroupedSampler, get_length_grouped_indices

from pipeline_model import (
    Qwen3SpeculativePipelineModel,
    default_aggr_feature_bound,
    load_base_model_for_pipeline,
    num_hidden_layers_from_hf_config,
)

setproctitle.setproctitle("speculative_pipeline_train")

_SPEC_HEAD_KEY_PREFIXES = ("speculation_head.", "_orig_mod.speculation_head.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)
os.environ["WANDB_PROJECT"] = "pipeline_decoding"
os.environ["WANDB_MODE"] = "offline"

ENCODE_PIPELINE_CACHE_VERSION = "v11-encode-1"

DEFAULT_TRAIN_DATA_PATHS: List[str] = []


def _sanitize_model_name_for_path(model_name: str) -> str:
    s = str(model_name).strip().replace("\\", "/")
    s = os.path.basename(s.rstrip("/"))
    for ch in (":", "<", ">", "|", "*", "?", '"', "/", "\\"):
        s = s.replace(ch, "_")
    return s or "model"


def _parse_optional_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_aggr_feature_bound(s: Optional[str]) -> List[int]:
    if s is None or not str(s).strip():
        raise ValueError("--aggr_feature_bound is required when not using auto mode.")
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def init_speculation_lm_head_from_base(pipeline_model: Qwen3SpeculativePipelineModel) -> None:
    base_w = pipeline_model.lm_head.weight
    spec_w = pipeline_model.speculation_head.lm_head.weight
    if spec_w.shape[1] != base_w.shape[1]:
        raise ValueError(
            f"Hidden dim mismatch between base/spec lm_head: {tuple(base_w.shape)} vs {tuple(spec_w.shape)}"
        )

    with torch.no_grad():
        if spec_w.shape[0] == base_w.shape[0]:
            copied = base_w
            copy_mode = "full_vocab"
        else:
            k = spec_w.shape[0]
            draft_ids = getattr(pipeline_model, "_draft_token_ids", None)
            if draft_ids is not None:
                copied = base_w.index_select(0, draft_ids.to(device=base_w.device))
                copy_mode = "draft_token_ids"
            else:
                copied = base_w[:k]
                copy_mode = "prefix_fallback"
            if copied.shape[0] != k:
                raise ValueError(f"Selected rows ({copied.shape[0]}) != speculation vocab size ({k}).")

        spec_w.copy_(copied.to(device=spec_w.device, dtype=spec_w.dtype))

    log.info(
        "Initialized speculation lm_head from base lm_head (%s): base=%s -> spec=%s",
        copy_mode,
        tuple(base_w.shape),
        tuple(spec_w.shape),
    )


def _extract_speculation_head_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not state:
        return {}
    out: Dict[str, Any] = {}
    for k, v in state.items():
        inner: Optional[str] = None
        for p in _SPEC_HEAD_KEY_PREFIXES:
            if k.startswith(p):
                inner = k[len(p) :]
                break
        if inner is not None:
            out[inner] = v.detach().cpu() if isinstance(v, torch.Tensor) else v
    return out


def load_draft_token_ids_from_json(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "token_ids" not in obj:
        raise KeyError(f"{path!r} must contain a 'token_ids' list.")
    ids = [int(x) for x in obj["token_ids"]]
    if "draft_vocab_size" in obj and int(obj["draft_vocab_size"]) != len(ids):
        raise ValueError(
            f"draft_vocab_size ({obj['draft_vocab_size']}) != len(token_ids) ({len(ids)}) in {path!r}"
        )
    return ids


def _model_type_looks_like_qwen(model_type: Optional[str]) -> bool:
    if not model_type:
        return False
    t = str(model_type).lower()
    return t.startswith("qwen")


def _model_type_is_qwen35(model_type: Optional[str]) -> bool:
    t = (model_type or "").lower()
    return "qwen3_5" in t


def apply_qwen3_file_chat_template(
    tokenizer, generation_path: str, template_dir: Optional[str] = None, *, model_type: Optional[str] = None
) -> None:
    if model_type is not None and not _model_type_looks_like_qwen(model_type):
        log.info("Skip Qwen file chat template (model_type=%s).", model_type)
        return
    if template_dir is None:
        template_dir = "."
    if model_type is not None and _model_type_is_qwen35(model_type):
        path = os.path.join(template_dir, "qwen3.5-template")
        if not os.path.isfile(path):
            log.warning("Qwen3.5 chat template not found: %s (use tokenizer default).", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
        log.info("Loaded Qwen3.5 chat template from %s (model_type=%s).", path, model_type)
        return
    base = os.path.join(template_dir, "qwen3-{}-template")
    if "instruct" in generation_path.lower():
        path = base.format("instruct")
    elif "thinking" in generation_path.lower():
        path = base.format("thinking")
    else:
        path = base.format("nonthink")
    if not os.path.isfile(path):
        log.warning("Qwen chat template file not found: %s (use tokenizer default).", path)
        return
    with open(path, "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()


class StepAlignedLengthGroupedSampler(LengthGroupedSampler):
    def __init__(self, *args, num_replicas: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_replicas = max(1, int(num_replicas))

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        if self.num_replicas <= 1:
            return iter(indices)

        batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        n = self.num_replicas
        full_rounds = len(batches) // n
        rounds = [batches[i * n : (i + 1) * n] for i in range(full_rounds)]
        tail = batches[full_rounds * n :]

        if rounds:
            if self.generator is not None:
                perm = torch.randperm(len(rounds), generator=self.generator).tolist()
            else:
                perm = torch.randperm(len(rounds)).tolist()
            rounds = [rounds[i] for i in perm]
            shuffled_rounds = []
            for round_batches in rounds:
                if self.generator is not None:
                    in_round_perm = torch.randperm(len(round_batches), generator=self.generator).tolist()
                else:
                    in_round_perm = torch.randperm(len(round_batches)).tolist()
                shuffled_rounds.append([round_batches[i] for i in in_round_perm])
            rounds = shuffled_rounds

        ordered_batches = [b for r in rounds for b in r] + tail
        return iter([i for b in ordered_batches for i in b])


class SpeculationHeadTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux_kl_sum = 0.0
        self._aux_total_sum = 0.0
        self._aux_acc_sum = 0.0
        self._aux_n = 0

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None

        if getattr(self.args, "train_sampling_strategy", None) != "group_by_length":
            return super()._get_train_sampler(train_dataset)

        lengths = None
        length_col = getattr(self.args, "length_column_name", "length")
        if hasattr(train_dataset, "column_names") and length_col in train_dataset.column_names:
            lengths = train_dataset[length_col]

        model_input_name = self.processing_class.model_input_names[0] if self.processing_class is not None else None
        world_size = int(getattr(self.args, "world_size", 1))
        return StepAlignedLengthGroupedSampler(
            batch_size=int(self.args.per_device_train_batch_size),
            dataset=train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
            generator=None,
            num_replicas=world_size,
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:  # type: ignore[override]
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "speculation_head.pt")
        save_speculation_head_checkpoint(self, save_path, state_dict=state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        outputs = model(**inputs)
        loss = outputs["loss"]
        if isinstance(outputs, dict) and "kl_loss" in outputs:
            self._aux_kl_sum += float(outputs["kl_loss"])
            self._aux_total_sum += float(loss.detach())
            if "acc" in outputs:
                self._aux_acc_sum += float(outputs["acc"])
            self._aux_n += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:  # type: ignore[override]
        if self._aux_n > 0 and "loss" in logs and "learning_rate" in logs:
            kl_sum = self._aux_kl_sum
            tot_sum = self._aux_total_sum
            acc_sum = self._aux_acc_sum
            n = float(self._aux_n)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                t = torch.tensor([kl_sum, tot_sum, acc_sum, n], dtype=torch.float64, device=self.args.device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                kl_sum, tot_sum, acc_sum, n = (t[0].item(), t[1].item(), t[2].item(), t[3].item())
            denom = max(n, 1.0)
            logs["kl_loss"] = kl_sum / denom
            logs["total_loss"] = tot_sum / denom
            logs["acc"] = acc_sum / denom
            self._aux_kl_sum = self._aux_total_sum = self._aux_acc_sum = 0.0
            self._aux_n = 0
        super().log(logs, start_time=start_time)


class SpeculationHeadTrainerWrapper(nn.Module):
    def __init__(
        self,
        pipeline_model: Qwen3SpeculativePipelineModel,
        temperature: float = 1.0,
        trained_with_use_deepest: bool = False,
    ):
        super().__init__()
        object.__setattr__(self, "pipeline_model", pipeline_model)
        self.temperature = temperature
        self.trained_with_use_deepest = bool(trained_with_use_deepest)
        self.speculation_head = pipeline_model.speculation_head

    def _sample_simulated_pipeline_fill(self, *, device: torch.device) -> int:
        n = int(self.pipeline_model.num_stages)
        if (not self.trained_with_use_deepest) or n <= 1:
            return n

        fill = torch.empty(1, device=device, dtype=torch.long)
        dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()

        def _sample_local() -> int:
            use_staircase = bool((torch.rand((), device=device) < 0.5).item())
            if use_staircase:
                return n
            return int(torch.randint(1, n, (1,), device=device, dtype=torch.long).item())

        if dist_ready:
            if torch.distributed.get_rank() == 0:
                fill.fill_(_sample_local())
            torch.distributed.broadcast(fill, src=0)
        else:
            fill.fill_(_sample_local())
        return int(fill.item())

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        embed_device = self.pipeline_model.embed_tokens.weight.device
        input_ids = input_ids.to(embed_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(embed_device)
        if labels is not None:
            labels = labels.to(embed_device)

        simulated_pipeline_fill = kwargs.pop("simulated_pipeline_fill", None)
        if simulated_pipeline_fill is None:
            simulated_pipeline_fill = self._sample_simulated_pipeline_fill(device=embed_device)

        return self.pipeline_model.training_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            temperature=self.temperature,
            simulated_pipeline_fill=int(simulated_pipeline_fill),
        )


def _qwen35_plain_user_query_ok(messages: List[Dict[str, str]]) -> bool:
    multi_step_tool = True
    for message in reversed(messages):
        if not multi_step_tool:
            break
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        is_tool_round = content.startswith("<tool_response>") and content.endswith("</tool_response>")
        if not is_tool_round:
            multi_step_tool = False
    return not multi_step_tool


def _validate_hf_conversations(conversations: Any, *, context: str) -> List[Dict[str, str]]:
    if not isinstance(conversations, list):
        raise ValueError(f"{context}: `conversations` must be a list, got {type(conversations)}")
    out: List[Dict[str, str]] = []
    for idx, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            try:
                turn = dict(turn)
            except (TypeError, ValueError):
                raise ValueError(f"{context}: turn[{idx}] is not a dict-like object") from None
        if "from" in turn or "value" in turn:
            raise ValueError(
                f"{context}: ShareGPT-style `from/value` is not supported. Please use role/content format."
            )
        if "role" not in turn or "content" not in turn:
            raise ValueError(
                f"{context}: turn[{idx}] must contain `role` and `content`, got keys={list(turn.keys())}"
            )
        out.append({"role": str(turn["role"]), "content": str(turn["content"])})
    return out


def encode_conversation(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    mask_assistant: bool = False,
    model_type: Optional[str] = None,
) -> Dict[str, Any]:
    _empty = {"input_ids": None, "attention_mask": None, "labels": None}
    messages: Optional[List[Dict[str, str]]] = example.get("conversations") or example.get("messages")
    if not messages:
        return _empty
    messages = _validate_hf_conversations(messages, context="encode_conversation")
    if model_type is not None and _model_type_is_qwen35(model_type) and not _qwen35_plain_user_query_ok(
        messages
    ):
        return _empty
    common: Dict[str, Any] = {
        "add_generation_prompt": False,
        "tokenize": True,
        "truncation": False,
        "return_dict": True,
    }
    if mask_assistant:
        common["return_assistant_tokens_mask"] = True
    common["enable_thinking"] = False

    try:
        enc = tokenizer.apply_chat_template(messages, **common)
    except TypeError:
        common.pop("enable_thinking", None)
        try:
            enc = tokenizer.apply_chat_template(messages, **common)
        except TypeError:
            common.pop("return_assistant_tokens_mask", None)
            if mask_assistant:
                log.warning(
                    "return_assistant_tokens_mask not supported for this tokenizer; "
                    "training on all tokens (model_type=%s).",
                    model_type,
                )
            enc = tokenizer.apply_chat_template(messages, **common)
    except Exception as exc:
        roles_str = [m.get("role") for m in messages]
        roles_repr = [repr(m.get("role")) for m in messages]
        dbg = (
            f"encode_conversation: apply_chat_template failed: {exc}\n"
            f"  message_count={len(messages)}\n"
            f"  roles (as in data): {roles_str!r}\n"
            f"  roles (repr per turn): {roles_repr!r}"
        )
        print(dbg, flush=True)
        log.error("%s", dbg)
        raise ValueError(dbg) from exc

    labels = enc["input_ids"].copy()
    if mask_assistant and "assistant_masks" in enc:
        for i, mask in enumerate(enc["assistant_masks"]):
            if mask == 0:
                labels[i] = -100
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def _truncate_encoded_to_max_length(example: Dict[str, Any], max_length: int) -> Dict[str, Any]:
    ids = example.get("input_ids")
    if ids is None or len(ids) <= max_length:
        return example
    return {
        "input_ids": ids[:max_length],
        "attention_mask": example["attention_mask"][:max_length],
        "labels": example["labels"][:max_length],
    }


def _encoded_example_length_ok(
    ex: Dict[str, Any],
    *,
    min_length: int,
    max_length: int,
    max_length_overflow: str,
) -> bool:
    ids = ex.get("input_ids")
    if ids is None:
        return False
    n = len(ids)
    if n < min_length:
        return False
    if max_length_overflow == "discard" and n > max_length:
        return False
    return True


def _get_file_paths(directory: str, suffix: str) -> List[str]:
    paths: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(suffix):
                paths.append(os.path.join(root, fname))
    paths.sort()
    return paths


def load_conversation_dataset(data_path: str, *, num_proc: int = 1) -> Dataset:
    del num_proc
    if os.path.isdir(data_path):
        parquet_files = _get_file_paths(data_path, ".parquet")
        json_files = _get_file_paths(data_path, ".jsonl") + _get_file_paths(data_path, ".json")
        datasets: List[Dataset] = []
        for fp in parquet_files:
            log.info("Loading parquet: %s", fp)
            datasets.append(Dataset.from_parquet(fp))
        for fp in json_files:
            log.info("Loading json: %s", fp)
            datasets.append(Dataset.from_json(fp))
        if not datasets:
            raise FileNotFoundError(f"No .parquet / .json / .jsonl files found in {data_path}")
        if len(datasets) == 1:
            return datasets[0]
        unified = [unify_conversation_dataset(d, num_proc=num_proc) for d in datasets]
        return concatenate_datasets(unified)

    if data_path.endswith(".parquet"):
        log.info("Loading parquet: %s", data_path)
        return Dataset.from_parquet(data_path)
    if data_path.endswith((".json", ".jsonl")):
        log.info("Loading json: %s", data_path)
        return Dataset.from_json(data_path)
    raise ValueError(f"Unsupported data format for path: {data_path}")


def unify_conversation_dataset(ds: Dataset, *, num_proc: int) -> Dataset:
    del num_proc
    cols = list(ds.column_names)
    if "conversations" in cols:
        ds = ds.select_columns(["conversations"])
    elif "messages" in cols:
        ds = ds.rename_column("messages", "conversations")
        ds = ds.select_columns(["conversations"])
    else:
        raise ValueError(
            "Dataset must have a `conversations` or `messages` column; "
            f"got columns: {cols!r}"
        )
    return ds


def load_mixed_training_datasets(data_paths: List[str], *, num_proc: int) -> Dataset:
    if not data_paths:
        raise ValueError("data_paths must be non-empty")
    parts: List[Dataset] = []
    for p in data_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Training data path not found: {p}")
        raw = load_conversation_dataset(p, num_proc=num_proc)
        log.info("Loaded %d raw rows from %s", len(raw), p)
        raw = unify_conversation_dataset(raw, num_proc=num_proc)
        parts.append(raw)
    if len(parts) == 1:
        return parts[0]
    out = concatenate_datasets(parts)
    log.info("Concatenated %d sources -> %d total rows", len(parts), len(out))
    return out


def _collect_data_source_stats(data_paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in data_paths:
        p = os.path.normpath(os.path.abspath(raw))
        if not os.path.exists(p):
            raise FileNotFoundError(f"Training data path not found: {raw}")
        if os.path.isdir(p):
            parquet_files = _get_file_paths(p, ".parquet")
            json_files = _get_file_paths(p, ".jsonl") + _get_file_paths(p, ".json")
            files = sorted(parquet_files + json_files)
        elif os.path.isfile(p):
            files = [p]
        else:
            raise ValueError(f"Unsupported training data path: {raw!r}")
        for fp in files:
            fp_abs = os.path.normpath(os.path.abspath(fp))
            st = os.stat(fp_abs)
            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
            rows.append({"path": fp_abs, "mtime_ns": int(mtime_ns), "size": int(st.st_size)})
    rows.sort(key=lambda r: r["path"])
    return rows


def _chat_template_sha256(tokenizer: AutoTokenizer) -> str:
    tpl = getattr(tokenizer, "chat_template", None)
    blob = tpl if isinstance(tpl, str) else repr(tpl)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_encoded_dataset_cache_key(
    *,
    encode_pipeline_version: str,
    model_name: str,
    data_paths: List[str],
    tokenizer: AutoTokenizer,
    mask_assistant: bool,
    model_type: Optional[str],
    max_length: int,
    min_length: int,
    max_length_overflow: str,
) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "encode_pipeline_version": encode_pipeline_version,
        "model_name": os.path.normpath(os.path.abspath(str(model_name))),
        "data_sources": _collect_data_source_stats(data_paths),
        "chat_template_sha256": _chat_template_sha256(tokenizer),
        "mask_assistant": bool(mask_assistant),
        "model_type": model_type,
        "max_length": int(max_length),
        "min_length": int(min_length),
        "max_length_overflow": str(max_length_overflow),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return fingerprint, payload


def encode_raw_to_filtered_dataset(
    raw_ds: Dataset,
    tokenizer: AutoTokenizer,
    *,
    mask_assistant: bool,
    model_type: Optional[str],
    max_length: int,
    min_length: int,
    max_length_overflow: str,
    num_proc: int,
) -> Dataset:
    encode_fn = partial(
        encode_conversation,
        tokenizer=tokenizer,
        mask_assistant=mask_assistant,
        model_type=model_type,
    )
    encoded_ds_uf = raw_ds.map(
        encode_fn,
        remove_columns=raw_ds.column_names,
        num_proc=num_proc,
        desc="Encoding conversations",
        load_from_cache_file=True,
    )
    if max_length_overflow == "truncate":
        encoded_ds_uf = encoded_ds_uf.map(
            partial(_truncate_encoded_to_max_length, max_length=max_length),
            num_proc=num_proc,
            desc="Truncating sequences longer than max_length",
            load_from_cache_file=True,
        )

    encoded_ds = encoded_ds_uf.filter(
        partial(
            _encoded_example_length_ok,
            min_length=min_length,
            max_length=max_length,
            max_length_overflow=max_length_overflow,
        ),
        num_proc=num_proc,
        desc="Filtering by validity and length",
    )
    return encoded_ds


def _aggr_bound_arg_means_auto(s: Optional[str]) -> bool:
    if s is None:
        return True
    t = str(s).strip().lower()
    return t in ("", "none", "auto")


def compute_auto_aggr_feature_bound(num_layers: int, num_stages: int) -> List[int]:
    return default_aggr_feature_bound(num_layers, num_stages)


def auto_output_dir(
    *,
    model_name: str,
    num_datasets: int,
    num_spec_layers: int,
    num_stages: int,
    draft_vocab_size: int,
    temperature: float,
) -> str:
    m = _sanitize_model_name_for_path(model_name)
    vocab_k = f"{max(1, draft_vocab_size // 1000)}k"
    return os.path.join(
        ".",
        "training_output",
        f"pipeline_spd_{m}_datanum={num_datasets}_layer={num_spec_layers}_"
        f"stage={num_stages}_vocab={vocab_k}_temp={temperature:.2f}_use_deepest",
    )


def save_speculation_head_checkpoint(
    trainer: SpeculationHeadTrainer,
    path: str,
    state_dict: Optional[Dict[str, Any]] = None,
) -> None:
    wrapped = trainer.model
    inner = trainer.accelerator.unwrap_model(wrapped, keep_torch_compile=False)
    pm = inner.pipeline_model

    if (
        state_dict is None
        and getattr(trainer, "is_fsdp_enabled", False)
        and "FULL_STATE_DICT" in str(trainer.accelerator.state.fsdp_plugin.state_dict_type)
    ):
        state_dict = trainer.accelerator.get_state_dict(wrapped)

    if not trainer.args.should_save:
        return

    head_sd = _extract_speculation_head_state(state_dict)
    if head_sd:
        torch.save(
            {
                "state_dict": head_sd,
                "config": {
                    "hidden_size": pm.hidden_size,
                    "vocab_size": pm.vocab_size,
                    "model_type": getattr(pm.config, "model_type", None),
                    "base_model_path": str(getattr(pm, "base_model_path", "")),
                    "draft_vocab_size": getattr(pm, "draft_vocab_size", pm.vocab_size),
                    "num_stages": pm.num_stages,
                    "num_spec_layers": pm.num_spec_layers,
                    "version": 11,
                    "trained_with_use_deepest": bool(getattr(pm, "trained_with_use_deepest", False)),
                    "aggr_feature_bound": list(pm.aggr_feature_bound),
                    "num_aggr_types": int(pm.num_aggr_types),
                    **(
                        {"spec_init_from_base_layers": pm.spec_init_from_base_layers}
                        if getattr(pm, "spec_init_from_base_layers", None)
                        else {}
                    ),
                    **(
                        {"draft_token_ids": pm._draft_token_ids.detach().cpu().tolist()}
                        if getattr(pm, "_use_draft_vocab", False)
                        else {}
                    ),
                    "spec_intermediate_size_fallback": int(
                        getattr(pm, "spec_intermediate_size_fallback", 9216)
                    ),
                },
            },
            path,
        )
        log.info("Saved speculation head checkpoint -> %s", path)
        return

    pm.save_speculation_head(path)
    log.info("Saved speculation head checkpoint -> %s", path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train speculation head for pipelined speculative decoding (v11)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--data_path", action="append", default=None, metavar="PATH")
    p.add_argument("--num_stages", type=int, default=8)
    p.add_argument("--num_spec_layers", type=int, default=4)
    p.add_argument("--spec_init_from_base_layers", type=str, default=None)#"18,24,29,33")
    p.add_argument(
        "--aggr_feature_bound",
        type=str,
        default="0,8,16,24,31",
        help="Comma-separated HF hidden-state anchor indices for g_0..g_{m-1}. Empty, 'none', or 'auto' "
        "enables automatic bounds from decoder layer count and num_stages (see compute_auto_aggr_feature_bound).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.00)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--mask_assistant", type=bool, default=True)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_length_overflow", type=str, choices=("discard", "truncate"), default="discard")
    p.add_argument("--min_length", type=int, default=10)
    p.add_argument("--num_proc", type=int, default=40)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument(
        "--wandb_log_subdir",
        type=str,
        default="wandb",
        help="Subdirectory under --output_dir where Weights & Biases stores run data (WANDB_DIR).",
    )
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--draft_vocab_json", type=str, default="")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument(
        "--use_bnb_quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the frozen base model in 4-bit via bitsandbytes (NF4) to reduce GPU memory.",
    )
    p.add_argument(
        "--spec_intermediate_size_fallback",
        type=int,
        default=9216,
        help=(
            "Dense MLP intermediate_size for the speculation tower when the base config lacks "
            "intermediate_size (e.g. Qwen3.5/3.6 MoE text_config only has moe_intermediate_size)."
        ),
    )
    p.add_argument("--chat_template_dir", type=str, default=".")
    p.add_argument("--encoded_dataset_cache_dir", type=str, default="./dataset_cache/pipeline_spd_encoded")
    p.add_argument("--force_rebuild_encoded_cache", action="store_true")
    p.add_argument(
        "--trained_with_use_deepest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If True, train with mixed simulated pipeline fill states: 50% full staircase (fill=n), "
            "50% random fill in [1, n-1] using deepest-available approximation."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec_init_from_base = _parse_optional_int_list(args.spec_init_from_base_layers)
    if spec_init_from_base is not None and len(spec_init_from_base) != args.num_spec_layers:
        raise ValueError(
            f"--spec_init_from_base_layers must have length --num_spec_layers ({args.num_spec_layers}), "
            f"got {len(spec_init_from_base)}"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map: Any = {"": local_rank} if torch.cuda.is_available() else "cpu"

    hf_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    num_dec_layers = num_hidden_layers_from_hf_config(hf_config)
    if _aggr_bound_arg_means_auto(args.aggr_feature_bound):
        aggr_feature_bound = compute_auto_aggr_feature_bound(num_dec_layers, args.num_stages)
        log.info(
            "Auto aggr_feature_bound (num_layers=%s, num_stages=%s): %s",
            num_dec_layers,
            args.num_stages,
            aggr_feature_bound,
        )
    else:
        aggr_feature_bound = _parse_aggr_feature_bound(args.aggr_feature_bound)

    model_type = getattr(hf_config, "model_type", None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_qwen3_file_chat_template(
        tokenizer, args.model_name, template_dir=args.chat_template_dir, model_type=model_type
    )
    if args.use_bnb_quant:
        log.info("Loading base model with bitsandbytes 4-bit quantization (NF4).")
    base_model = load_base_model_for_pipeline(
        args.model_name,
        dtype="auto",
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        use_bnb_quant=args.use_bnb_quant,
    )

    draft_token_ids: Optional[List[int]] = None
    if args.draft_vocab_json:
        if not os.path.isfile(args.draft_vocab_json):
            raise FileNotFoundError(f"--draft_vocab_json={args.draft_vocab_json!r} not found.")
        draft_token_ids = load_draft_token_ids_from_json(args.draft_vocab_json)

    pipeline_model = Qwen3SpeculativePipelineModel(
        base_model=base_model,
        num_stages=args.num_stages,
        num_spec_layers=args.num_spec_layers,
        spec_init_from_base_layers=spec_init_from_base,
        draft_token_ids=draft_token_ids,
        aggr_feature_bound=aggr_feature_bound,
        trained_with_use_deepest=args.trained_with_use_deepest,
        spec_intermediate_size_fallback=args.spec_intermediate_size_fallback,
    )
    init_speculation_lm_head_from_base(pipeline_model)

    data_paths_for_name = list(args.data_path) if args.data_path else list(DEFAULT_TRAIN_DATA_PATHS)
    num_datasets = len(data_paths_for_name)
    draft_vocab_size = int(getattr(pipeline_model, "draft_vocab_size", pipeline_model.vocab_size))
    if not args.output_dir:
        args.output_dir = auto_output_dir(
            model_name=args.model_name,
            num_datasets=num_datasets,
            num_spec_layers=args.num_spec_layers,
            num_stages=args.num_stages,
            draft_vocab_size=draft_vocab_size,
            temperature=args.temperature,
        )

    output_dir_abs = os.path.normpath(os.path.abspath(args.output_dir))
    os.makedirs(output_dir_abs, exist_ok=True)
    wandb_dir = os.path.normpath(os.path.join(output_dir_abs, args.wandb_log_subdir))
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir

    wrapper = SpeculationHeadTrainerWrapper(
        pipeline_model,
        temperature=args.temperature,
        trained_with_use_deepest=args.trained_with_use_deepest,
    )
    training_args = TrainingArguments(
        report_to="wandb",
        run_name=args.output_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=100,
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,
        logging_steps=args.log_interval,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=10,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        save_only_model=True,
        train_sampling_strategy="group_by_length",
    )

    data_paths = data_paths_for_name
    fingerprint, cache_manifest = compute_encoded_dataset_cache_key(
        encode_pipeline_version=ENCODE_PIPELINE_CACHE_VERSION,
        model_name=args.model_name,
        data_paths=data_paths,
        tokenizer=tokenizer,
        mask_assistant=args.mask_assistant,
        model_type=model_type,
        max_length=args.max_length,
        min_length=args.min_length,
        max_length_overflow=args.max_length_overflow,
    )
    cache_root = os.path.normpath(os.path.abspath(args.encoded_dataset_cache_dir))
    bundle_dir = os.path.join(cache_root, fingerprint)
    encoded_disk_path = os.path.join(bundle_dir, "encoded_dataset")
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with training_args.main_process_first(desc="pipeline v11 dataset preprocessing"):
        if training_args.local_process_index == 0:
            if args.force_rebuild_encoded_cache and os.path.isdir(bundle_dir):
                shutil.rmtree(bundle_dir)
            if not os.path.isdir(encoded_disk_path):
                raw_ds = load_mixed_training_datasets(data_paths, num_proc=args.num_proc)
                encoded_ds = encode_raw_to_filtered_dataset(
                    raw_ds,
                    tokenizer,
                    mask_assistant=args.mask_assistant,
                    model_type=model_type,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    max_length_overflow=args.max_length_overflow,
                    num_proc=args.num_proc,
                )
                if len(encoded_ds) == 0:
                    raise RuntimeError("No valid training examples after encoding.")
                os.makedirs(bundle_dir, exist_ok=True)
                encoded_ds.save_to_disk(encoded_disk_path)
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    json.dump({"fingerprint": fingerprint, "payload": cache_manifest}, mf, indent=2, ensure_ascii=False)

    encoded_ds = load_from_disk(encoded_disk_path)
    if len(encoded_ds) == 0:
        raise RuntimeError("No valid training examples after encoding.")
    dataset = encoded_ds.shuffle(seed=42)

    trainer = SpeculationHeadTrainer(
        model=wrapper,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    trainer.train()
    final_path = os.path.join(args.output_dir, "speculation_head_final.pt")
    save_speculation_head_checkpoint(trainer, final_path, state_dict=None)
    if trainer.args.should_save:
        log.info("Training complete. Speculation head saved to %s", final_path)


if __name__ == "__main__":
    main()

