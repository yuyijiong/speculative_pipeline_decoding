"""
Evaluate pipeline decoding on MT-Bench, HumanEval, and GSM8K-style question files.

Each row under ``eval_data`` uses the EAGLE ``question.jsonl`` shape: ``turns`` is a list
of user turns; we take **only the first turn** as the prompt (one generation per row).
**All** rows from each dataset under ``--data_dir`` are evaluated, in config order
(``DATASET_CONFIG``), concatenated into a single indexed prompt list.

Runs always use verification (`verify=True`) so text follows the base-model
decision path under the selected decoding mode.

**Timing:** v9/v10 set ``_last_generate_timing`` (prefill vs decode split, per-stage GPU
time, ideal parallel decode). v5/v6/v7/v8 record only total wall time in
``run_pipeline_generate``; per-sample ``pipeline_decode_wall_sec`` falls back to that
wall time when split timing is missing. With ``--baseline``
(default on), standard HuggingFace ``generate`` is run **once per base model and
temperature** (same prompts as this run); results are stored under
``--baseline_cache_dir`` (default ``<output_dir>/baseline``). Later runs reuse that file
when metadata matches; use ``--force_baseline_recompute`` to refresh. With multiple
``--gpus`` and CUDA, baseline cache fill uses the same multi-process sharding as
pipeline decoding; a single GPU runs baseline in the main process. Pipeline eval for
each spec checkpoint only merges cached baseline metrics per index (no repeated baseline
``generate``).

Outputs under ``--output_dir`` (default ``./pipeline_eval``), split into folders::

    baseline/
        baseline__<model_tag>__t<temp>__...jsonl
            (first line: metadata; following lines: per-index baseline timings for HF ``generate``)
    raw/pipeline_eval__<checkpoint_tag>__nt<total>__per_sample.jsonl
        (single ``--temperature`` and single ``--draft_top_k``; multi-GPU shard files also live here)
    raw/pipeline_eval__<checkpoint_tag>__nt<total>__t<temp_tag>__per_sample.jsonl
        (when multiple ``--temperature`` values are given, one file per temperature)
    raw/...__dtk<k>__per_sample.jsonl
        (when multiple ``--draft_top_k`` values; combined with ``__t...`` when both vary)
    summary/pipeline_eval__<checkpoint_tag>__nt<total>__summary.json
        (one JSON: ``results`` lists each (temperature, ``draft_top_k``) run's aggregates and ``per_sample_path``;
        if only one temperature and one ``draft_top_k`` are evaluated, top-level ``overall`` / ``temperature`` / ``draft_top_k`` mirror that row for backward compatibility)

The filename ``checkpoint_tag`` is derived from the spec-head checkpoint path
(``<parent>__<stem>``) so different checkpoints do not overwrite each other.

``--spec_head_ckpt`` may be passed multiple times (or as several paths after one flag)
to evaluate each checkpoint separately (same prompts, separate raw/summary files).

Stochastic decoding (``temperature`` > 0) resets the PyTorch RNG **per global prompt
index** from ``--seed`` so repeated evaluations produce identical token samples (same
as fixing the RNG stream for that index).

Reported **aggregate** acceptance rate is ``sum(new_tokens) / sum(decode_loop_steps)``
(pooled over samples): actual generated tokens per decoding step. **Equivalent accept
length** is ``num_stages * acceptance_rate`` (same ``n`` as pipeline stages). Per-sample
``n_accepted`` / ``n_acceptance_flags`` are still logged for draft-verify flags.
Theoretical throughput gain in the summary is averaged with weights ``new_tokens``;
totals include ``decode_loop_steps`` for reference.

``--gpus 0,1,2`` runs one process per listed physical GPU, splits the workload,
merges per-sample results (and summary) when all finish. Single GPU (e.g.
``--gpus 0``) runs in-process without process spawn overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

# Do not import ``pipeline_inference`` at module load: it imports ``torch``,
# and ``torch`` must initialize only after ``CUDA_VISIBLE_DEVICES`` is set in
# ``spawn`` worker processes (see ``_gpu_process_entry`` / ``_baseline_gpu_process_entry``).

DATASET_CONFIG: tuple[tuple[str, str], ...] = (
    ("mt_bench", "mt_bench/question.jsonl"),
    ("humaneval", "humaneval/question.jsonl"),
    ("gsm8k", "gsm8k/question.jsonl"),
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline decoding eval (acceptance & toy speedup)")
    p.add_argument(
        "--spec_head_ckpt",
        nargs="+",
        type=_non_empty_ckpt_path,
        default=[],
        help="One or more speculation_head checkpoints (each evaluated separately)",
    )
    p.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help=(
            "Hugging Face id or local path for the base model. "
            "When non-empty, overrides config['base_model_path'] inside each checkpoint."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./eval_output",
        help="Root directory; writes raw per-sample jsonl under raw/ and summary json under summary/",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="eval_data",
        help="Directory containing mt_bench/, humaneval/, gsm8k/ each with question.jsonl",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; with temperature>0, PyTorch is re-seeded per prompt index so runs are reproducible.",
    )
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=[0.0],
        help="One or more decoding temperatures; runs sequentially and writes one summary.json with all.",
    )
    p.add_argument(
        "--use_deepest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pipeline generate: use deepest snapshots for spec rows.",
    )
    p.add_argument(
        "--draft_top_k",
        nargs="+",
        type=int,
        default=[4],
        help="One or more draft top-k values; runs sequentially (with each --temperature) and writes one summary.json with all.",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default="0",
        help=(
            "Comma-separated physical GPU ids (e.g. 0,1,2). "
            "Uses one process per id and splits the dataset across them. "
            "A single id (e.g. 0) runs in the main process without spawning."
        ),
    )
    p.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When True (default), merge standard HuggingFace generate timings from the baseline cache (see --baseline_cache_dir) for speed comparison.",
    )
    p.add_argument(
        "--baseline_cache_dir",
        type=str,
        default=None,
        help="Directory for cached baseline HF generate results; default <output_dir>/baseline.",
    )
    p.add_argument(
        "--force_baseline_recompute",
        action="store_true",
        default=False,
        help="Ignore existing baseline cache files and recompute them (overwrites).",
    )

    p.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Use raw question string without apply_chat_template (instruct models usually need template).",
    )
    p.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using chat template, pass enable_thinking to tokenizer.apply_chat_template (e.g. Qwen3).",
    )
    return p.parse_args()


@dataclass
class UnifiedItem:
    dataset: str
    question_id: int
    question: str


def _item_to_dict(u: UnifiedItem) -> dict[str, Any]:
    return {"dataset": u.dataset, "question_id": u.question_id, "question": u.question}


def _item_from_dict(d: dict[str, Any]) -> UnifiedItem:
    return UnifiedItem(
        dataset=str(d["dataset"]),
        question_id=int(d["question_id"]),
        question=str(d["question"]),
    )


def _parse_gpus(s: str) -> list[int]:
    t = (s or "").strip()
    if not t:
        return []
    out: list[int] = []
    for part in re.split(r"[,\s;]+", t):
        p = part.strip()
        if not p:
            continue
        g = int(p, 10)
        if g < 0:
            raise argparse.ArgumentTypeError(f"Invalid --gpus: negative id {g}")
        out.append(g)
    return out


def _temperature_filename_tag(value: float) -> str:
    s = str(float(value)).replace("-", "neg")
    return s.replace(".", "p")


def _normalize_temperature_list(values: list[float]) -> list[float]:
    return list(dict.fromkeys(float(x) for x in values))


def _normalize_draft_top_k_list(values: list[int]) -> list[int]:
    return list(dict.fromkeys(int(x) for x in values))


def _per_sample_file_stem(
    base_name: str,
    temperature: float,
    draft_top_k: int,
    *,
    temperatures: list[float],
    draft_top_ks: list[int],
) -> str:
    """Stem ``...__per_sample`` (no ``.jsonl``) for raw per-sample outputs."""
    if len(temperatures) <= 1 and len(draft_top_ks) <= 1:
        return f"{base_name}__per_sample"
    tags: list[str] = []
    if len(temperatures) > 1:
        tags.append(f"t{_temperature_filename_tag(float(temperature))}")
    if len(draft_top_ks) > 1:
        tags.append(f"dtk{int(draft_top_k)}")
    return f"{base_name}__{'__'.join(tags)}__per_sample"


def _base_model_path_tag(model_path: str) -> str:
    p = Path(model_path)
    stem = (p.name or "model").strip() or "model"
    parent = (p.parent.name or "pretrained").strip() or "pretrained"
    tag = f"{parent}__{stem}"
    for ch in '<>:"/\\|?*':
        tag = tag.replace(ch, "_")
    tag = re.sub(r"\s+", "_", tag)
    return tag[:200] if len(tag) > 200 else tag


def _data_dir_tag(data_dir: Path) -> str:
    t = str(data_dir.resolve())
    for ch in '<>:"/\\|?*':
        t = t.replace(ch, "_")
    t = re.sub(r"\s+", "_", t)
    return t[:120] if len(t) > 120 else t


def _set_torch_rng_for_eval_sample(base_seed: int, sample_index: int) -> None:
    import torch

    s = int(base_seed) + int(sample_index) * 1_000_003
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _baseline_cache_filename(
    base_model_path: str,
    temperature: float,
    *,
    seed: int,
    max_new_tokens: int,
    no_chat_template: bool,
    enable_thinking: bool,
    data_dir: Path,
    n_total: int,
) -> str:
    temp_tag = _temperature_filename_tag(float(temperature))
    model_tag = _base_model_path_tag(base_model_path)
    dtag = _data_dir_tag(data_dir)
    chat = "0" if no_chat_template else "1"
    think = "1" if enable_thinking else "0"
    return (
        f"baseline__{model_tag}__t{temp_tag}__"
        f"nt{int(n_total)}__seed{int(seed)}__mnt{int(max_new_tokens)}__chat{chat}__think{think}__{dtag}.jsonl"
    )


def _baseline_meta_payload(
    base_model_path: str,
    temperature: float,
    *,
    seed: int,
    max_new_tokens: int,
    no_chat_template: bool,
    enable_thinking: bool,
    data_dir: Path,
    n_total: int,
) -> dict[str, Any]:
    return {
        "__baseline_meta__": 1,
        "base_model_path": str(base_model_path),
        "temperature": float(temperature),
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "no_chat_template": bool(no_chat_template),
        "enable_thinking": bool(enable_thinking),
        "data_dir": str(Path(data_dir).resolve()),
        "n_total": int(n_total),
    }


def _baseline_meta_matches_disk(meta: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = (
        "base_model_path",
        "temperature",
        "seed",
        "max_new_tokens",
        "no_chat_template",
        "enable_thinking",
        "data_dir",
        "n_total",
    )
    for k in keys:
        mv = meta.get(k)
        if k == "enable_thinking" and "enable_thinking" not in meta:
            mv = False
        if mv != expected.get(k):
            return False
    return True


def _read_baseline_cache_jsonl(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    meta: dict[str, Any] = {}
    by_index: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        first = f.readline()
        if not first.strip():
            raise ValueError(f"Empty baseline cache: {path}")
        head = json.loads(first)
        if not isinstance(head, dict) or "__baseline_meta__" not in head:
            raise ValueError(f"Invalid baseline cache (missing meta line): {path}")
        meta = head
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_index[int(row["index"])] = row
    return meta, by_index


def _validate_baseline_cache_indices(by_index: dict[int, dict[str, Any]], n_total: int) -> None:
    if len(by_index) != n_total:
        raise ValueError(
            f"Baseline cache row count {len(by_index)} != n_total {n_total}; delete cache or use --force_baseline_recompute."
        )
    for i in range(n_total):
        if i not in by_index:
            raise ValueError(
                f"Baseline cache missing index {i}; delete cache or use --force_baseline_recompute."
            )


def _build_baseline_stack(base_model_path: str) -> Tuple[Any, Any, Any]:
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    device = next(base_model.parameters()).device
    return tokenizer, base_model, device


def _run_baseline_batch_for_cache(
    indexed: list[tuple[int, UnifiedItem]],
    *,
    base_model_path: str,
    temperature: float,
    max_new_tokens: int,
    no_chat_template: bool,
    enable_thinking: bool,
    eval_seed: int,
) -> list[dict[str, Any]]:
    tokenizer, base_model, device = _build_baseline_stack(base_model_path)
    greedy = bool(float(temperature) == 0.0)
    use_chat = not bool(no_chat_template)
    out: list[dict[str, Any]] = []
    for idx, it in tqdm(
        indexed,
        total=len(indexed),
        desc=f"Baseline HF generate (T={temperature})",
        unit="sample",
    ):
        _set_torch_rng_for_eval_sample(int(eval_seed), int(idx))
        input_ids, attention_mask = _encode_prompt(
            tokenizer, it.question, use_chat, device, enable_thinking=bool(enable_thinking)
        )
        _seq_std, baseline_prefill_sec, _gen_tot, baseline_decode_sec = _baseline_standard_decode_timing(
            base_model,
            tokenizer,
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            temperature=float(temperature),
        )
        prompt_len_b = int(input_ids.shape[1])
        baseline_new_tokens = int(_seq_std.numel() - prompt_len_b)
        baseline_decode_tok_s = (
            float(baseline_new_tokens) / float(baseline_decode_sec) if baseline_decode_sec > 0 else 0.0
        )
        out.append(
            {
                "index": int(idx),
                "baseline_prefill_wall_sec": float(baseline_prefill_sec),
                "baseline_decode_wall_sec": float(baseline_decode_sec),
                "baseline_new_tokens": int(baseline_new_tokens),
                "baseline_decode_tok_s": float(baseline_decode_tok_s),
            }
        )
    del tokenizer, base_model
    import gc

    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _baseline_gpu_process_entry(
    physical_gpu_id: int,
    payload: list[tuple[int, dict[str, Any]]],
    base_model_path: str,
    temperature: float,
    max_new_tokens: int,
    no_chat_template: bool,
    enable_thinking: bool,
    eval_seed: int,
    out_jsonl_path: str,
    progress_q: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    try:
        indexed_local: list[tuple[int, UnifiedItem]] = [(i, _item_from_dict(d)) for i, d in payload]
        rows = _run_baseline_batch_for_cache(
            indexed_local,
            base_model_path=base_model_path,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            no_chat_template=bool(no_chat_template),
            enable_thinking=bool(enable_thinking),
            eval_seed=int(eval_seed),
        )
        outp = Path(out_jsonl_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for _ in rows:
            progress_q.put(1, block=True)
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        raise


def _run_baseline_batch_multiprocess(
    indexed: list[tuple[int, UnifiedItem]],
    *,
    base_model_path: str,
    temperature: float,
    max_new_tokens: int,
    no_chat_template: bool,
    enable_thinking: bool,
    eval_seed: int,
    gpus: list[int],
    cache_path: Path,
) -> list[dict[str, Any]]:
    n_total = len(indexed)
    n_wish = int(min(len(gpus), n_total))
    shards = _split_list_into_shards(indexed, n_wish)
    nproc = len(shards)
    g_used = [int(gpus[si]) for si in range(nproc)]
    part_paths = [cache_path.parent / f"{cache_path.stem}__baseline_part{si}.jsonl" for si in range(nproc)]
    ctx = mp.get_context("spawn")
    progress_q: Any = ctx.Queue()
    procs: list[mp.context.BaseProcess] = []
    for si, shard in enumerate(shards):
        pl = [(int(i), _item_to_dict(u)) for i, u in shard]
        p = ctx.Process(
            target=_baseline_gpu_process_entry,
            args=(
                g_used[si],
                pl,
                base_model_path,
                float(temperature),
                int(max_new_tokens),
                bool(no_chat_template),
                bool(enable_thinking),
                int(eval_seed),
                str(part_paths[si]),
                progress_q,
            ),
        )
        p.start()
        procs.append(p)
    pbar = tqdm(
        range(n_total),
        total=n_total,
        desc=f"Baseline HF generate T={temperature} ({nproc} GPUs)",
        unit="sample",
    )
    _wait_progress_with_worker_watch(progress_q, n_total, procs, pbar)
    pbar.close()
    for p in procs:
        p.join()
    for p in procs:
        if p.exitcode is not None and p.exitcode != 0:
            raise RuntimeError("A baseline worker process failed; see stderr for traceback.")
    tmp_merge = cache_path.with_name(cache_path.name + ".merge_tmp.jsonl")
    merged_rows = _merge_shards_to_jsonl([pp for pp in part_paths if pp.is_file()], tmp_merge)
    if tmp_merge.is_file():
        tmp_merge.unlink()
    for pp in part_paths:
        if pp.is_file():
            try:
                pp.unlink()
            except OSError:
                pass
    return merged_rows


def _write_baseline_cache_jsonl(
    path: Path,
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def ensure_baseline_cache(
    *,
    cache_path: Path,
    base_model_path: str,
    temperature: float,
    indexed: list[tuple[int, UnifiedItem]],
    args: argparse.Namespace,
    data_dir: Path,
    force_recompute: bool,
    gpus: list[int],
    use_multi: bool,
    has_cuda: bool,
) -> dict[int, dict[str, Any]]:
    n_total = len(indexed)
    expected_meta = _baseline_meta_payload(
        base_model_path,
        temperature,
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
        no_chat_template=bool(args.no_chat_template),
        enable_thinking=bool(args.enable_thinking),
        data_dir=data_dir,
        n_total=n_total,
    )
    if (
        not force_recompute
        and cache_path.is_file()
        and n_total > 0
    ):
        meta_disk, by_index = _read_baseline_cache_jsonl(cache_path)
        if _baseline_meta_matches_disk(meta_disk, expected_meta):
            _validate_baseline_cache_indices(by_index, n_total)
            return by_index
    if n_total == 0:
        _write_baseline_cache_jsonl(cache_path, expected_meta, [])
        return {}
    use_baseline_mp = bool(has_cuda and use_multi and len(gpus) >= 2 and n_total >= 2)
    if use_baseline_mp:
        rows = _run_baseline_batch_multiprocess(
            indexed,
            base_model_path=base_model_path,
            temperature=float(temperature),
            max_new_tokens=int(args.max_new_tokens),
            no_chat_template=bool(args.no_chat_template),
            enable_thinking=bool(args.enable_thinking),
            eval_seed=int(args.seed),
            gpus=gpus,
            cache_path=cache_path,
        )
    else:
        rows = _run_baseline_batch_for_cache(
            indexed,
            base_model_path=base_model_path,
            temperature=float(temperature),
            max_new_tokens=int(args.max_new_tokens),
            no_chat_template=bool(args.no_chat_template),
            enable_thinking=bool(args.enable_thinking),
            eval_seed=int(args.seed),
        )
    _write_baseline_cache_jsonl(cache_path, expected_meta, rows)
    return {int(r["index"]): r for r in rows}


def _unique_base_model_paths_in_order(
    ckpt_list: list[str],
    *,
    override: str | None = None,
) -> list[str]:
    from pipeline_inference import _base_model_path_override, _resolve_base_model_path

    o = _base_model_path_override(override)
    if o is not None:
        return [o]
    seen: set[str] = set()
    out: list[str] = []
    for ck in ckpt_list:
        b = _resolve_base_model_path(_read_spec_config_lite(ck), ck)
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _split_list_into_shards(xs: List[Any], n: int) -> List[List[Any]]:
    """Contiguous, nearly equal split."""
    m = len(xs)
    n = int(n)
    if n < 1:
        raise ValueError("n < 1")
    if m == 0:
        return [[] for _ in range(n)]
    n = min(n, m)
    if n == 1:
        return [list(xs)]
    base = m // n
    rem = m % n
    out: List[List[Any]] = []
    start = 0
    for i in range(n):
        l = base + (1 if i < rem else 0)
        out.append(xs[start : start + l])
        start += l
    return out


def _load_question_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def unify_from_jsonl(relative_key: str, data_dir: Path) -> list[UnifiedItem]:
    full = data_dir / relative_key
    if not full.is_file():
        raise FileNotFoundError(
            f"Expected dataset file at {full}. "
            "Create eval_data/<name>/question.jsonl (EAGLE format) or set --data_dir."
        )
    out: list[UnifiedItem] = []
    for obj in _load_question_rows(full):
        turns = obj.get("turns")
        if not turns:
            continue
        q0 = str(turns[0])
        if not q0.strip():
            continue
        out.append(
            UnifiedItem(
                dataset=relative_key.split("/")[0],
                question_id=int(obj.get("question_id", len(out))),
                question=q0,
            )
        )
    return out


def _looks_like_step_checkpoint_dir(name: str) -> bool:
    """
    Heuristic: HF ``Trainer`` / common save dirs: ``checkpoint-60000``, ``ckpt-5000``,
    ``step_2000``, etc., so the tag can add the *grandparent* (run / output name).
    """
    n = (name or "").strip()
    if not n or n in (".", ".."):
        return False
    return bool(
        re.match(
            r"^(checkpoint|ckpt|global[-_]?step|step|epoch)([-_]?)(\d+)([kK])?$",
            n,
            re.IGNORECASE,
        )
    )


def ckpt_to_filename_tag(ckpt_path: str) -> str:
    """
    Build a filesystem-safe label from the path.

    If the file lives under a step subfolder (e.g. ``.../run_id/checkpoint-60000/speculation_head.pt``),
    include *both* the run directory and the step directory so different steps do not collide
    in output filenames. Otherwise keep ``<immediate_parent>__<stem>`` (e.g. ``.../out/speculation_head_final.pt``).
    """
    p = Path(ckpt_path)
    parent = p.parent.name or "ckpt"
    stem = p.stem or "spec"
    grand = p.parent.parent
    gname: str
    if grand and grand != p.parent and len(p.parents) > 0:
        gname = (grand.name or "").strip() or ""
    else:
        gname = ""
    if _looks_like_step_checkpoint_dir(parent) and gname and gname not in (".", ".."):
        tag = f"{gname}__{parent}__{stem}"
    else:
        tag = f"{parent}__{stem}"
    for ch in '<>:"/\\|?*':
        tag = tag.replace(ch, "_")
    tag = re.sub(r"\s+", "_", tag)
    return tag[:240] if len(tag) > 240 else tag


def _non_empty_ckpt_path(s: str) -> str:
    t = (s or "").strip()
    if not t:
        raise argparse.ArgumentTypeError(
            "--spec_head_ckpt entries must be non-empty paths to speculation head checkpoints."
        )
    return t


def _normalize_spec_head_ckpt_list(x: Any) -> list[str]:
    """``str`` or ``list[str]`` / nested list -> flat ``list[str]`` for ``nargs='+'`` defaults and ``main``."""
    if isinstance(x, str):
        return [_non_empty_ckpt_path(x)]
    out: list[str] = []
    for item in x:
        if isinstance(item, str):
            out.append(_non_empty_ckpt_path(item))
        elif isinstance(item, (list, tuple)):
            out.extend(_normalize_spec_head_ckpt_list(item))
        else:
            out.append(_non_empty_ckpt_path(str(item)))
    return out





def _encode_prompt(
    tokenizer: Any,
    question: str,
    use_chat: bool,
    device: Any,
    *,
    enable_thinking: bool = False,
) -> tuple[Any, Any]:
    import torch

    if use_chat:
        batch = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=bool(enable_thinking),
        )
    else:
        batch = tokenizer(question, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attn = batch.get("attention_mask")
    if attn is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = attn.to(device)
    return input_ids, attention_mask


def _baseline_standard_decode_timing(
    base_model: Any,
    tokenizer: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
) -> tuple[Any, float, float, float]:
    """Standard HF ``generate``; decode-only wall time is ``gen_sec - prefill_sec`` (prefill from one matching forward)."""
    import torch

    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        base_model(input_ids, attention_mask=attention_mask, use_cache=True)
    _sync()
    prefill_sec = time.perf_counter() - t0

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "attention_mask": attention_mask.to(input_ids.device),
    }
    if greedy:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature

    _sync()
    t1 = time.perf_counter()
    with torch.inference_mode():
        out = base_model.generate(input_ids, **kwargs)
    _sync()
    gen_sec = time.perf_counter() - t1
    decode_sec = max(gen_sec - prefill_sec, 0.0)
    return out[0], float(prefill_sec), float(gen_sec), float(decode_sec)


def _args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in vars(args).items():
        if v is not None and not callable(v) and not k.startswith("_"):
            d[k] = v
    return d


def _build_inference_stack(args: Any) -> Tuple[Any, Any, Any, int]:
    """Load tokenizer, base model, pipeline once. ``args.temperature`` must be a float (worker / per-step scalar)."""
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from pipeline_inference import (
        _infer_pipeline_kind,
        _read_spec_config,
        _resolve_base_model_path,
        build_pipeline_from_spec_ckpt,
    )

    spec_cfg = _read_spec_config(args.spec_head_ckpt)
    base_model_path = _resolve_base_model_path(
        spec_cfg,
        args.spec_head_ckpt,
        override=getattr(args, "base_model_path", ""),
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    _infer_pipeline_kind(spec_cfg)
    map_loc = "cuda"
    pipeline = build_pipeline_from_spec_ckpt(
        base_model,
        args.spec_head_ckpt,
        spec_cfg,
        map_location=map_loc,
    )
    num_stages = int(pipeline.num_stages)
    device = next(pipeline.base_model.parameters()).device
    return tokenizer, pipeline, device, num_stages


def _run_indexed_inference_loop(
    indexed: List[Tuple[int, UnifiedItem]],
    args: Any,
    tokenizer: Any,
    pipeline: Any,
    device: Any,
    num_stages: int,
    *,
    progress_q: Any = None,
    use_tqdm: bool = True,
    tqdm_desc: str = "Pipeline inference",
    baseline_by_index: Optional[dict[int, dict[str, Any]]] = None,
) -> List[dict[str, Any]]:
    from pipeline_inference import run_pipeline_generate, theoretical_speedup_vs_standard

    greedy = bool(float(args.temperature) == 0.0)
    use_chat = not bool(args.no_chat_template)
    enable_thinking = bool(getattr(args, "enable_thinking", False))
    it_seq: Any = indexed
    if use_tqdm and progress_q is None:
        it_seq = tqdm(
            indexed,
            total=len(indexed),
            desc=tqdm_desc,
            unit="sample",
        )
    out_rows: list[dict[str, Any]] = []
    baseline_on = bool(getattr(args, "baseline", True))
    if baseline_on and baseline_by_index is None:
        raise ValueError(
            "baseline=True requires baseline_by_index (build or load baseline cache before pipeline inference)."
        )
    for idx, it in it_seq:
        _set_torch_rng_for_eval_sample(int(getattr(args, "seed", 0)), int(idx))
        input_ids, attention_mask = _encode_prompt(
            tokenizer, it.question, use_chat, device, enable_thinking=enable_thinking
        )
        baseline_decode_tok_s = None
        baseline_prefill_sec = None
        baseline_decode_sec = None
        baseline_new_tokens = None
        if baseline_on:
            b = baseline_by_index[int(idx)]
            baseline_prefill_sec = float(b["baseline_prefill_wall_sec"])
            baseline_decode_sec = float(b["baseline_decode_wall_sec"])
            baseline_new_tokens = int(b["baseline_new_tokens"])
            baseline_decode_tok_s = float(b["baseline_decode_tok_s"])

        (
            full_ids,
            wall_s,
            token_acceptance,
            decode_loop_steps,
            timing,
        ) = run_pipeline_generate(
            pipeline,
            input_ids,
            device,
            max_new_tokens=args.max_new_tokens,
            greedy=greedy,
            temperature=float(args.temperature),
            verify=True,
            use_deepest=bool(args.use_deepest),
            draft_top_k=int(args.draft_top_k),
        )
        prompt_len = int(input_ids.shape[1])
        gen_only_ids = full_ids[prompt_len:]
        gen_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
        n_flags = len(token_acceptance)
        n_acc = int(sum(1 for x in token_acceptance if x))
        new_tokens = int(gen_only_ids.shape[0])
        steps = int(decode_loop_steps)
        acc_rate = (float(new_tokens) / float(steps)) if steps else 0.0
        equiv_accept_len = float(num_stages) * acc_rate
        t_std, t_pipe, v_std, v_pipe = theoretical_speedup_vs_standard(
            num_stages=num_stages,
            num_new_tokens=new_tokens,
            pipeline_decode_steps=decode_loop_steps,
        )
        th_pct = (v_pipe / v_std - 1.0) * 100.0 if v_std > 0 else 0.0

        pip_prefill = float(timing.get("prefill_wall_sec", 0.0))
        pip_decode = float(timing.get("decode_wall_sec", 0.0))
        if pip_decode <= 0.0:
            pip_decode = float(wall_s)
        pip_ideal_decode = float(timing.get("ideal_decode_wall_sec", 0.0))
        if pip_ideal_decode <= 0.0:
            pip_ideal_decode = pip_decode
        pip_stage_gpu = float(timing.get("pipeline_stage_stream_gpu_sec", 0.0))
        pip_saved = float(timing.get("pipeline_ideal_parallel_saved_sec", 0.0))

        pipe_decode_tok_s = float(new_tokens) / pip_decode if pip_decode > 0 else 0.0
        pipe_ideal_tok_s = float(new_tokens) / pip_ideal_decode if pip_ideal_decode > 0 else 0.0

        speedup_vs_baseline = None
        ideal_speedup_vs_baseline = None
        if baseline_on and baseline_decode_tok_s is not None and baseline_decode_tok_s > 0:
            speedup_vs_baseline = pipe_decode_tok_s / baseline_decode_tok_s
            ideal_speedup_vs_baseline = pipe_ideal_tok_s / baseline_decode_tok_s

        row: dict[str, Any] = {
            "index": int(idx),
            "dataset": it.dataset,
            "question_id": it.question_id,
            "question": it.question,
            "generated": gen_text,
            "new_tokens": new_tokens,
            "decode_loop_steps": steps,
            "acceptance_rate": acc_rate,
            "equivalent_accept_length": equiv_accept_len,
            "n_accepted": n_acc,
            "n_acceptance_flags": n_flags,
            "theoretical_std_time_sec": t_std,
            "theoretical_pipe_time_sec": t_pipe,
            "theoretical_std_tok_s": v_std,
            "theoretical_pipe_tok_s": v_pipe,
            "theoretical_throughput_gain_pct": th_pct,
            "wall_time_sec": float(wall_s),
            "num_stages": num_stages,
            "use_deepest": bool(args.use_deepest),
            "draft_top_k": int(args.draft_top_k),
            "pipeline_prefill_wall_sec": pip_prefill,
            "pipeline_decode_wall_sec": pip_decode,
            "pipeline_decode_tok_s": pipe_decode_tok_s,
            "pipeline_stage_stream_gpu_sec": pip_stage_gpu,
            "pipeline_ideal_parallel_saved_sec": pip_saved,
            "ideal_decode_wall_sec": pip_ideal_decode,
            "pipeline_ideal_decode_tok_s": pipe_ideal_tok_s,
        }
        if baseline_on:
            row["baseline_prefill_wall_sec"] = baseline_prefill_sec
            row["baseline_decode_wall_sec"] = baseline_decode_sec
            row["baseline_new_tokens"] = baseline_new_tokens
            row["baseline_decode_tok_s"] = baseline_decode_tok_s
            row["speedup_vs_baseline"] = speedup_vs_baseline
            row["ideal_speedup_vs_baseline"] = ideal_speedup_vs_baseline

        out_rows.append(row)
        if progress_q is not None:
            progress_q.put(1, block=True)
    return out_rows


def _run_indexed_inference(
    indexed: List[Tuple[int, UnifiedItem]],
    args: Any,
    *,
    progress_q: Any = None,
    use_tqdm: bool = True,
    baseline_by_index: Optional[dict[int, dict[str, Any]]] = None,
) -> List[dict[str, Any]]:
    """Run pipeline on a list of (global_index, item) pairs. Set ``CUDA_VISIBLE_DEVICES`` before calling if needed."""
    if not indexed:
        return []
    tokenizer, pipeline, device, num_stages = _build_inference_stack(args)
    return _run_indexed_inference_loop(
        indexed,
        args,
        tokenizer,
        pipeline,
        device,
        num_stages,
        progress_q=progress_q,
        use_tqdm=use_tqdm,
        baseline_by_index=baseline_by_index,
    )


def _gpu_process_entry(
    physical_gpu_id: int,
    payload: list[tuple[int, dict[str, Any]]],
    args_dict: dict[str, Any],
    out_jsonl_path: str,
    progress_q: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    try:
        indexed: list[tuple[int, UnifiedItem]] = [(i, _item_from_dict(d)) for i, d in payload]
        args = SimpleNamespace(**args_dict)
        baseline_by_index: Optional[dict[int, dict[str, Any]]] = None
        if bool(getattr(args, "baseline", True)):
            bcp = str(getattr(args, "baseline_cache_path", "") or "").strip()
            if not bcp:
                raise ValueError("baseline=True requires baseline_cache_path in worker args_dict")
            _, baseline_by_index = _read_baseline_cache_jsonl(Path(bcp))
            wanted = {int(i) for i, _ in indexed}
            missing = wanted - baseline_by_index.keys()
            if missing:
                raise ValueError(
                    f"Baseline cache {bcp!r} missing indices (e.g. {sorted(missing)[:8]}); "
                    "re-run main to rebuild cache or fix shard/cache mismatch."
                )
        rows = _run_indexed_inference(
            indexed, args, progress_q=progress_q, use_tqdm=False, baseline_by_index=baseline_by_index
        )
        p = Path(out_jsonl_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except BaseException:  # noqa: BLE001 — re-raise so Process.exitcode != 0
        import traceback

        traceback.print_exc()
        raise


def _merge_shards_to_jsonl(shard_paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for sp in shard_paths:
        if not sp.is_file():
            continue
        with sp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_rows.append(json.loads(line))
    all_rows.sort(key=lambda r: int(r.get("index", 0)))
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return all_rows


def _terminate_workers_and_join(procs: list[mp.context.BaseProcess], join_timeout_sec: float = 30.0) -> None:
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=join_timeout_sec)


def _raise_if_any_worker_failed(procs: list[mp.context.BaseProcess]) -> None:
    for p in procs:
        if not p.is_alive() and p.exitcode not in (0, None):
            _terminate_workers_and_join(procs)
            raise RuntimeError(
                f"A worker process failed (exitcode={p.exitcode}); see stderr above for traceback."
            )


def _wait_progress_with_worker_watch(
    progress_q: Any,
    n_total: int,
    procs: list[mp.context.BaseProcess],
    pbar: Any,
    poll_timeout_sec: float = 2.0,
) -> None:
    """Block until ``n_total`` progress signals or any worker exits with failure."""
    done = 0
    while done < n_total:
        _raise_if_any_worker_failed(procs)
        if not any(p.is_alive() for p in procs) and done < n_total:
            _terminate_workers_and_join(procs)
            codes = [p.exitcode for p in procs]
            raise RuntimeError(
                "All worker processes exited before reporting completion "
                f"({done}/{n_total} progress signals); exitcodes={codes!r}."
            )
        try:
            progress_q.get(timeout=poll_timeout_sec)
        except queue.Empty:
            continue
        done += 1
        pbar.update(1)


def _aggregate_for(per_sample: list[dict[str, Any]], ds: Optional[str]) -> dict[str, float]:
    """
    Pool **acceptance rate** as ``sum(new_tokens) / sum(decode_loop_steps)`` (generated
    tokens per decoding step). **Equivalent accept length** uses ``num_stages`` from rows:
    ``num_stages * aggregate_acceptance_rate``.

    Theoretical throughput gain is token-weighted:
    ``sum(gain_pct_i * new_tokens_i) / sum(new_tokens_i)``.
    """
    sub = [r for r in per_sample if ds is None or r.get("dataset") == ds]
    if not sub:
        return {
            "aggregate_acceptance_rate": 0.0,
            "aggregate_equivalent_accept_length": 0.0,
            "aggregate_theoretical_throughput_gain_pct": 0.0,
            "total_new_tokens": 0.0,
            "total_decode_loop_steps": 0.0,
            "total_n_accepted": 0.0,
            "total_n_acceptance_flags": 0.0,
            "num_samples": 0.0,
            "aggregate_pipeline_decode_tok_s": 0.0,
            "aggregate_pipeline_ideal_decode_tok_s": 0.0,
            "aggregate_baseline_decode_tok_s": 0.0,
            "aggregate_speedup_vs_baseline": 0.0,
            "aggregate_ideal_speedup_vs_baseline": 0.0,
        }
    tot_flags = sum(int(r.get("n_acceptance_flags", 0) or 0) for r in sub)
    tot_acc = sum(int(r.get("n_accepted", 0) or 0) for r in sub)
    tot_steps = sum(int(r.get("decode_loop_steps", 0) or 0) for r in sub)
    tot_new = sum(int(r.get("new_tokens", 0) or 0) for r in sub)
    agg_acc = (float(tot_new) / float(tot_steps)) if tot_steps else 0.0
    n_stages = int(sub[0].get("num_stages", 0) or 0)
    agg_equiv = float(n_stages) * agg_acc
    w_gain = sum(
        float(r.get("theoretical_throughput_gain_pct", 0.0) or 0.0)
        * int(r.get("new_tokens", 0) or 0)
        for r in sub
    )
    agg_gain = (w_gain / float(tot_new)) if tot_new else 0.0

    tot_pipe_decode = sum(float(r.get("pipeline_decode_wall_sec", 0) or 0) for r in sub)
    tot_ideal_decode = sum(float(r.get("ideal_decode_wall_sec", 0) or 0) for r in sub)
    agg_pipe_tok_s = float(tot_new) / tot_pipe_decode if tot_pipe_decode > 0 else 0.0
    agg_ideal_tok_s = float(tot_new) / tot_ideal_decode if tot_ideal_decode > 0 else 0.0

    has_baseline = bool(sub) and sub[0].get("baseline_decode_wall_sec") is not None
    if has_baseline:
        tot_base_nt = sum(int(r.get("baseline_new_tokens", 0) or 0) for r in sub)
        tot_base_dec = sum(float(r.get("baseline_decode_wall_sec", 0) or 0) for r in sub)
        agg_base_tok_s = float(tot_base_nt) / tot_base_dec if tot_base_dec > 0 else 0.0
        agg_speedup = agg_pipe_tok_s / agg_base_tok_s if agg_base_tok_s > 0 else 0.0
        agg_ideal_speedup = agg_ideal_tok_s / agg_base_tok_s if agg_base_tok_s > 0 else 0.0
    else:
        agg_base_tok_s = 0.0
        agg_speedup = 0.0
        agg_ideal_speedup = 0.0

    out = {
        "aggregate_acceptance_rate": agg_acc,
        "aggregate_equivalent_accept_length": agg_equiv,
        "aggregate_theoretical_throughput_gain_pct": agg_gain,
        "total_new_tokens": float(tot_new),
        "total_decode_loop_steps": float(tot_steps),
        "total_n_accepted": float(tot_acc),
        "total_n_acceptance_flags": float(tot_flags),
        "num_samples": float(len(sub)),
        "aggregate_pipeline_decode_tok_s": agg_pipe_tok_s,
        "aggregate_pipeline_ideal_decode_tok_s": agg_ideal_tok_s,
        "aggregate_baseline_decode_tok_s": agg_base_tok_s,
        "aggregate_speedup_vs_baseline": agg_speedup,
        "aggregate_ideal_speedup_vs_baseline": agg_ideal_speedup,
    }
    return out


def _print_aggregates(per_sample: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    o = _aggregate_for(per_sample, None)
    print()
    print(
        f"Overall — acceptance rate (new_tokens / decode_steps): "
        f"{o['aggregate_acceptance_rate']:.4f}, "
        f"equiv. accept length (n×rate, n=num_stages): {o['aggregate_equivalent_accept_length']:.4f} "
        f"[{int(o['total_new_tokens'])} tokens / {int(o['total_decode_loop_steps'])} steps; "
        f"draft flags accepted {int(o['total_n_accepted'])}/{int(o['total_n_acceptance_flags'])}]"
    )
    print(
        f"Overall — token-weighted mean theoretical throughput gain (% vs toy standard decode): "
        f"{o['aggregate_theoretical_throughput_gain_pct']:.2f}%"
    )
    print(
        f"Overall — pooled decode throughput (prefill excluded): "
        f"pipeline wall {o['aggregate_pipeline_decode_tok_s']:.2f} tok/s, "
        f"pipeline ideal (parallel stages) {o['aggregate_pipeline_ideal_decode_tok_s']:.2f} tok/s"
        + (
            f"; baseline standard HF {o['aggregate_baseline_decode_tok_s']:.2f} tok/s "
            f"(speedup ×{o['aggregate_speedup_vs_baseline']:.3f}, ideal speedup ×{o['aggregate_ideal_speedup_vs_baseline']:.3f})"
            if (o.get("aggregate_baseline_decode_tok_s") or 0) > 0
            else ""
        )
    )
    for ds, _ in DATASET_CONFIG:
        m = _aggregate_for(per_sample, ds)
        print(
            f"  [{ds}] acceptance={m['aggregate_acceptance_rate']:.4f}, "
            f"equiv. accept len={m['aggregate_equivalent_accept_length']:.4f}, "
            f"token-weighted theoretical gain={m['aggregate_theoretical_throughput_gain_pct']:.2f}% "
            f"(tokens={int(m['total_new_tokens'])}, steps={int(m['total_decode_loop_steps'])})"
        )
    w = float(summary.get("total_wall_sec", 0) or 0.0)
    nproc = int(summary.get("num_worker_processes", 1) or 1)
    gpus = summary.get("gpus_used", [])
    print(
        f"\nWall time (entire run): {w:.2f}s, worker processes: {nproc}, "
        f"GPUs: {gpus!r}."
    )


def main() -> None:
    from pipeline_inference import _resolve_base_model_path

    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    summary_dir = out_dir / "summary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    gpus = _parse_gpus(str(getattr(args, "gpus", "0") or "0"))
    if not gpus:
        gpus = [0]

    ckpt_list: list[str] = _normalize_spec_head_ckpt_list(args.spec_head_ckpt)
    if not ckpt_list:
        raise ValueError("--spec_head_ckpt is required (provide one or more checkpoint paths).")

    all_items: list[UnifiedItem] = []
    for _ds, rel in DATASET_CONFIG:
        unified = unify_from_jsonl(rel, data_dir)
        for it in unified:
            all_items.append(it)

    indexed: list[tuple[int, UnifiedItem]] = list(enumerate(all_items))
    n_total = len(indexed)

    # Set CUDA_DEVICE visibility before the first `import torch` in single-GPU (or CPU) runs
    if n_total < 2 or len(gpus) < 2:
        if n_total and gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpus[0]))
        import torch

        has_cuda = bool(torch.cuda.is_available())
        use_multi = False
    else:
        import torch

        has_cuda = bool(torch.cuda.is_available())
        use_multi = has_cuda
        if not has_cuda and len(gpus) > 1:
            print(
                "Warning: --gpus listed multiple devices but CUDA is not available; "
                "using CPU, single process."
            )

    temperatures = _normalize_temperature_list(list(args.temperature))
    draft_top_ks = _normalize_draft_top_k_list(list(args.draft_top_k))

    baseline_root = Path(args.baseline_cache_dir) if args.baseline_cache_dir else (out_dir / "baseline")
    baseline_path_by_key: dict[tuple[str, float], Path] = {}
    baseline_map_by_key: dict[tuple[str, float], dict[int, dict[str, Any]]] = {}
    if bool(args.baseline):
        baseline_root.mkdir(parents=True, exist_ok=True)
        unique_bases = _unique_base_model_paths_in_order(
            ckpt_list, override=args.base_model_path
        )
        prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        modified_cuda_for_baseline_single = False
        try:
            if (not use_multi) and has_cuda and gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpus[0]))
                modified_cuda_for_baseline_single = True
            force_br = bool(getattr(args, "force_baseline_recompute", False))
            for base_path in unique_bases:
                for temp in temperatures:
                    fname = _baseline_cache_filename(
                        base_path,
                        float(temp),
                        seed=int(args.seed),
                        max_new_tokens=int(args.max_new_tokens),
                        no_chat_template=bool(args.no_chat_template),
                        enable_thinking=bool(args.enable_thinking),
                        data_dir=data_dir,
                        n_total=n_total,
                    )
                    cache_p = baseline_root / fname
                    key = (base_path, float(temp))
                    baseline_map_by_key[key] = ensure_baseline_cache(
                        cache_path=cache_p,
                        base_model_path=base_path,
                        temperature=float(temp),
                        indexed=indexed,
                        args=args,
                        data_dir=data_dir,
                        force_recompute=force_br,
                        gpus=gpus,
                        use_multi=use_multi,
                        has_cuda=has_cuda,
                    )
                    baseline_path_by_key[key] = cache_p
            print(f"[baseline] cache directory: {baseline_root.resolve()}")
        finally:
            if modified_cuda_for_baseline_single:
                if prev_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda

    for ckpt_idx, ckpt_path in enumerate(ckpt_list):
        if len(ckpt_list) > 1:
            print(f"\n=== Checkpoint {ckpt_idx + 1}/{len(ckpt_list)}: {ckpt_path} ===\n")

        ckpt_tag = ckpt_to_filename_tag(ckpt_path)
        base_name = f"pipeline_eval__{ckpt_tag}__nt{n_total}"
        summary_path = summary_dir / f"{base_name}__summary.json"

        results_meta: list[dict[str, Any]] = []
        elapsed_total_sum = 0.0
        n_workers_done = 1
        gpus_used: list[int] = []
        num_stages = int(_read_spec_config_lite(ckpt_path).get("num_stages", 0))

        if n_total == 0:
            pass
        elif use_multi:
            n_wish = int(min(len(gpus), n_total))
            shards = _split_list_into_shards(indexed, n_wish)
            nproc = len(shards)
            g_used = [int(gpus[si]) for si in range(nproc)]
            ctx = mp.get_context("spawn")

            for temp in temperatures:
                for dtk in draft_top_ks:
                    stem = _per_sample_file_stem(
                        base_name,
                        float(temp),
                        int(dtk),
                        temperatures=temperatures,
                        draft_top_ks=draft_top_ks,
                    )
                    per_sample_path = raw_dir / f"{stem}.jsonl"
                    part_paths = [raw_dir / f"{stem}__part{si}.jsonl" for si in range(nproc)]

                    base_vars = {
                        **vars(args),
                        "spec_head_ckpt": ckpt_path,
                        "temperature": float(temp),
                        "draft_top_k": int(dtk),
                    }
                    base_model_path = _resolve_base_model_path(
                        _read_spec_config_lite(ckpt_path),
                        ckpt_path,
                        override=args.base_model_path,
                    )
                    if bool(args.baseline):
                        base_vars["baseline_cache_path"] = str(
                            baseline_path_by_key[(base_model_path, float(temp))]
                        )
                    else:
                        base_vars["baseline_cache_path"] = ""
                    args_run_t = SimpleNamespace(**base_vars)
                    args_dict = _args_to_dict(args_run_t)
                    t0 = time.perf_counter()
                    progress_q: Any = ctx.Queue()
                    procs: list[mp.context.BaseProcess] = []
                    for si, shard in enumerate(shards):
                        g = g_used[si]
                        pl = [(int(i), _item_to_dict(u)) for i, u in shard]
                        p = ctx.Process(
                            target=_gpu_process_entry,
                            args=(g, pl, args_dict, str(part_paths[si]), progress_q),
                        )
                        p.start()
                        procs.append(p)
                    pbar = tqdm(
                        range(n_total),
                        desc=f"Pipeline inference T={temp} dtk={dtk} ({nproc} GPUs)",
                        unit="sample",
                    )
                    _wait_progress_with_worker_watch(progress_q, n_total, procs, pbar)
                    pbar.close()
                    for p in procs:
                        p.join()
                    for p in procs:
                        if p.exitcode is not None and p.exitcode != 0:
                            raise RuntimeError("A worker process failed; see stderr for traceback.")
                    per_sample = _merge_shards_to_jsonl(
                        [pp for pp in part_paths if pp.is_file()],
                        per_sample_path,
                    )
                    for pp in part_paths:
                        if pp.is_file() and pp.resolve() != per_sample_path.resolve():
                            try:
                                pp.unlink()
                            except OSError:
                                pass
                    elapsed_part = time.perf_counter() - t0
                    elapsed_total_sum += elapsed_part
                    if per_sample:
                        num_stages = int(per_sample[0].get("num_stages", 0))
                    results_meta.append(
                        {
                            "temperature": float(temp),
                            "greedy": bool(float(temp) == 0.0),
                            "use_deepest": bool(args.use_deepest),
                            "draft_top_k": int(dtk),
                            "enable_thinking": bool(args.enable_thinking),
                            "total_wall_sec": float(elapsed_part),
                            "overall": _aggregate_for(per_sample, None),
                            "per_dataset": {ds: _aggregate_for(per_sample, ds) for ds, _ in DATASET_CONFIG},
                            "per_sample_path": str(per_sample_path),
                        }
                    )
                    mini = {
                        "total_wall_sec": float(elapsed_part),
                        "num_worker_processes": len(procs),
                        "gpus_used": g_used,
                        "draft_top_k": int(dtk),
                    }
                    print(f"\n--- temperature={temp} draft_top_k={dtk} ---")
                    _print_aggregates(per_sample, mini)

            n_workers_done = nproc
            gpus_used = g_used
        else:
            first_vars = {
                **vars(args),
                "spec_head_ckpt": ckpt_path,
                "temperature": float(temperatures[0]),
                "draft_top_k": int(draft_top_ks[0]),
            }
            args_first = SimpleNamespace(**first_vars)
            tokenizer, pipeline, device, num_stages_stack = _build_inference_stack(args_first)
            num_stages = int(num_stages_stack)
            base_model_path_ck = _resolve_base_model_path(
                _read_spec_config_lite(ckpt_path),
                ckpt_path,
                override=args.base_model_path,
            )

            for temp in temperatures:
                for dtk in draft_top_ks:
                    base_vars = {
                        **vars(args),
                        "spec_head_ckpt": ckpt_path,
                        "temperature": float(temp),
                        "draft_top_k": int(dtk),
                    }
                    args_run_t = SimpleNamespace(**base_vars)
                    b_line = (
                        baseline_map_by_key[(base_model_path_ck, float(temp))]
                        if bool(args.baseline)
                        else None
                    )
                    t0 = time.perf_counter()
                    rows = _run_indexed_inference_loop(
                        indexed,
                        args_run_t,
                        tokenizer,
                        pipeline,
                        device,
                        num_stages,
                        progress_q=None,
                        use_tqdm=True,
                        tqdm_desc=f"Pipeline inference (T={temp} dtk={dtk})",
                        baseline_by_index=b_line,
                    )
                    elapsed_part = time.perf_counter() - t0
                    elapsed_total_sum += elapsed_part

                    stem = _per_sample_file_stem(
                        base_name,
                        float(temp),
                        int(dtk),
                        temperatures=temperatures,
                        draft_top_ks=draft_top_ks,
                    )
                    per_sample_path = raw_dir / f"{stem}.jsonl"

                    with per_sample_path.open("w", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    results_meta.append(
                        {
                            "temperature": float(temp),
                            "greedy": bool(float(temp) == 0.0),
                            "use_deepest": bool(args.use_deepest),
                            "draft_top_k": int(dtk),
                            "enable_thinking": bool(args.enable_thinking),
                            "total_wall_sec": float(elapsed_part),
                            "overall": _aggregate_for(rows, None),
                            "per_dataset": {ds: _aggregate_for(rows, ds) for ds, _ in DATASET_CONFIG},
                            "per_sample_path": str(per_sample_path),
                        }
                    )
                    mini = {
                        "total_wall_sec": float(elapsed_part),
                        "num_worker_processes": 1,
                        "gpus_used": [int(gpus[0])] if (has_cuda and n_total and gpus) else [],
                        "draft_top_k": int(dtk),
                    }
                    print(f"\n--- temperature={temp} draft_top_k={dtk} ---")
                    _print_aggregates(rows, mini)

            n_workers_done = 1
            gpus_used = [int(gpus[0])] if (has_cuda and n_total and gpus) else []

        summary: dict[str, Any] = {
            "checkpoint_path": ckpt_path,
            "spec_head_ckpt_list_in_session": ckpt_list,
            "checkpoint_filename_tag": ckpt_tag,
            "model": _resolve_base_model_path(
                _read_spec_config_lite(ckpt_path),
                ckpt_path,
                override=args.base_model_path,
            ),
            "data_dir": str(data_dir),
            "num_prompts": n_total,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "verify": True,
            "use_deepest": bool(args.use_deepest),
            "enable_thinking": bool(args.enable_thinking),
            "baseline": bool(args.baseline),
            "baseline_cache_dir": str(baseline_root.resolve()) if bool(args.baseline) else None,
            "force_baseline_recompute": bool(getattr(args, "force_baseline_recompute", False)),
            "pipeline_implementation": _pipeline_implementation_tag(ckpt_path),
            "num_stages": num_stages,
            "gpus_requested": gpus,
            "gpus_used": gpus_used,
            "num_worker_processes": n_workers_done,
            "multiprocess_parallel": use_multi and n_total >= 2,
            "temperatures_evaluated": temperatures,
            "draft_top_ks_evaluated": draft_top_ks,
            "results": results_meta,
            "total_wall_sec": float(elapsed_total_sum),
        }
        if len(draft_top_ks) == 1:
            summary["draft_top_k"] = int(draft_top_ks[0])
        if len(results_meta) == 1:
            r0 = results_meta[0]
            summary["temperature"] = r0["temperature"]
            summary["greedy"] = r0["greedy"]
            summary["overall"] = r0["overall"]
            summary["per_dataset"] = r0["per_dataset"]
            summary["per_sample_path"] = r0["per_sample_path"]
            summary["total_wall_sec"] = float(r0["total_wall_sec"])
            summary["draft_top_k"] = int(r0["draft_top_k"])

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        for r in results_meta:
            print(f"Wrote: {r['per_sample_path']}")
        print(f"Wrote: {summary_path}")


def _read_spec_config_lite(ckpt_path: str) -> dict[str, Any]:
    from pipeline_inference import _read_spec_config

    return _read_spec_config(ckpt_path)


def _pipeline_implementation_tag(ckpt_path: str) -> str:
    from pipeline_inference import _infer_pipeline_kind

    return f"v{_infer_pipeline_kind(_read_spec_config_lite(ckpt_path))}"


if __name__ == "__main__":
    main()
