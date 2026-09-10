"""Official Instruction-Lens target-token resolution.

This module intentionally reproduces the released InsLen detector's target
lookup, including its first-subtoken and first-occurrence behaviour.  It does
not compute or use exact response offsets to repair a mismatch.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import nltk


INSLEN_TARGET_PROTOCOL = "inslen_official_first_token_first_occurrence"
INSLEN_SAMPLE_UNIT = "inslen_unique_detected_word"
INSLEN_UPSTREAM_REPOSITORY = (
    "https://github.com/Fraserlairh/Instruction-Lens-Score"
)
INSLEN_UPSTREAM_COMMIT = "d36f6f76023e7625ef86df7b60acf7a99ae3707d"
INSLEN_INTERNVL_BOS_COMPATIBILITY_NOTE = (
    "InternVL find_word candidate tokenization uses "
    "add_special_tokens=False so input_ids[0,0] is the first lexical "
    "subtoken rather than the tokenizer-added BOS token."
)
INSLEN_QWEN_COMPATIBILITY_NOTE = (
    "Qwen2.5/Qwen3 use the released QwenVL3 find_word branch; "
    "the official repository does not publish a Qwen2.5 wrapper."
)

_PREFIXES = ("", " ")
_SUFFIXES = ("", "s", "es")
_COMPLEMENT_WORDS = ("person", "people")

_MODEL_BRANCHES = {
    "minigpt4_7b": "llava_default_surface",
    "shikra_7b": "llava_default_surface",
    "llava_1_5_7b": "llava_default_surface",
    "internvl_2_5_8b": "internvl_find_word",
    "qwen2_5_vl_7b": "qwenvl3_find_word",
    "qwen3_vl_8b": "qwenvl3_find_word",
    "llava_onevision_1_5_8b": "llava_onevision_find_word",
    "llava_onevision_1_5_8b_instruct": "llava_onevision_find_word",
}


def inslen_model_branch(model_key: str) -> str:
    """Return the official resolver family selected for a local model key."""

    key = str(model_key).strip()
    if key in _MODEL_BRANCHES:
        return _MODEL_BRANCHES[key]
    if key.startswith("llava_next"):
        raise ValueError(
            "InsLen target protocol has no released LLaVA-NeXT resolver; "
            f"refusing to invent one for model={key!r}."
        )
    raise ValueError(
        f"Unsupported model for InsLen target protocol: {key!r}; "
        f"supported={sorted(_MODEL_BRANCHES)}"
    )


def inslen_resolver_compatibility_note(model_key: str) -> str | None:
    """Describe the model-specific compatibility choice, if one is needed."""

    branch = inslen_model_branch(model_key)
    if model_key in {"minigpt4_7b", "shikra_7b"}:
        return ("Vicuna/Llama tokenizer adaptation: apply the released default "
                "surface first-subtoken/first-occurrence rule; not an upstream model wrapper.")
    if branch == "internvl_find_word":
        return INSLEN_INTERNVL_BOS_COMPATIBILITY_NOTE
    if branch == "qwenvl3_find_word":
        return INSLEN_QWEN_COMPATIBILITY_NOTE
    return None


def official_caption_tokens(caption: str) -> list[str]:
    """Match the lower-cased NLTK word tokens used by the official wrappers."""

    return [str(value) for value in nltk.word_tokenize(str(caption).lower())]


def resolve_inslen_candidate(
    *,
    model_key: str,
    tokenizer: Any,
    response_token_ids: Sequence[int],
    raw_caption_tokens: Sequence[str],
    mention: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the candidate vocabulary ID before duplicate filtering.

    The default LLaVA branch returns its re-tokenized surface ID even when the
    ID is absent from the answer; the caller performs the official subsequent
    membership check.  Special-model ``find_word`` branches return no token
    unless a candidate ID is already present in the answer.
    """

    branch = inslen_model_branch(model_key)
    response_ids = [int(value) for value in response_token_ids]
    normalized_word = str(
        mention.get("normalized_word") or mention.get("word") or ""
    )
    word_index = int(mention.get("word_idx", -1))

    if branch == "llava_default_surface":
        if word_index < 0 or word_index >= len(raw_caption_tokens):
            return _not_found(
                branch=branch,
                detected_word=normalized_word,
                reason="word_index_out_of_range",
            )
        detected_word = str(raw_caption_tokens[word_index])
        token_id = _default_llava_first_id(tokenizer, detected_word)
        if token_id is None:
            return _not_found(
                branch=branch,
                detected_word=detected_word,
                reason="empty_tokenization",
            )
        return {
            "status": "candidate",
            "resolver_branch": branch,
            "detected_word": detected_word,
            "candidate_surface": detected_word,
            "target_token_id": int(token_id),
            "candidate_attempts": [detected_word],
        }

    search_words = (
        _COMPLEMENT_WORDS
        if normalized_word in _COMPLEMENT_WORDS
        else (normalized_word,)
    )
    attempts: list[str] = []
    for search_word in search_words:
        for prefix in _PREFIXES:
            for suffix in _SUFFIXES:
                surface = f"{prefix}{search_word}{suffix}"
                attempts.append(surface)
                token_id = _special_first_id(
                    tokenizer=tokenizer,
                    surface=surface,
                    branch=branch,
                )
                if token_id is not None and int(token_id) in response_ids:
                    return {
                        "status": "candidate",
                        "resolver_branch": branch,
                        "detected_word": normalized_word,
                        "candidate_surface": surface,
                        "target_token_id": int(token_id),
                        "candidate_attempts": attempts,
                    }
    return _not_found(
        branch=branch,
        detected_word=normalized_word,
        reason="find_word_no_candidate_in_output",
        attempts=attempts,
    )


def first_token_occurrence(
    response_token_ids: Sequence[int],
    token_id: int,
) -> int | None:
    """Return the first answer position with ``token_id``, as official code does."""

    target = int(token_id)
    for index, value in enumerate(response_token_ids):
        if int(value) == target:
            return int(index)
    return None


def _default_llava_first_id(tokenizer: Any, surface: str) -> int | None:
    encoded = tokenizer(
        surface,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return _first_input_id(encoded)


def _special_first_id(
    *,
    tokenizer: Any,
    surface: str,
    branch: str,
) -> int | None:
    if branch == "internvl_find_word":
        encoded = tokenizer(
            surface,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return _first_input_id(encoded)
    if branch in {"qwenvl3_find_word", "llava_onevision_find_word"}:
        encoded = tokenizer.encode(surface)
        return _first_scalar(encoded)
    raise ValueError(f"Unknown InsLen resolver branch: {branch}")


def _first_input_id(encoded: Any) -> int | None:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return _first_scalar(encoded)


def _first_scalar(value: Any) -> int | None:
    current = value
    while current is not None:
        if hasattr(current, "tolist"):
            current = current.tolist()
            continue
        if isinstance(current, (list, tuple)):
            if not current:
                return None
            current = current[0]
            continue
        try:
            return int(current)
        except (TypeError, ValueError):
            return None
    return None


def _not_found(
    *,
    branch: str,
    detected_word: str,
    reason: str,
    attempts: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "status": "not_found",
        "resolver_branch": branch,
        "detected_word": str(detected_word),
        "candidate_surface": None,
        "target_token_id": None,
        "candidate_attempts": [str(value) for value in attempts],
        "skip_reason": str(reason),
    }
