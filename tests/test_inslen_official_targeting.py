from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.extractor import _resolve_causal_target, _targeting_metadata
from models.dgst_capture import target_logits_multi
from utils.inslen_targeting import (
    INSLEN_INTERNVL_BOS_COMPATIBILITY_NOTE,
    INSLEN_SAMPLE_UNIT,
    INSLEN_TARGET_PROTOCOL,
    INSLEN_UPSTREAM_COMMIT,
    first_token_occurrence,
    inslen_model_branch,
    resolve_inslen_candidate,
)


def _load_label_module():
    path = ROOT / "coco-labeling" / "label_coco.py"
    spec = importlib.util.spec_from_file_location("gfchair_label_coco", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = {str(key): list(value) for key, value in mapping.items()}
        self.calls: list[str] = []
        self.call_kwargs: list[dict] = []

    def encode(self, surface: str) -> list[int]:
        self.calls.append(surface)
        return list(self.mapping.get(surface, []))

    def __call__(self, surface: str, **_kwargs):
        self.calls.append(surface)
        self.call_kwargs.append(dict(_kwargs))
        return {"input_ids": [list(self.mapping.get(surface, []))]}


class _BosAddingTokenizer(_FakeTokenizer):
    def __call__(self, surface: str, **kwargs):
        self.calls.append(surface)
        self.call_kwargs.append(dict(kwargs))
        token_ids = list(self.mapping.get(surface, []))
        if kwargs.get("add_special_tokens", True):
            token_ids.insert(0, 1)
        return {"input_ids": [token_ids]}


class InsLenTargetResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.label_coco = _load_label_module()

    def test_model_family_mapping_and_llava_next_rejection(self) -> None:
        self.assertEqual(inslen_model_branch("llava_1_5_7b"), "llava_default_surface")
        self.assertEqual(inslen_model_branch("qwen2_5_vl_7b"), "qwenvl3_find_word")
        self.assertEqual(inslen_model_branch("qwen3_vl_8b"), "qwenvl3_find_word")
        self.assertEqual(inslen_model_branch("internvl_2_5_8b"), "internvl_find_word")
        with self.assertRaisesRegex(ValueError, "no released LLaVA-NeXT resolver"):
            inslen_model_branch("llava_next_8b")

    def test_default_llava_uses_raw_surface_not_normalized_word(self) -> None:
        tokenizer = _FakeTokenizer({"cars": [22], "car": [11]})
        result = resolve_inslen_candidate(
            model_key="llava_1_5_7b",
            tokenizer=tokenizer,
            response_token_ids=[7, 22, 8],
            raw_caption_tokens=["cars"],
            mention={"normalized_word": "car", "word_idx": 0},
        )
        self.assertEqual(result["detected_word"], "cars")
        self.assertEqual(result["target_token_id"], 22)
        self.assertEqual(tokenizer.calls, ["cars"])

    def test_qwen_find_word_uses_official_prefix_suffix_order(self) -> None:
        tokenizer = _FakeTokenizer(
            {
                "car": [11],
                "cars": [12],
                "cares": [13],
                " car": [21],
                " cars": [22],
            }
        )
        result = resolve_inslen_candidate(
            model_key="qwen2_5_vl_7b",
            tokenizer=tokenizer,
            response_token_ids=[7, 22, 8],
            raw_caption_tokens=["cars"],
            mention={"normalized_word": "car", "word_idx": 0},
        )
        self.assertEqual(result["candidate_surface"], " cars")
        self.assertEqual(result["target_token_id"], 22)
        self.assertEqual(
            result["candidate_attempts"],
            ["car", "cars", "cares", " car", " cars"],
        )

    def test_internvl_find_word_excludes_bos_before_first_subtoken_lookup(self) -> None:
        tokenizer = _BosAddingTokenizer({" officer": [9581]})
        result = resolve_inslen_candidate(
            model_key="internvl_2_5_8b",
            tokenizer=tokenizer,
            response_token_ids=[7, 9581, 8],
            raw_caption_tokens=["officer"],
            mention={"normalized_word": "officer", "word_idx": 0},
        )
        self.assertEqual(result["candidate_surface"], " officer")
        self.assertEqual(result["target_token_id"], 9581)
        self.assertTrue(tokenizer.call_kwargs)
        self.assertTrue(
            all(
                call.get("add_special_tokens") is False
                for call in tokenizer.call_kwargs
            )
        )

    def test_person_complement_and_unresolved_skip(self) -> None:
        tokenizer = _FakeTokenizer({" people": [41]})
        found = resolve_inslen_candidate(
            model_key="qwen3_vl_8b",
            tokenizer=tokenizer,
            response_token_ids=[41],
            raw_caption_tokens=["people"],
            mention={"normalized_word": "person", "word_idx": 0},
        )
        self.assertEqual(found["candidate_surface"], " people")
        self.assertEqual(found["target_token_id"], 41)

        missing = resolve_inslen_candidate(
            model_key="qwen3_vl_8b",
            tokenizer=_FakeTokenizer({" dog": [51]}),
            response_token_ids=[99],
            raw_caption_tokens=["dogs"],
            mention={"normalized_word": "dog", "word_idx": 0},
        )
        self.assertEqual(missing["status"], "not_found")
        self.assertIsNone(missing["target_token_id"])

    def test_special_model_searches_generated_word_not_canonical_node(self) -> None:
        tokenizer = _FakeTokenizer({" woman": [61], " person": [62]})
        chair_info = {
            "object_mentions": [
                {
                    "word": "woman",
                    "normalized_word": "woman",
                    "canonical_object": "person",
                    "surface": "woman",
                    "word_idx": 0,
                    "label": 1,
                }
            ],
            "metrics": {"CHAIRs": 0, "CHAIRi": 0.0},
        }
        _all_spans, selected, _summary = (
            self.label_coco._inslen_official_token_spans(
                model_key="qwen2_5_vl_7b",
                tokenizer=tokenizer,
                caption="woman",
                token_ids=[61],
                chair_info=chair_info,
            )
        )
        self.assertEqual(selected[0]["word"], "woman")
        self.assertEqual(selected[0]["canonical_object"], "person")
        self.assertEqual(selected[0]["target_token_id"], 61)
        self.assertFalse(any("person" in value for value in tokenizer.calls))

    def test_first_occurrence_is_used_for_the_formal_target(self) -> None:
        tokenizer = _FakeTokenizer(
            {
                "handbag": [10661],
                "handbags": [10662],
                "handbages": [10663],
                " handbag": [1424, 21250],
            }
        )
        chair_info = {
            "object_mentions": [
                {
                    "surface": "handbag",
                    "normalized_word": "handbag",
                    "canonical_object": "handbag",
                    "word_idx": 1,
                    "char_start": 5,
                    "char_end": 12,
                    "label": 1,
                }
            ],
            "metrics": {"CHAIRs": 0, "CHAIRi": 0.0},
        }
        response_ids = [1424, 9, 1424, 21250]
        all_spans, selected, summary = self.label_coco._inslen_official_token_spans(
            model_key="qwen2_5_vl_7b",
            tokenizer=tokenizer,
            caption="hand handbag",
            token_ids=response_ids,
            chair_info=chair_info,
        )
        self.assertEqual(first_token_occurrence(response_ids, 1424), 0)
        self.assertEqual(selected[0]["token_indices"], [0])
        self.assertEqual(selected[0]["target_token_id"], 1424)
        self.assertEqual(selected[0]["inslen_official_resolution"]["status"], "found")
        self.assertEqual(len(all_spans), 1)

        index, target_id, prefix = _resolve_causal_target(
            response_token_ids=response_ids,
            span=selected[0],
            image_id=1,
        )
        self.assertEqual((index, target_id, prefix), (0, 1424, []))

        corrupted = dict(selected[0])
        corrupted["target_token_id"] = 9
        with self.assertRaisesRegex(ValueError, "target ID mismatch"):
            _resolve_causal_target(
                response_token_ids=response_ids,
                span=corrupted,
                image_id=1,
            )

    def test_duplicate_detected_word_and_missing_word_are_not_selected(self) -> None:
        tokenizer = _FakeTokenizer({" car": [22], " dog": [33]})
        mentions = [
            {
                "normalized_word": "car",
                "canonical_object": "car",
                "word_idx": 0,
                "label": 1,
            },
            {
                "normalized_word": "car",
                "canonical_object": "car",
                "word_idx": 1,
                "label": 1,
            },
            {
                "normalized_word": "dog",
                "canonical_object": "dog",
                "word_idx": 2,
                "label": 0,
            },
        ]
        chair_info = {
            "object_mentions": mentions,
            "metrics": {"CHAIRs": 1, "CHAIRi": 1 / 3},
        }
        all_spans, selected, summary = self.label_coco._inslen_official_token_spans(
            model_key="qwen2_5_vl_7b",
            tokenizer=tokenizer,
            caption="car car dog",
            token_ids=[22, 22, 99],
            chair_info=chair_info,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["official_detected_word"], "car")
        self.assertEqual(
            all_spans[1]["inslen_official_resolution"]["skip_reason"],
            "duplicate_detected_word",
        )
        self.assertEqual(all_spans[2]["inslen_official_resolution"]["status"], "not_found")
        self.assertEqual(summary["duplicate_objects_skipped"], 1)
        self.assertEqual(summary["skipped_or_unresolved_objects"], 2)

    def test_default_llava_dedupes_after_candidate_before_membership(self) -> None:
        tokenizer = _FakeTokenizer({"car": [11]})
        mentions = [
            {
                "word": "car",
                "normalized_word": "car",
                "canonical_object": "car",
                "word_idx": index,
                "label": 1,
            }
            for index in (0, 1)
        ]
        all_spans, selected, summary = (
            self.label_coco._inslen_official_token_spans(
                model_key="llava_1_5_7b",
                tokenizer=tokenizer,
                caption="car car",
                token_ids=[99],
                chair_info={
                    "object_mentions": mentions,
                    "metrics": {"CHAIRs": 0, "CHAIRi": 0.0},
                },
            )
        )
        self.assertEqual(selected, [])
        self.assertEqual(
            all_spans[0]["inslen_official_resolution"]["skip_reason"],
            "candidate_token_id_absent_from_output",
        )
        self.assertEqual(
            all_spans[1]["inslen_official_resolution"]["skip_reason"],
            "duplicate_detected_word",
        )
        self.assertEqual(summary["duplicate_objects_skipped"], 1)

    def test_compact_entry_and_feature_metadata_declare_protocol(self) -> None:
        resolution = {
            "status": "found",
            "resolver_branch": "qwenvl3_find_word",
            "detected_word": "car",
            "candidate_surface": " cars",
            "target_token_id": 22,
            "first_occurrence_index": 3,
        }
        span = {
            "word": "car",
            "canonical_object": "car",
            "surface": "cars",
            "token_indices": [3],
            "target_token_id": 22,
            "label": 1,
            "target_locator_protocol": INSLEN_TARGET_PROTOCOL,
            "inslen_official_resolution": resolution,
        }
        entry = self.label_coco._compact_inslen_label_entry(
            image_id=7,
            model_key="qwen2_5_vl_7b",
            caption="cars",
            all_spans=[span],
            selected_spans=[span],
            chair_info={"metrics": {"CHAIRs": 0, "CHAIRi": 0.0}},
            resolution_summary={
                "chair_object_mentions": 1,
                "resolved_unique_objects": 1,
                "skipped_or_unresolved_objects": 0,
                "duplicate_objects_skipped": 0,
            },
        )
        self.assertEqual(
            entry["labeling_protocol"]["primary_locator"],
            INSLEN_TARGET_PROTOCOL,
        )
        self.assertEqual(
            entry["labeling_protocol"]["sample_unit"], INSLEN_SAMPLE_UNIT
        )
        self.assertEqual(
            entry["labeling_protocol"]["upstream_commit"],
            INSLEN_UPSTREAM_COMMIT,
        )
        metadata = _targeting_metadata(span)
        self.assertEqual(metadata["official_resolver_branch"], "qwenvl3_find_word")
        self.assertNotIn("shadow_exact_response_token_indices", metadata)

    def test_internvl_compatibility_note_invalidates_old_bos_labels(self) -> None:
        samples = [{"image_id": 1}]
        generations = {
            "1": {
                "generated_text": "an officer",
                "response_token_ids": [7, 9581],
            }
        }
        entry = self.label_coco._compact_inslen_label_entry(
            image_id=1,
            model_key="internvl_2_5_8b",
            caption="an officer",
            all_spans=[],
            selected_spans=[],
            chair_info={"metrics": {"CHAIRs": 0, "CHAIRi": 0.0}},
            resolution_summary={
                "chair_object_mentions": 0,
                "resolved_unique_objects": 0,
                "skipped_or_unresolved_objects": 0,
                "duplicate_objects_skipped": 0,
            },
        )
        self.assertEqual(
            entry["resolver_compatibility_note"],
            INSLEN_INTERNVL_BOS_COMPATIBILITY_NOTE,
        )
        expected = {
            "expected_sample_unit": INSLEN_SAMPLE_UNIT,
            "expected_primary_locator": INSLEN_TARGET_PROTOCOL,
            "expected_upstream_commit": INSLEN_UPSTREAM_COMMIT,
            "expected_resolver_model": "internvl_2_5_8b",
            "expected_resolver_compatibility_note": (
                INSLEN_INTERNVL_BOS_COMPATIBILITY_NOTE
            ),
        }
        old_entry = dict(entry, resolver_compatibility_note=None)
        self.assertFalse(
            self.label_coco._all_labeling_available(
                samples,
                generations,
                {"1": old_entry},
                **expected,
            )
        )
        self.assertTrue(
            self.label_coco._all_labeling_available(
                samples,
                generations,
                {"1": entry},
                **expected,
            )
        )

    def test_logit_lens_indexes_the_resolved_first_subtoken_id(self) -> None:
        output_layer = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            output_layer.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [2.0, 3.0],
                        [-1.0, 4.0],
                    ]
                )
            )
        states = torch.tensor([[5.0, 7.0]])
        logits = target_logits_multi(
            output_layer=output_layer,
            states=states,
            target_token_ids=[2],
        )
        self.assertEqual(tuple(logits.shape), (1, 1))
        self.assertAlmostEqual(float(logits[0, 0]), 31.0)


if __name__ == "__main__":
    unittest.main()
