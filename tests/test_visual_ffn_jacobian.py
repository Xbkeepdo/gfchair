import unittest
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from features.visual_ffn_jacobian import (
    aggregate_visual_jvp,
    build_jffn_source_payload,
    choose_adaptive_jvp_chunk_size,
    entropy_matched_jffn_distribution,
    exact_visual_token_jvps,
    reconstruct_visual_directions,
    resolve_decoder_layer_adapter,
)


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = nn.Sequential(nn.Linear(4, 7), nn.SiLU(), nn.Linear(7, 4))


class _SeparateAttention(nn.Module):
    def __init__(self, *, output_bias: bool = True) -> None:
        super().__init__()
        self.head_dim = 2
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.num_key_value_groups = 2
        self.config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.v_proj = nn.Linear(8, 4, bias=True)
        self.o_proj = nn.Linear(8, 8, bias=output_bias)


class _SeparateLayer(nn.Module):
    def __init__(self, *, output_bias: bool = True) -> None:
        super().__init__()
        self.self_attn = _SeparateAttention(output_bias=output_bias)
        self.input_layernorm = nn.LayerNorm(8)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 13), nn.SiLU(), nn.Linear(13, 8))


class _PackedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 2
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.num_key_value_groups = 2
        # [Hkv, query_group_0, query_group_1, key, value, Dh]
        self.wqkv = nn.Linear(8, 2 * (2 + 2) * 2, bias=True)
        self.wo = nn.Linear(8, 8, bias=True)


class _PackedLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _PackedAttention()
        self.attention_norm = nn.LayerNorm(8)
        self.ffn_norm = nn.LayerNorm(8)
        self.feed_forward = nn.Sequential(
            nn.Linear(8, 15), nn.SiLU(), nn.Linear(15, 8)
        )


class _TinyModel(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer])


def _capture_for_layer(layer: nn.Module, *, sequence_length: int = 6):
    positions = [4, 5]
    h_prev = torch.randn(sequence_length, 8)
    adapter = resolve_decoder_layer_adapter(layer)
    normalized = adapter.attention_norm(h_prev)
    projected = adapter.value_projection(normalized)
    if adapter.family == "internlm2_packed_qkv":
        values = projected.view(sequence_length, 2, 4, 2)[:, :, -1, :]
    else:
        values = projected.view(sequence_length, 2, 2)
    values = values.permute(1, 0, 2).repeat_interleave(2, dim=0)
    attention = torch.softmax(torch.randn(2, 4, sequence_length), dim=-1)
    heads = torch.einsum("ths,hsd->thd", attention, values).reshape(2, 8)
    updates = F.linear(
        heads,
        adapter.output_projection.weight,
        adapter.output_projection.bias,
    )
    h_mid = h_prev.clone()
    h_mid[positions] += updates
    capture = {
        "h_prev": h_prev.unsqueeze(0),
        "h_mid": h_mid.unsqueeze(0),
        "o_ffn": torch.zeros_like(h_mid).unsqueeze(0),
        "attn_weights": attention.permute(1, 0, 2).unsqueeze(0),
        "attention_query_positions": tuple(positions),
    }
    return capture, positions, updates


class VisualFFNJacobianTest(unittest.TestCase):
    def test_full_and_chunked_vmap_match_and_sum_is_linear(self) -> None:
        torch.manual_seed(7)
        layer = _TinyLayer().eval()
        z = torch.randn(3, 4)
        directions = torch.randn(6, 3, 4)
        full, _ = exact_visual_token_jvps(
            layer=layer, z=z, a_tokens=directions, chunk_size=None
        )
        chunked, _ = exact_visual_token_jvps(
            layer=layer, z=z, a_tokens=directions, chunk_size=2
        )
        self.assertTrue(torch.allclose(full, chunked, atol=1e-6, rtol=1e-6))
        aggregate = aggregate_visual_jvp(
            layer=layer, z=z, a_visual=directions.sum(dim=0)
        )
        self.assertTrue(
            torch.allclose(full.sum(dim=0), aggregate, atol=2e-6, rtol=2e-6)
        )

    def test_separate_qkv_gqa_reconstruction_and_bias_policy(self) -> None:
        torch.manual_seed(11)
        layer = _SeparateLayer(output_bias=True).eval()
        capture, positions, updates = _capture_for_layer(layer)
        result = reconstruct_visual_directions(
            layer=layer,
            capture=capture,
            prediction_positions=positions,
            visual_start=0,
            visual_end=6,
        )
        self.assertLess(result["reconstruction_relative_error"], 1e-6)
        self.assertGreater(result["reconstruction_cosine"], 0.999999)
        self.assertLess(result["component_sum_relative_error"], 1e-6)
        bias = layer.self_attn.o_proj.bias.detach().unsqueeze(0)
        self.assertTrue(
            torch.allclose(result["a_visual"], updates - bias, atol=2e-6, rtol=2e-6)
        )

    def test_internlm2_packed_value_slot_and_gqa_reconstruction(self) -> None:
        torch.manual_seed(13)
        layer = _PackedLayer().eval()
        capture, positions, updates = _capture_for_layer(layer)
        result = reconstruct_visual_directions(
            layer=layer,
            capture=capture,
            prediction_positions=positions,
            visual_start=0,
            visual_end=6,
        )
        self.assertEqual(result["adapter_family"], "internlm2_packed_qkv")
        self.assertLess(result["reconstruction_relative_error"], 1e-6)
        self.assertLess(result["component_sum_relative_error"], 1e-6)
        bias = layer.attention.wo.bias.detach().unsqueeze(0)
        self.assertTrue(
            torch.allclose(result["a_visual"], updates - bias, atol=2e-6, rtol=2e-6)
        )

    def test_payload_is_finite_normalized_and_batches_targets(self) -> None:
        torch.manual_seed(17)
        layer = _SeparateLayer(output_bias=False).eval()
        capture, positions, _updates = _capture_for_layer(layer)
        payload = build_jffn_source_payload(
            model=_TinyModel(layer),
            captures=[capture],
            prediction_positions=positions,
            visual_start=0,
            visual_end=6,
            entropy_beta_by_layer=[1.7],
            fixed_chunk_size=2,
        )
        self.assertEqual(len(payload), 1)
        self.assertFalse(payload[0]["validation_computed"])
        for field in (
            "input_norm",
            "response_norm",
            "gain",
            "direction_cosine",
            "response_sum_relative_error",
            "local_fd_cosine",
            "local_fd_relative_error",
        ):
            self.assertTrue(torch.isfinite(payload[0][field]).all(), field)
        self.assertTrue(
            torch.isfinite(
                torch.tensor(payload[0]["parallel_chunk_relative_error"])
            )
        )
        for mode in ("jffn", "jffn_entropy_matched"):
            distribution = payload[0]["distributions"][mode]
            self.assertEqual(tuple(distribution.shape), (2, 6))
            self.assertTrue(torch.isfinite(distribution).all())
            self.assertTrue(
                torch.allclose(distribution.sum(dim=-1), torch.ones(2), atol=1e-6)
            )

    def test_entropy_matching_preserves_energy_ranking(self) -> None:
        energy = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        matched = entropy_matched_jffn_distribution(energy, beta=2.5)
        self.assertEqual(
            torch.argsort(energy, dim=-1).tolist(),
            torch.argsort(matched, dim=-1).tolist(),
        )
        self.assertAlmostEqual(float(matched.sum()), 1.0, places=6)

    def test_second_round_write_signed_and_attention_payload(self) -> None:
        torch.manual_seed(19)
        layer = _SeparateLayer(output_bias=True).eval()
        capture, positions, _updates = _capture_for_layer(layer)
        payload = build_jffn_source_payload(
            model=_TinyModel(layer),
            captures=[capture],
            prediction_positions=positions,
            visual_start=0,
            visual_end=6,
            fixed_chunk_size=2,
            include_second_round_diagnostics=True,
        )[0]
        second = payload["second_round"]
        for field in (
            "attention_distribution",
            "write_distribution",
        ):
            self.assertEqual(tuple(second[field].shape), (2, 6))
            self.assertTrue(
                torch.allclose(second[field].sum(dim=-1), torch.ones(2), atol=1e-6)
            )
        self.assertEqual(tuple(second["write_energy"].shape), (2, 6))
        self.assertEqual(tuple(second["signed_q"].shape), (2, 6))
        self.assertLess(
            float(second["signed_q_conservation_relative_error"].max()),
            2e-6,
        )
        self.assertTrue(
            torch.allclose(
                second["signed_q"].sum(dim=-1),
                payload["response_norm"],
                atol=2e-6,
                rtol=2e-6,
            )
        )

    def test_vector_path_only_payload_has_compact_js_ot_statistics(self) -> None:
        torch.manual_seed(23)
        layer = _SeparateLayer(output_bias=False).eval()
        capture, positions, _updates = _capture_for_layer(layer)
        target = torch.softmax(torch.randn(2, 1, 6), dim=-1)
        payload = build_jffn_source_payload(
            model=_TinyModel(layer),
            captures=[capture],
            prediction_positions=positions,
            visual_start=0,
            visual_end=6,
            vector_path_integration_points=4,
            vector_path_chunk_size=2,
            vector_path_only=True,
            vector_path_target_distributions=target,
            vector_path_evidence_strengths=torch.ones(2, 1),
        )[0]["vector_path"]
        self.assertEqual(tuple(payload["ffn_path_gross"].shape), (2, 6))
        self.assertEqual(tuple(payload["path_signed_q"].shape), (2, 6))
        self.assertTrue(
            torch.allclose(payload["ffn_distribution"].sum(-1), torch.ones(2))
        )
        for family in ("js", "ot"):
            for name in ("D_EW", "D_WF", "D_EF"):
                self.assertEqual(tuple(payload[family][name].shape), (2,))
                self.assertTrue(torch.isfinite(payload[family][name]).all())

    def test_adaptive_chunk_obeys_tiny_budget(self) -> None:
        layer = _SeparateLayer().eval()
        chunk = choose_adaptive_jvp_chunk_size(
            layer=layer,
            num_visual_tokens=576,
            num_targets=8,
            dtype=torch.float16,
            max_increment_bytes=1,
        )
        self.assertEqual(chunk, 64)


if __name__ == "__main__":
    unittest.main()
