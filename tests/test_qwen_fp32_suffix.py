import unittest

import torch

from features.qwen_fp32_suffix import qwen_query_layer_fp32


class QwenFP32SuffixTest(unittest.TestCase):
    def test_query_row_matches_full_decoder_layer_and_vjp(self):
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLTextConfig,
        )
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLDecoderLayer,
            Qwen2_5_VLRotaryEmbedding,
        )
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
            Qwen3VLTextRotaryEmbedding,
        )

        torch.manual_seed(20260829)
        variants = []
        qwen2_config = Qwen2_5_VLTextConfig(
            hidden_size=24,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 1]},
        )
        qwen2_config._attn_implementation = "eager"
        variants.append(
            (
                "qwen2",
                Qwen2_5_VLDecoderLayer(qwen2_config, 0).float().eval(),
                Qwen2_5_VLRotaryEmbedding(qwen2_config),
            )
        )
        qwen3_config = Qwen3VLTextConfig(
            hidden_size=24,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=6,
            rope_scaling={
                "mrope_interleaved": True,
                "mrope_section": [1, 1, 1],
                "rope_type": "default",
            },
        )
        qwen3_config._attn_implementation = "eager"
        variants.append(
            (
                "qwen3",
                Qwen3VLTextDecoderLayer(qwen3_config, 0).float().eval(),
                Qwen3VLTextRotaryEmbedding(qwen3_config),
            )
        )

        for family, layer, rotary in variants:
            with self.subTest(family=family):
                context = torch.randn(6, 24)
                query = torch.randn(1, 24, requires_grad=True)
                hidden = torch.cat((context, query), dim=0).unsqueeze(0)
                positions = torch.arange(7).view(1, 1, 7).expand(3, 1, 7)
                cos, sin = rotary(hidden, positions)
                mask = torch.full(
                    (7, 7), torch.finfo(torch.float32).min, dtype=torch.float32
                )
                mask = torch.triu(mask, diagonal=1).view(1, 1, 7, 7)
                if family == "qwen2":
                    full = layer(
                        hidden,
                        attention_mask=mask,
                        position_embeddings=(cos, sin),
                        use_cache=False,
                    )[0]
                else:
                    full = layer(
                        hidden,
                        attention_mask=mask,
                        position_embeddings=(cos, sin),
                        use_cache=False,
                    )
                compact = qwen_query_layer_fp32(
                    layer=layer,
                    clean_context=context,
                    query_states=query,
                    cos=cos,
                    sin=sin,
                    family=family,
                )
                self.assertTrue(
                    torch.allclose(compact, full[0, -1:], atol=2e-6, rtol=2e-6)
                )
                direction = torch.randn_like(compact)
                full_gradient = torch.autograd.grad(
                    (full[0, -1:] * direction).sum(), query, retain_graph=True
                )[0]
                compact_gradient = torch.autograd.grad(
                    (compact * direction).sum(), query
                )[0]
                self.assertTrue(
                    torch.allclose(
                        compact_gradient,
                        full_gradient,
                        atol=3e-6,
                        rtol=3e-6,
                    )
                )

                query_variants = torch.randn(7, 24)
                unbatched = qwen_query_layer_fp32(
                    layer=layer,
                    clean_context=context,
                    query_states=query_variants,
                    cos=cos,
                    sin=sin,
                    family=family,
                )
                microbatched = qwen_query_layer_fp32(
                    layer=layer,
                    clean_context=context,
                    query_states=query_variants,
                    cos=cos,
                    sin=sin,
                    family=family,
                    query_batch_size=2,
                )
                self.assertTrue(
                    torch.allclose(microbatched, unbatched, atol=2e-6, rtol=2e-6)
                )

                unbatched_leaf = query_variants.clone().requires_grad_(True)
                microbatched_leaf = query_variants.clone().requires_grad_(True)
                cotangent = torch.randn_like(query_variants)
                unbatched_output = qwen_query_layer_fp32(
                    layer=layer,
                    clean_context=context,
                    query_states=unbatched_leaf,
                    cos=cos,
                    sin=sin,
                    family=family,
                )
                microbatched_output = qwen_query_layer_fp32(
                    layer=layer,
                    clean_context=context,
                    query_states=microbatched_leaf,
                    cos=cos,
                    sin=sin,
                    family=family,
                    query_batch_size=2,
                )
                unbatched_vjp = torch.autograd.grad(
                    (unbatched_output * cotangent).sum(), unbatched_leaf
                )[0]
                microbatched_vjp = torch.autograd.grad(
                    (microbatched_output * cotangent).sum(), microbatched_leaf
                )[0]
                self.assertTrue(
                    torch.allclose(
                        microbatched_vjp,
                        unbatched_vjp,
                        atol=3e-6,
                        rtol=3e-6,
                    )
                )


if __name__ == "__main__":
    unittest.main()
