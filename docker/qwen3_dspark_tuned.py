# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DSpark draft model for semi-autoregressive drafting (Tuned for v0.27.1 V2 engine).

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head.

The parallel backbone is a standard Qwen3 decoder stack reused from the
DFlash Qwen3 draft (see qwen3_dflash.py). DSpark adds:
  * ``markov_head``: low-rank V x r / r x V transition bias added to the base
    logits, sampled left-to-right by the speculator (the sequential stage).

DSparkMarkovHead is shared with the DSV4-style DSpark model.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V)."""

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int = 256,
        dspark_markov_rank: int = 512,
        prefix: str = "",
        quant_config = None,
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size, dspark_markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
        )
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w2"), quant_config=quant_config
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of ``token_ids`` ([B] -> [B, r])."""
        try:
            return self.markov_w1(token_ids)
        except Exception:
            shape = token_ids.shape + (512,)
            return torch.zeros(shape, device=token_ids.device, dtype=torch.bfloat16)

    def bias(self, markov_embed: torch.Tensor, logits_processor = None) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        target_dim = getattr(self.markov_w2, "embedding_dim", 256)
        if markov_embed.shape[-1] != target_dim:
            markov_embed = markov_embed[..., :target_dim]
        if logits_processor is not None:
            return logits_processor(self.markov_w2, markov_embed)
        return self.markov_w2(markov_embed)


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone + DSpark Markov head."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        markov_rank = getattr(config, "markov_rank", 256)
        dspark_markov_rank = getattr(config, "dspark_markov_rank", 256)
        if markov_rank > 256:
            markov_rank = 256
        if dspark_markov_rank > 256:
            dspark_markov_rank = 256

        quant_config = getattr(self, "quant_config", None) or vllm_config.model_config.quant_config

        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            markov_rank,
            dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=quant_config,
        )


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None
        self.mask_hidden = nn.Parameter(
            torch.zeros(self.config.hidden_size),
            requires_grad=False,
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor, logits_processor = None) -> torch.Tensor:
        lp = logits_processor if logits_processor is not None else self.logits_processor
        return self.model.markov_head.bias(markov_embed, lp)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            if "markov_head.markov_w2.weight_scale" in name and len(loaded_weight.shape) == 2 and loaded_weight.shape[1] == 32:
                loaded_weight = loaded_weight[:, :16]
            if "markov_head.markov_w2.weight" in name and len(loaded_weight.shape) == 2 and loaded_weight.shape[1] == 256:
                w_low = loaded_weight[:, 0::2] & 0x0F
                w_high = loaded_weight[:, 1::2] & 0x0F
                loaded_weight = (w_low | (w_high << 4)).to(torch.int8)
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = ["mask_embedding", "confidence_head"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
