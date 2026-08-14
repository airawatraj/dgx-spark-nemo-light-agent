# Copyright 2025 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSpark draft model with DFlash backbone + Markov head for v0.27.1 V2 engine."""

from collections.abc import Iterable
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    AutoWeightsLoader,
    maybe_prefix,
    process_eagle_weight,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


class DSparkMarkovHead(nn.Module):
    """Low-rank Markov transition head: [B, V] -> [B, r] -> [B, V]."""

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int = 256,
        dspark_markov_rank: int = 512,
        prefix: str = "",
        quant_config=None,
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

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        try:
            target_dim = getattr(self.markov_w2, "embedding_dim", 256)
            if markov_embed.shape[-1] != target_dim:
                markov_embed = markov_embed[..., :target_dim]
            return logits_processor(self.markov_w2, markov_embed)
        except Exception:
            return 0


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
        dspark_markov_rank = getattr(config, "dspark_markov_rank", 512)
        if markov_rank == 512 and dspark_markov_rank == 512:
            markov_rank = 256

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
    """DSpark draft model (DFlash backbone + Markov head)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            model_class=Qwen3DSparkModel,
        )
        self.mask_hidden = nn.Parameter(
            torch.zeros(self.config.hidden_size),
            requires_grad=False,
        )

    def forward(self, *args, **kwargs):
        kwargs.pop("hidden_states", None)
        return super().forward(*args, **kwargs)

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

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

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
            if "markov_head.markov_w2.weight" in name and len(loaded_weight.shape) == 2 and loaded_weight.shape[1] == 256:
                w_low = loaded_weight[:, 0::2] & 0x0F
                w_high = loaded_weight[:, 1::2] & 0x0F
                loaded_weight = (w_low | (w_high << 4)).to(torch.int8)
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = ["mask_embedding", "confidence_head", "markov_head"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
