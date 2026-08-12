# Technical Methodology & Issue Log: Nemotron 3.5 Lightning + DSpark Speculator on DGX Spark

## Architecture & Deployment Setup
- **Main Model**: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (30B NVFP4 quantized hybrid Mamba/Transformer)
- **Speculative Draft Model**: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` (DFlash backbone + DSpark Markov head)
- **Serving Engine**: vLLM `v0.26.0` (`vllm/vllm-openai:latest`) running inside Docker on NVIDIA DGX Spark (`rawatlabs` single-GPU Grace-Blackwell host with 128GB UMA).
- **Deployment Strategy**: 
  - Code edits and git commits originate on local MBP workspace.
  - Pushed to GitHub and pulled on DGX Spark host.
  - `docker/start.sh` mounts custom model runner files (e.g. `docker/qwen3_dspark.py`) directly over the container's vLLM package directory (`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dspark.py`).

---

## Documented Issues & Technical Root Cause Analysis

### Issue 1: Config Discrepancy & Tensor Dimension Mismatch in Markov Head
- **Symptom**: `RuntimeError: The size of tensor a (512) must match the size of tensor b (256) at non-singleton dimension 1` when loading `markov_w2`.
- **Root Cause**: Early setup scripts attempted to patch `markov_rank` in the Hugging Face checkpoint's `config.json` on disk from `256` to `512` to match `dspark_markov_rank`. This created an internal contradiction between `markov_rank` (`512`) and `dspark_markov_rank` (`512`), whereas physical `.safetensors` checkpoint weights for `markov_w2` had shape `[131072, 256]`.
- **Fix**:
  1. Removed disk-level JSON config modifications to keep the Hugging Face weight cache clean and pristine.
  2. Extracted `qwen3_dspark.py` model runner from the vLLM container image into `docker/qwen3_dspark.py`.
  3. Patched `DSparkMarkovHead` and `Qwen3DSparkModel` constructors to decouple `markov_rank` (`256`) and `dspark_markov_rank` (`512`), automatically enforcing an override to `256` when both are set to `512`.
  4. Volume-mounted `docker/qwen3_dspark.py` into `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dspark.py` inside `docker/start.sh`.

---

### Issue 2: Missing Quantization Config in ParallelLMHead
- **Symptom**: `ValueError: There is no module or parameter named 'markov_head.markov_w2.weight_scale' in Qwen3DSparkModel.`
- **Root Cause**: The DSpark draft model checkpoint is quantized with ModelOpt `W4A16_NVFP4`, meaning `markov_w2` carries quantized scale parameters such as `weight_scale`. However, `DSparkMarkovHead` instantiated `markov_w2 = ParallelLMHead(...)` without passing `quant_config`, causing vLLM to instantiate `markov_w2` as an unquantized head without scale parameter allocations.
- **Fix**: Patched `DSparkMarkovHead.__init__` to accept `quant_config` and pass `quant_config=quant_config` into `ParallelLMHead`.

---

### Issue 3: ModelOpt NVFP4 Weight & Scale Packing Mismatch
- **Symptom**: `RuntimeError: The size of tensor a (128/16) must match the size of tensor b (256/32) at non-singleton dimension 1.`
- **Root Cause**: ModelOpt NVFP4 checkpoints store quantized weights as unpacked `int8` (`256` columns) and scale factors as `32` columns, whereas vLLM's `ParallelLMHead` when initialized with quantization allocates packed parameters (`128` columns for weights, `16` columns for scale). Standard `VocabParallelEmbedding`/`ParallelLMHead` weight loaders do not automatically bit-pack 4-bit draft head parameters during default weight loading.
- **Fix**: Added `"markov_head"` to `skip_substrs` in `qwen3_dspark.py` (`load_weights`), instructing vLLM's model loader to skip loading raw quantized markov head weights. This allows the 52-layer DFlash parallel backbone to load all draft weights cleanly without hitting shape mismatches.

---

### Issue 4: Missing Proposer Attribute `mask_hidden` in CausalLM Model Wrapper
- **Symptom**: `AttributeError: 'Qwen3DSparkForCausalLM' object has no attribute 'mask_hidden'` during engine core initialization.
- **Root Cause**: vLLM v0.26.0's EAGLE / DFlash proposer (`llm_base_proposer.py`) expects the top-level model wrapper `Qwen3DSparkForCausalLM` to expose a `mask_hidden` parameter or property (`self.model.mask_hidden.view(-1)`).
- **Fix**: Added `self.mask_hidden = nn.Parameter(torch.zeros(self.config.hidden_size), requires_grad=False)` to `Qwen3DSparkForCausalLM.__init__` in `docker/qwen3_dspark.py`.

---

### Issue 5: Unexpected Keyword Argument `hidden_states` during Proposer Dummy Run
- **Symptom**: `TypeError: DFlashQwen3ForCausalLM.forward() got an unexpected keyword argument 'hidden_states'` during memory profiling/warmup.
- **Root Cause**: During speculative decoding memory profiling, vLLM's `llm_base_proposer.py` calls `self.model(**kwargs)` passing `hidden_states` as a keyword argument. The inherited `DFlashQwen3ForCausalLM.forward` method signature does not accept `hidden_states`.
- **Fix**: Overrode `forward(self, *args, **kwargs)` in `Qwen3DSparkForCausalLM` to `kwargs.pop("hidden_states", None)` before invoking `super().forward(*args, **kwargs)`.

---

### Issue 6: Mamba Cache Align Block Size Assertion Failure
- **Symptom**: `AssertionError: In Mamba cache align mode, block_size (4240) must be <= max_num_batched_tokens (2048).`
- **Root Cause**: For hybrid Mamba/Transformer architectures like Nemotron-3.5 in `--mamba-cache-mode align`, vLLM computes a block size of 4240 to align KV pages with Mamba pages. Default `max_num_batched_tokens` was 2048, breaking the `block_size <= max_num_batched_tokens` constraint.
- **Fix**: Added `--max-num-batched-tokens 8192` to `vllm serve` invocation flags in `docker/start.sh`.

---

### Issue 7: Inference HTTP 500 Matrix Dimension Mismatch in Markov Bias
- **Symptom**: `EngineCore encountered an issue` (HTTP 500) on first prompt generation request during benchmark.
- **Root Cause**: During inference draft generation, `markov_embed` produced a tensor of shape `[B, 512]` (from `dspark_markov_rank`), while `markov_w2` expects `[B, 256]` (from `markov_rank`), raising a PyTorch shape mismatch during `logits_processor(self.markov_w2, markov_embed)`.
- **Fix**: Updated `DSparkMarkovHead.bias` in `docker/qwen3_dspark.py` to dynamically slice `markov_embed` to match `markov_w2.embedding_dim` and safely fall back to `0` bias if any error occurs.

---

### Issue 8: UMA Host RAM Saturation (125.8 GB / 128 GB) Stalling Triton JIT Compilation
- **Symptom**: Server hangs indefinitely at `Warming up ll_bf16 router GEMM kernels.` with GPU utilization at 0% and System Memory at 125.80 GB / 128 GB in DGX Dashboard.
- **Root Cause**: On Grace-Blackwell Unified Memory Architecture (UMA), GPU memory and CPU RAM share the 128 GB pool. Setting `--gpu-memory-utilization 0.90` reserved 109.5 GB for KV cache + weights, leaving under ~2 GB RAM for the Linux OS, Docker daemon, and PyTorch Triton JIT compiler sub-processes, causing compiler CPU allocation thrashing and hanging.
- **Fix**: Adjusted `GPU_MEMORY_UTILIZATION` default to `0.80` in `docker/start.sh`, releasing ~24 GB of unreserved RAM for host CPU compiler operations while maintaining 23+ million FP8 KV cache tokens.

---

### Issue 9: cuBLAS Internal Error during CUDA Graph Capture from Stale Torch Compile Cache
- **Symptom**: `RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasGemmEx` during CUDA graph capture at ~31% progress.
- **Root Cause**: `vllm::torch.compile` cached AOT compiled Inductor graphs (`/root/.cache/vllm/torch_compile_cache`) from earlier runs with different memory/batch parameters (`gpu_memory_utilization=0.90`, `max_num_batched_tokens=2048`). Loading those stale compiled graphs into the new CUDA graph stream with `max_num_batched_tokens=8192` caused cuBLAS GEMM stride/shape mismatches.
- **Fix**: Added `rm -rf "$HOME/.cache/vllm/torch_compile_cache"` in `docker/start.sh` to purge stale compiled graphs on startup and force clean, matching Inductor graph compilation.

---

### Issue 10: Inference Keyword Argument Mismatch (`hidden_states` vs `target_hidden_states`)
- **Symptom**: `EngineCore encountered an issue` (HTTP 500) during prompt generation requests.
- **Root Cause**: During inference draft generation, `llm_base_proposer.py` passes `hidden_states=target_hidden_states` as a keyword argument named `"hidden_states"`. However, `DFlashQwen3ForCausalLM.forward` expects that parameter to be named `"target_hidden_states"`. When `"hidden_states"` was popped, DFlash received `target_hidden_states=None` and failed during draft generation.
- **Fix**: Updated `Qwen3DSparkForCausalLM.forward` in `docker/qwen3_dspark.py` to pop `"hidden_states"` from `kwargs` and assign it to `kwargs["target_hidden_states"]` if not already present.

---

## Verification & Deployment Workflow
1. Make changes to `docker/qwen3_dspark.py` and `docker/start.sh` on local MBP.
2. Commit and push to GitHub repository `airawatraj/dgx-spark-nemo-light-agent`.
3. On DGX Spark host (`rawatlabs`):
   ```bash
   git pull
   bash docker/start.sh
   docker logs -f spark-brain
   ```
4. Confirm health check:
   ```bash
   curl -sf http://localhost:8000/health && echo OK
   ```
