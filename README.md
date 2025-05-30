## First Block Cache and TeaCache, in Forge webUI ##
### accelerate inference at some, perhaps minimal, quality cost ###

derived, with lots of reworking, from:
* https://github.com/likelovewant/sd-forge-teacache (flux only, teacache only)

more info:
* https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4FLUX
* https://github.com/chengzeyi/Comfy-WaveSpeed

install:
**Extensions** tab, **Install from URL**, use URL for this repo

>[!NOTE]
>This handles SelfAttentionGuidance and PerturbedAttentionGuidance (and anything else that calculates a cond), and applies the caching to them too, independently.
>
>Previous implementation moved to `old` branch.
>
>(30/05/2025) pre-SD3/Chroma version moved to `less-old` branch

usage:
1. Enable the extension
2. select caching threshold: higher threshold = more caching = faster + lower quality
3. low step models (Hyper) will need higher threshold to do anything
4. Generate
5. You'll need to experiment to find settings that work with your favoured models, step counts, samplers.

>[!NOTE]
>Both methods work with SD1.5, SD2, SDXL (including separated cond processing), and Flux.
>
>(30/05/2025) added versions for SD3(.5) and Chroma. Caching SD3 does not seem to work especially well, tends to reduce detail too much, but may be more useful with higher steps.
>
>The use of cached residuals applies to the whole batch, so results will not be identical between different batch sizes. This is absolutely 100% *will not fix*.

Now works with batch_size > 1, but results will not be consistent with same seed at batch_size == 1.

Added option for maximum consecutive cached steps (0: no limit); and made not using cache for final step an option (previously always processed the final step).

Some samplers (DPM++ 2M, UniPC, likely others) need very low threshold and/or delayed start + limit to consecutive cached steps.
---
---
original README:

## Sd-Forge-TeaCache: Speed up Your Diffusion Models

**Introduction**

Timestep Embedding Aware Cache (TeaCache) is a revolutionary training-free caching approach that leverages the
fluctuating differences between model outputs across timesteps. This acceleration technique significantly boosts
inference speed for various diffusion models, including Image, Video, and Audio.

 TeaCache's integration into SD Forge WebUI for Flux only. Installation is as
straightforward as any other extension:

* **Clone:**  `git clone https://github.com/likelovewant/sd-forge-teacache.git`

into extensions directory ,relauch the system .


**Speed Up Your Diffusion Generation**

TeaCache can accelerate FLUX inference by up to 2x with minimal visual quality degradation, all without requiring any training. 

Within the Forge WebUI, you can easily adjust the following settings:

* **Relative L1 Threshold:** Controls the sensitivity of TeaCache's caching mechanism.
* **Steps:**  Matches the number of sampling steps used in TeaCache.

**Performance Tuning**

Based on [TeaCache4FLUX](https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4FLUX), you can achieve different
speedups:

* 0.25 threshold for 1.5x speedup
* 0.4 threshold for 1.8x speedup
* 0.6 threshold for 2.0x speedup
* 0.8 threshold for 2.25x speedup

**Important Notes:**

* **Maintain Consistency:** Keep the sampling steps in TeaCache aligned with the steps used in your Flux Sampling steps .Discrepancies can lead to lower quality outputs.
* **LoRA Considerations:** When utilizing LoRAs, adjust the steps or scales based on your GPU's capabilities. A recommended starting point is 28 steps or more.

To ensure smooth operation, remember to:

1. **Clear Residual Cache (optional):** When changing image sizes or disabling the TeaCache extension, always click "Clear Residual Cache" within the Forge WebUI. This prevents potential conflicts and maintains optimal performance.
2. **Disable TeaCache Properly:**  Ensure disable the TeaCache extension if you don't need it in your Forge WebUI. If not proper `Clear Residual Cache`, you may encounter unexpected behavior and require a full relaunch.


Several AI assistants has assisting with code generation and refinement for this extension based on the below resources.

**Credits and Resources**

This adaptation leverages [TeaCache4FLUX](https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4FLUX)
From ali-vilab TeaCache repository:[TeaCache](https://github.com/ali-vilab/TeaCache).

For additional information and other integrations, explore:

* [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache)

