# Usage Notes

vLLM is very powerful, but it can be tough to use, especially for those unfamiliar with its twists and turns. This document exists to share some wisdom about the use of vLLM in ways that are less documented: targeting small-VRAM host configurations, and of course using EXL3 quantization.

## Why?

Running EXL3 quantizations on vLLM seems a little counterintuitive at first blush: EXL3 is a highly efficient quantization format, certainly exceeding that of anything else in the vLLM ecosystem, but the typical vLLM user operates at large scale, where the part of the system that EXL3 can help with (the model weights) is usually dwarfed by the enormous concurrent KV cache. EXL3 does offer an impressive trade as compared to the commonly deployed fp8, usually scoring better on divergence at a significantly smaller 6 bits per weight, but the tradeoff here is probably a rounding error for many large scale setups. It may still be of mild interest in these sorts of deployments -- any VRAM savings translates to higher throughput, which translates to free money -- but probably not enough for most operators to consider deploying it.

The more interesting target for this project is also counterintuitive, from the other direction: small-scale users with very modest hardware. vLLMs strengths lie in ways important to larger deployments: concurrency from continuous batching, etc. Why would a small-scale user want this enough to justify the complexity of using vLLM?

My answer is three-fold: one, there are small teams with very limited resources available that could still benefit from shared use of a small system, and vLLM has something to offer there. And two, even a single user may easily want higher degrees of concurrency; multiple agents, either independently or as parallel sub-agents within a single harness, have become quite common as local AI adoption continues. And finally, many hobbyists just enjoy exploring technology, and vLLM has plenty to offer in that regard.


## Something To Be Aware Of

A couple of vLLM behaviors to be aware of that might be mistaken for failures if they aren't expected:

- Loading of models that aren't already present in the huggingface cache will happen automatically, but vLLM provides no progress of any kind, nor indication that a download has started: it simply hangs silently during model loading, potentially for a long time depending on the size of the model and the speed of your network connection. Pre-loading into the cache using the usual huggingface tools (i.e. `hf download`) will provide sensible progress information and avoid this disconcerting behavior.

- vLLM in general can take a surprising amount of time the first time it loads a model -- it does a substantial amount of pre-compiling and autotuning, and depending on the host CPU this may take many minutes. This does get stored in a cache, and subsequent loads will be substantially faster, but the initial load can be a bit painful.

## Sizing

vLLM has several interacting knobs for setting the size of different aspects of the system, and it's important to know how these work.

- `--gpu-memory-utilization`: This specifies the amount of VRAM that vLLM will attempt to use, as a fraction of what is available.
- `--max-model-len`: This specifies the maximum context length that a served model will allow, up to its trained maximum. Note that this is NOT the same as KV cache space.
- `--max-num-seqs`: The expected level of concurrency that will happen during serving. If up to 8 requests are expected to be running at once, set it to 8.
- `--max-num-batched-tokens`: This controls the batch size used during prefill, and directly affects prefill performance, at a VRAM cost. Larger values do tend to have diminishing returns, but it is very system dependent.

The value for `gpu-memory-utilization` should be as high as possible to maximize utilization of VRAM, but there is a limit. The primary limit is that vLLM doesn't have access to the entire VRAM of the card(s) at startup. It logs exactly what it was, for example:

`Free memory on device (10.98/12.0 GiB) on startup`

In this case, `gpu-memory-utilization` cannot be greater than that literal fraction: `10.98 / 12.0` = `0.915` is the absolute upper limit. Normally one would leave some significant headroom here for safety from transient allocations and fragmentation, but this vLLM fork has many kinds of transients reduced or better profiled so it is generally possible to run very close to the limit here (see [patches.md](patches.md), "The commits" for details). When running at very high utilization it may also be important to enable expandable segments to avoid later failures due to memory fragmentation (see "Memory Fragmentation, below)

The defaults for `max-num-seqs` and `max-num-batched-tokens` are quite large by the standard of a small VRAM system and should be reduced in such situations. `--max-num-seqs 4` and `--max-num-batched-tokens 512` may be a reasonable place to start with a small system that expects limited concurrency.

The default for `max-model-len` is the maximum the model allows, and the system will OOM at startup if there is not sufficient VRAM for this, which will often be the case with smaller VRAM systems. A safer (but less consistent) option is `--max-model-len auto`, which will size the max context to be equal to available VRAM. It is an especially good idea to use this when first trying a new model, before you have an idea how big things can really be.

Regardless of the setting of `max-model-len`, vLLM will allocate as much VRAM as is possible within its `gpu-memory-utilization` budget, and any leftover VRAM is allocated to the KV cache -- and any excess beyond the model context limit is simply used for the processing of concurrent requests.

This all means there are a number of different ways to experience an out-of-memory condition with vLLM, each with a different solution:

- gpu-memory-utilization exceeds the actual available VRAM: an OOM crash very early, and reducing this parameter is the solution.
- gpu-memory-utilization is achievable, but doesn't leave enough room for allocations that happen outside of vLLM/pytorch's accounting. Again an OOM crash, but potentially later in startup. Reducing this parameter is the solution. The vLLM fork used by this project is less prone to this, but it is still possible.
- Model weights (minus any configured CPU offload) are too large to fit in VRAM. This will be reported directly by vLLM. Either more offload, choosing a smaller quant, or raising gpu-memory-utilization is the solution.
- Insufficient space for the configured max-model-len KV cache. This will be reported directly by vLLM. Reducing max-model-len (or setting to auto), enabling kv cache quantization, or reducing the size of model weights (as above) is the solution
- vLLM starts up, serves for a while, then OOMs later on a request. Fragmentation or allocation transients are the likely cause: the former may be helped by enabling expandable segments.

## Model Evaluation

Before we get into model recommendations, a word on the current state of model evaluations. When picking a model, one naturally wants solid information about how it performs on certain kinds of tasks relative to other candidates. The best thing one could want is a series of standardized benchmarks designed to produce statistically valid results that operate the model in mock simulations of the tasks under evaluation. There are *many* such benchmarks, but there are also many serious problems:

- Evaluations can be costly to run, and often the only available evaluations are from the model's producer.
- Unless the evaluation takes special measures to defend against it, models released after the benchmarks have been published may have included some or all of the benchmark in its training, contaminating the result ("benchmaxxing")
- Because of this fact, and the fact that model capability continues to rapidly improve, new benchmarks are constantly being created.
- Benchmarks have many more parameters than just the model under test and the benchmark name: the harness used, the exact version of the evaluation, the exact configuration of the model as served (sampling parameters, KV setup, etc.), the inference stack, etc. Almost no reported eval includes 100% of this information, and many include essentially none.
- Coming up with a problem set that thoroughly exercises the models and can also be fairly judged is difficult, and it's difficult to guarantee that the judgement criteria will always correlate with the best outcome.

What this all adds up to is that there is *incredibly* thin statistically valid cross-model data available: what is presented on a model card is some subset of benchmarks that often barely overlaps with scores presented on other cards even roughly, and basically not at all when considering the variations possible even within each benchmark. Many cards do include select competitor scores alongside theirs, and that is probably the closest we get: but there we have to trust that the model's creator treated their competitors fairly, and there are a lot of reasons why that could be a source of significant skew (and not all of them require actual intent to mislead)

In the lack of valid concrete data, there is a tendency for users to fall back to individual anecdotal experience: what it "feels" like, how successful outcomes were in some (from a statistical perspective) vanishingly small set of tasks that any one person tends to run, how frustrating it is watching the sometimes-meandering reasoning streams, etc. All of which is both valid in its own way, and completely statistically invalid.

In terms of evaluating quants of models, the situation is in some ways a little better. The situation with capability testing is not any better, but there is a different question that is much easier to pin down: how much does this quant statistically diverge from what the original, undamaged model would have produced?

There are many metrics for this: in my opinion the most generally useful is "Kullback–Leibler divergence", which on the quants I produce is the main basis for most of the results presented. Like capability evaluations, this is more complicated than just one thing; you still need to have some content on which to measure what the answers diverge *from*, so KLD metrics presented on model cards from different producers are not comparable. The measurements on my quants use a mix of captured real-world content for the scoring, including content from longer agentic sessions, so that kind of work is well-represented rather than just generic text or conversations.

This allows one to solidly answer the question of "how different is this quant from the original", but nothing more: there is no statement about to what degree or in what way this impacts capability or outcome. It is reasonable to guess that a more divergent quant would be less desirable than a less divergent one; more than this cannot be said.

On the quants I produce, I've added a different kind of divergence metric, which is slightly more concrete than this. It's still a divergence measurement, but it's scored based on a set of meaningful factual questions that the original model correctly answered, and the divergence then becomes the degree to which a model has forgotten facts it once knew.

There's still no direct connection to capability: it cannot be said that a model that was judged to have answered wrongly where the original model answered correctly would reach any worse outcome on a real task. But it is at least reasonable to assume that this kind of damage to previously known facts would not be helpful, and the presentation of statistics such as "forgot 82% of previously known facts" or "reported incorrect answers confidently at a 5% rate" is intuitively meaningful in way that other divergence metrics are not.

As most things with quantization, the amount of damage increases dramatically past a certain size


## Model Recommendations By Available VRAM

Here's a breakdown of general model recommendations at different levels of available VRAM. Not considered are much older models, or the more sketchy looking finetunes/abliterations/distillations.

One prominent gap in coverage in this list is nvidia's sprawling Nemotron family, some of which is EXL3 supported. I just personally haven't used them at all, and there may be appealing options there.

### 4-6GiB

This size category is extra-tricky: not just because it's very small, but because most cards in this category are too old of an architecture to support any of vLLM's KV cache quantizations: no fp8, no TurboQuant, no NVFP4.

This means the name of the game becomes KV management more than size of the model weights. As always, different model architectures have different levels of efficiency in KV memory requirements, and in this class, the Qwen3.5 family models are the king. Not only do they have the highest maximum context in the class (256K), they also have the lowest KiB/token requirement (12KiB/t and 32KiB/t respectively). Additionally, Qwen3.5 in this range generally have among the highest capability benchmark scores. Finally, the models are both vision and MTP enabled, so both of those features are available if desired and space allows.

So it ends up a race between `Qwen3.5-2B` and `Qwen3.5-4B`: the former is obviously a less capable model, but with a more efficient KV, so if long context is an absolute requirement this may still be the better pick. My advice would be that a 2B model (or really even the 4B model) is unlikely to be able to really take advantage of a much longer context, and the VRAM would very likely be better spent on the 4B model with a correspondingly reduced context size.

The main other contender in this space is `MiniCPM5-2B`, which by model size is competitive with Qwen3.5 on benchmarks, but lacks in the other ways Qwen is strong: a maximum 128K context, no vision, no MTP, and most importantly a considerably less efficient KV (at 42KiB/t). For use cases where longer context or vision is not a requirement, it could be worth considering.

LiquidAI's `LFM3-1.7B` (or its vision-enabled variant `LFM3-VL-2B`) is competitive as well: good KV efficiency (22KiB/t) and impressive looking (though thin) published benchmarks. There are currently no EXL3 quants available, however. Their prior LFM2.5 series are generally not as competitive, although the tiny MoE `LFM2.5-8B-A1B` deserves mention for being a much smaller MoE than any other model listed, with a KV efficiency that ties the best alternative (`Qwen3.5-2B`). With 1B active parameters, there are real questions about capability, but some benchmark scores look acceptable.


### 8GiB

There's really currently only two reasonable contenders in this space: Gemma 4's MoE `gemma-4-26B-A4B`, and `Qwen3.5-9B` (or finetunes such as `Ornith-1.5-9B`). Gemma generally has different strengths than Qwen, so the recommendation is more about use case than anything. For conversational tasks, many people find gemma a more pleasant model to work with. For agentic work, Qwen does score much better on benchmarks; it is worth nothing however, that agentic work at the 9B model class may be fairly disappointing by modern standards.


### 12GiB

The 12GiB cards open up considerably in terms of options. `Qwen3.6-35B-A3B` is probably the best choice for agentic work: a very efficient KV, vision and MTP support, the extra speed that MoEs can bring, and solid scores on benchmarks. Finetunes of 35B Qwen MoEs are available as well (i.e. `Ornith-1.5-35B-A3B`).

As at the 8GiB tier, for conversational work Gemma 4's 12B dense model `gemma-4-12B-it` is worth consideration. It has no vision, but otherwise offers similar capabilities as Qwen. As a dense model, it will likely run more slowly.

Two other MoEs deserve mention: poolside's `Laguna-XS-2.1` and zai-org's `GLM-4.7-Flash`. Neither model support vision, and both have a less efficient KV than Qwen or Gemma, but they have different strengths and could better fit certain use cases.


### 16GiB

The main model that opens up here is `Qwen3.8-27B`, although it is tight: KV costs more, and the model itself is much bigger in resident VRAM. However, the capability scores are potentially very worth it: it is a very strong model, especially for coding tasks. Likely it will be necessary to bump the body bitrate below 4bpw, which will affect quality; the main question is whether this degradation of a stronger model will be a fair trade as compared to a higher bitrate `Qwen3.6-35B-A3B`.

The wildcard option is `Qwen3.8-Flash-Next`: a much larger model that doesn't seem like it would plausibly fit, but the architecture means it can, barely -- with a combination of disk offload and host cpu offload. It does require a large amount of host RAM available, and generation speed will naturally be lower. But the KV efficiency is shockingly good, and its capability scores are in a different class entirely from anything else that can be run. In order to fit on a 16GiB card, it is necessary to use a 2bpw or lower quant, and there is the very real question of how much damage that represents to the model, and whether it is better to use a less damaged smaller model on the balance. There isn't solid data on that, but there is reason to think that larger models can be considerably more resilient to aggressive quantization than smaller ones, so it's a plausibly good trade.


### 24GiB+

`Qwen3.8-Flash-Next` becomes progressively more of a no-brainer at higher VRAM systems, as it becomes possible to run at bitrates that are more classically acceptable. At 32GiB, 4bpw is possible, and there this model becomes the obvious choice.

`Nanbeige4.3-33B` looks at first blush like an interesting model: benchmark scores are very high (almost too high to believe), but the benchmark validity question becomes irrelevant once you look at the KV efficiency: this model is essentially impossible to run at long context on cards of this VRAM class.

`gemma-4-31B-it` is possible, and would be a decent pick in all the same situations that Gemma 4 at other VRAM classes would be. It's got a more expensive KV, but not to a disqualifying degree like Nanbeige, particularly given Gemma's strength lies in directions that do not demand the longest contexts.

Meta's `Muse-Glimmer-30B` is another prominent option here, and its main claim to fame is an incredibly efficient KV architecture: at 15KiB/token, it is possible to run a quantized KV cache on this model at the maximum context in 0.5GiB of VRAM. The caveat is that the maximum context is lower than any nearby competitor, at 128K. But this is plenty of context for many kinds of work, and a larger KV than maximum context supports greater concurrent use, so this is still a potentially significant strength for certain kinds of use.


## Recommended Dense Models

A variety of recent dense models in the 9-30B range are available in EXL3 format:

### [Qwen/Qwen3.5-9B](https://hf.co/Qwen/Qwen3.5-9B)

A small, but still usefully sized entry in the Qwen lineup. 9B parameters, with vision.
Reasoning is supported via the `qwen3` reasoning parser, and tool calling via the `qwen3_coder` parser. There are no default sampling parameters defined in the model, so it is either necessary to set these at every client, or override these to the correct defaults via (for example) `--override-generation-config '{ "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0 }'` -- note that there are *four* different sampling parameter configurations recommended on the model card for different scenarios, so make sure you are using the right set.

Available EXL3 quants include [turboderp/Qwen3.5-9B-exl3](https://hf.co/turboderp/Qwen3.5-9B-exl3)

Example:

```
vllm serve turboderp/Qwen3.5-9B-exl3 --revision 4.00bpw \
    --max-model-len auto --max-num-seqs 4 --max-num-batched-tokens 512 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.95 \
    --override-generation-config '{ "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0 }' \
    --cpu-offload-gb 1 --cpu-offload-params visual --mm-processor-kwargs '{"max_pixels": 1048576}' 
```

### [ornith-ai/Ornith-1.5-9B](https://hf.co/ornith-ai/Ornith-1.5-9B)

A Qwen finetune oriented toward agentic work. Shares the reasoning and tool calling parameters with Qwen, as well as the requirement for setting sampling parameters.

Available EXL3 quants include [yeasah/Ornith-1.5-9B-exl3](https://hf.co/yeasah/Ornith-1.5-9B-exl3)

Example:

```
vllm serve yeasah/Ornith-1.5-9B-exl3 --revision 4.00bpw \
    --max-model-len auto --max-num-seqs 4 --max-num-batched-tokens 512 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.95 \
    --override-generation-config '{ "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0 }' \
    --cpu-offload-gb 1 --cpu-offload-params visual --mm-processor-kwargs '{"max_pixels": 1048576}' 
```

### [google/gemma-4-12B-it](https://hf.co/google/gemma-4-12B-it)

Gemma 4 may not be the strongest at coding, but it is quite strong as a conversational model. Reasoning and tool calling are supported via `gemma4` parsers. 

Gemma 4 12B is particularly unusual in that it has a "unified" multimodal architecture, meaning it does not have a separate vision or audio encoder, and so it is not possible to offload vision in any meaningful sense. `--language-model-only` is still useful if no vision is required, but it does not save a substantial amount of VRAM. It is also the only model on this list that can accept audio.

Gemma 4 has some unusual characteristics that make serving a little trickier:

 - Gemma 4 has varying head dimensions across its layers. Except on recent datacenter-class cards, this will result in the following message at startup `Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.`, and the forced use of triton attention will preclude or complicate the use of KV cache quantization other than `fp8`. TurboQuant is possible in some configurations, with considerable setup.
 - When vision is enabled, `max-num-batched-tokens` must be at least 2496, and if you try to run it with a lower value, the following error will be raised at startup: `Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens (512). Please increase max_num_batched_tokens.`
 - When vision is enabled, a load-time OOM may happen if expandable segments are not enabled (which can be done by setting the `PYTORCH_CUDA_ALLOC_CONF` environment variable to the value `expandable_segments:True`)

Available EXL3 quants include [turboderp/gemma-4-12B-it-exl3](https://hf.co/turboderp/gemma-4-12B-it-exl3)

Example:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    vllm serve turboderp/gemma-4-12B-it-exl3 --revision 4.00bpw_mul1 \
    --max-model-len auto --max-num-seqs 4 --max-num-batched-tokens 2496 \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.90
```


### [Qwen/Qwen3.8-27B](https://hf.co/Qwen/Qwen3.8-27B)

One of the current darlings of the community, a strong coder. Reasoning is supported via the `qwen3` parser, and tool calling is supported by the `qwen3_coder` parser. The sampling parameter situation is simpler than prior entries in the Qwen lineup, with only two configs recommended, one for thinking and one for non-thinking. Normally one would want to use this model in thinking mode, which is the default. However, note that this model supports different thinking effort levels, and the chat template's default is `xhigh`, which many find excessive (I certainly do, and I would recommend `medium` instead)

Available EXL3 quants include [yeasah/Qwen3.8-27B-exl3](https://hf.co/yeasah/Qwen3.8-27B-exl3) and [turboderp/Qwen3.8-27B-exl3](https://hf.co/turboderp/Qwen3.8-27B-exl3)

Example:

```
vllm serve yeasah/Qwen3.8-27B-exl3 --revision 3.50bpw-H5 \
    --max-model-len auto --max-num-seqs 4 --max-num-batched-tokens 512 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.95 \
    --cpu-offload-gb 2 --cpu-offload-params visual --mm-processor-kwargs '{"max_pixels": 1048576}
```


### [meta-models/Muse-Glimmer-30B](https://hf.co/meta-models/Muse-Glimmer-30B)

meta's 30B dense model. Extremely efficient KV, but ironically only trained for 128K of it. It does, however, allow for much greater KV concurrency in vLLM. Reasoning and tool calling are supported by the `muse_glimmer` parsers.

Available EXL3 quants include [yeasah/Muse-Glimmer-30B-exl3](https://hf.co/yeasah/Muse-Glimmer-30B-exl3) and [turboderp/Muse-Glimmer-30B-exl3](https://hf.co/turboderp/Muse-Glimmer-30B-exl3)

Note that turboderp uses a quantized vision tower, which makes offload more difficult.

Example:

```
vllm serve yeasah/Muse-Glimmer-30B-exl3 --revision 3.00bpw-H5 \
    --max-model-len auto --max-num-seqs 4 --max-num-batched-tokens 512 \
    --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
    --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.95 \
    --cpu-offload-gb 12 --cpu-offload-params visual
```

### [google/gemma-4-31B-it](https://hf.co/google/gemma-4-31B-it)

Gemma 4 12B, but more of it, and with a more traditional multimodal setup (with a separate vision encoder, and no audio)

Available EXL3 quants include [turboderp/gemma-4-31B-it-exl3](https://hf.co/turboderp/gemma-4-31B-it-exl3)


## Recommended MoE Models

Mixture-of-experts models in this range tend to be a little larger than the dense lineup in terms of total parameters (26B-35B), but they have two powerful advantages. First, only a small number of parameters are used at any given time, so generation tends to be faster than a corresponding dense model. And second, the same property means that a significant amount of the model can be offloaded to host RAM and paged in on demand without seriously impacting performance, so often they have a smaller effective VRAM footprint as well.

The MoE expert cache that allows this high-performance offload is simple to enable: configure a certain amount of memory for offload in the normal vLLM way (`--cpu-offload-gb N`, where N is the number of GiB you wish to allocate for this), and optionally explicitly specifying that the experts are the target via `--cpu-offload-params experts`. (If vision is being used, it should be included in the cpu offload list as well: vision offload is nearly free from a performance perspective and should be prioritized over experts if offload is to be used)

### [google/gemma-4-26B-A4B-it](https://hf.co/google/gemma-4-26B-A4B-it)

Gemma, MoE style, with a more traditional multimodal setup (with a separate vision encoder, and no audio). One of the smallest MoEs available at only 26B parameters. 

Available EXL3 quants include [turboderp/gemma-4-26B-A4B-it-exl3](https://hf.co/turboderp/gemma-4-26B-A4B-it-exl3)


### [Qwen/Qwen3.6-35B-A3B](https://hf.co/Qwen/Qwen3.6-35B-A3B)

The most recent Qwen MoE available. Reasoning is supported via the `qwen3` reasoning parser, and tool calling via the `qwen3_coder` parser. The model card calls out different sampling parameters for different use cases, so make sure you are using the right settings.

Available EXL3 quants include [yeasah/Qwen3.6-35B-A3B-exl3](https://hf.co/yeasah/Qwen3.6-35B-A3B-exl3)


### [ornith-ai/Ornith-1.5-35B-A3B](https://hf.co/ornith-ai/Ornith-1.5-35B-A3B)

The ornith counterpart to the Qwen MoE. Same settings and caveats as the Qwen MoE on which it is based.

Available EXL3 quants include [yeasah/Ornith-1.5-35B-A3B-exl3](https://hf.co/yeasah/Ornith-1.5-35B-A3B-exl3)


### [poolside/Laguna-XS-2.1](https://hf.co/poolside/Laguna-XS-2.1)

poolside's agentic-focused MoE. This is a 33B parameter model. Reasoning and tool calling are supported via the `poolside_v1` parsers.

Available EXL3 quants include [turboderp/Laguna-XS-2.1-exl3](https://hf.co/turboderp/Laguna-XS-2.1-exl3)

## Other models

Many other models are available, but in general a model must be supported both by exllamav3 (so that an EXL3 quantization may be produced in the first place), and vLLM (so that it may be served)


## KV cache quantization

Fitting a larger model into a smaller amount of VRAM is fine, but there needs to be room for a KV cache as well. Particularly for agentic loops, larger (128K+) contexts are important or even necessary. To this end, KV cache quantization can help. vLLM offers many options (via the `--kv-cache-dtype` option) for quantized KV caches, including:

 - `fp8`: The standard for anyone looking to save a bit of space without much loss, half the size. Broadly compatible when supported by the hardware.
 - TurboQuant: A much more advanced quantization scheme, with several different tiers of options. In my experience usually the best option is `turboquant_4bit_nc`, which has very good quality by most metrics and is almost one quarter the size. Currently turboquant can experience lower performance at longer contexts, and it is unknown if this is an inherent vLLM limitation.
 - `nvfp4`: Particularly performant on Blackwell cards, but it does run on others. I haven't explored the quality question as much. Performance holds up well at higher context lengths.

 Not every vLLM attention backend will allow every quantization scheme, and the relationship between model architectures, host hardware, and compatible attention backends is complicated.


## VRAM Fragmentation

Some configurations have more active VRAM allocations occur during use than others, and some patterns of allocation can lead to memory fragmentation -- a condition where there is still free VRAM available, but only in pieces smaller than what is needed, and an OOM can happen even with substantial "free" memory.

Pytorch provides an option to enable "expandable segments", which reduces the rate of fragmentation. When needed, enable this with the following environment variable:

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`


## Other knobs

vLLM offers a performance tuning setting `--performance-mode`, which allows hinting of the expected usage patterns for it to optimize for: `interactivity` favors lower latency, `throughput` favors higher overall rate across multiple sessions, and `balanced` (the default) attempts to find a good balance between the two. This setting may or may not be appropriate or helpful for low-VRAM use cases, and I have not investigated in detail.

Prefix caching is of fundamental importance: without this, a session of any length will have its prefill latency increase dramatically as the session grows. This is enabled by default, and should in almost every case be left enabled.

It is possible to extend context beyond the normal trained max model length on some models, which usually provide instructions on their model card. I do not generally recommend this -- it consumes a lot of VRAM, and model performance with context substantially greater than the training size is generally quite uneven.


## Speculative (draft) models

Speculative models are smaller models which run ahead of the main model, predicting output tokens in the hopes that many of them will match what the larger model would have produced. Acceptance rates can be relatively high and generation speed can be subtantially improved this way; the cost is a small additional model that is running in parallel, which requires additional VRAM. In some situations it could make sense even in highly VRAM-constrained environments -- for example in conjunction with expert offload on a MoE model, which frees a significant amount of VRAM at a relatively smaller performance cost; possibly much lower than the spec model's benefit.

Some models (Qwen family) ship with an embedded MTP draft model, which can be enabled without referencing or downloading any external model. Other models ship separate draft models, and for some of them there are community dflash/dflash2 draft models available as well that may further improve performance.

External draft models are currently not well supported with this plugin, but built-in draft heads work as you would expect. Generally instructions for use with vLLM can be found on model cards. For example, Qwen3.8-27B recommends the following:

`--speculative-config '{"method":"mtp","num_speculative_tokens":3,"revision":"3.00bpw"}'`

The main change that is required is to include the `revision` parameter, which should match the revision of the model being served.

Certain draft model configurations do not work at all, sometimes spectacularly: for example, MTP + TurboQuant, documented in [docs/turboquant-kv.md](docs/turboquant-kv.md)


## Multi-GPU operation

This plugin does support tensor parallelism across multiple GPUs, and generally speaking this is as simple as including the `--tensor-parallel-size N` option, where N is the desired number of GPUs to use. There are currently some model-specific limits as to which configurations can work, and the script `tools/tp_preflight.py` will check a model for support of TP configurations without having to first download the model itself, so planning can happen efficiently.


## A note on quant selection

Selecting a quant for a particular model involves a balancing act, starting with knowing your VRAM budget. A typical process would look like:

 - Know how much free VRAM your system has available (in total if there are multiple GPUs present) -- if it is shared with a desktop or other things, this can make a difference, and in general there is usually a small amount of VRAM that is allocated to the host system and cannot be used.
 - Know what your target context size is for the work you wish to do, and determine how much VRAM this will require: this is a model-dependent calculation, and also depends on any KV quantization that will be used. Online KV space calculators exist for this purpose and are quite helpful, although many do not cover the more aggressive KV quantization schemes. As an approximate reference, `turboquant-4bit-nc` is about 3.4x smaller than an unquantized (BF16) KV cache, so dividing the unquantized size by 3.4 will give about the right value for that option. `nvfp4` is similar.
 - Subtracting the KV space budget from the total VRAM available gives you your VRAM budget for the model itself, and armed with this information you can then visit the model page of the quant you wish to use. All of the quant model cards listed above give at least some idea of a quality vs. size comparison, though the amount of information presented varies widely. The goal is to select the highest bits per weight for which the model body fits into the VRAM budget.
 - In the case of MoE models, you may decide to consider bitrates that exceed the VRAM budget by a significant amount if you choose to use CPU expert offload. With the MoE expert cache, it is generally possible to offload 30% or more of the model's weights with a performance cost that may be relatively small (although this depends on the host system -- PCIe bus speed, as well as the model's architecture). The performance cost will increase at higher offload rates, but the degradation is generally gradual.
 - Once you have selected a specific quant and revision, put it all together into a `vllm serve` configuration, and hopefully it serves as expected. Using `--max-model-len auto` is a good idea at first run, even if a fixed context size is needed; if a calculation was off, this will give a direct indication of how much, by checking the actual KV space available once it loads.
 