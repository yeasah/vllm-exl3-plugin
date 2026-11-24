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

In vLLM v0.34, the meaning of `gpu-memory-utilization` has changed, with the introduction of the extensible KV cache. Previously it was necessary to adjust this to carefully fit a system to available VRAM; now it is recommended to leave this at the default (which is now 1.0), and the system will autofit itself to available space. It is only when the system should not use all of the available VRAM that this setting is now required: to leave half of the card unutilized for example, set it to 0.5.

The defaults for `max-num-seqs` is quite large by the standard of a small VRAM system and should be reduced in such situations. `--max-num-seqs 4` may be a reasonable place to start with a small system that expects limited concurrency.

The default for `max-num-batched-tokens` here is 2048, and this is a good starting place in terms of VRAM vs. prefill performance. It is possible to save a fair bit of VRAM by reducing this further, though past 512 the amount of VRAM saved versus the degree of prefill slowdown becomes less favorable.

The default for `max-model-len` is the maximum the model allows, and the system will OOM at startup if there is not sufficient VRAM for this, which will often be the case with smaller VRAM systems. A safer (but less consistent) option is `--max-model-len auto`, which will size the max context to be equal to available VRAM. It is an especially good idea to use this when first trying a new model, before you have an idea how big things can really be.


## Model Evaluation

Before we get into model recommendations, a word on the current state of model evaluations. When picking a model, one naturally wants solid information about how it performs on certain kinds of tasks relative to other candidates. The best thing one could want is a series of standardized benchmarks designed to produce statistically valid results that operate the model in mock simulations of the tasks under evaluation. There are *many* such benchmarks, but there are also many serious problems:

- Evaluations can be costly to run, and often the only available evaluations are from the model's producer.
- Unless the evaluation takes special measures to defend against it, models released after the benchmarks have been published may have included some or all of the benchmark in its training, contaminating the result ("benchmaxxing")
- Because of this fact, and the fact that model capability continues to rapidly improve, new benchmarks are constantly being created.
- Benchmarks have many more parameters than just the model under test and the benchmark name: the harness used, the exact version of the evaluation, the exact configuration of the model as served (sampling parameters, KV setup, etc.), the inference stack, etc. Almost no reported eval includes 100% of this information, and many include essentially none.
- Coming up with a problem set that thoroughly exercises the models and can also be fairly judged is difficult, and it's difficult to guarantee that the judgement criteria will always correlate with the best outcome.

What this all adds up to is that there is *incredibly* thin statistically valid cross-model data available: what is presented on a model card is some subset of benchmarks that often barely overlaps with scores presented on other cards even roughly, and basically not at all when considering the variations possible even within each benchmark. Many cards do include select competitor scores alongside theirs, and that is probably the closest we get: but there we have to trust that the model's creator treated their competitors fairly, and there are a lot of reasons why that could be a source of significant skew (and not all of them require actual intent to mislead)

In the lack of valid concrete data, there is a tendency for users to fall back to individual anecdotal experience: what it "feels" like, how successful outcomes were in some (from a statistical perspective) vanishingly small set of tasks that any one person tends to run, how frustrating it is watching the sometimes-meandering reasoning streams, etc. All of which is both valid in its own way, and completely statistically invalid.


## Quanitzed Model Evaluation

In terms of evaluating quants of models, the situation is in some ways a little better. The situation with capability testing is not any better, but there is a different question that is much easier to pin down: how much does this quant statistically diverge from what the original, undamaged model would have produced?

There are many metrics for this: in my opinion the most generally useful is "Kullback–Leibler divergence", which on the quants I produce is the main basis for most of the results presented. Like capability evaluations, this is more complicated than just one thing; you still need to have some content on which to measure what the answers diverge *from*, so KLD metrics presented on model cards from different producers are not comparable. The measurements on my quants use a mix of captured real-world content for the scoring, including content from longer agentic sessions, so that kind of work is well-represented rather than just generic text or conversations.

This allows one to solidly answer the question of "how different is this quant from the original", but nothing more: there is no statement about to what degree or in what way this impacts capability or outcome. It is reasonable to guess that a more divergent quant would be less desirable than a less divergent one; more than this cannot be said.

On the quants I produce, I've added a different kind of divergence metric, which is slightly more concrete than this. It's still a divergence measurement, but it's scored based on a set of meaningful factual questions that the original model correctly answered, and the divergence then becomes the degree to which a model has forgotten facts it once knew.

There's still no direct connection to capability: it cannot be said that a model that was judged to have answered wrongly where the original model answered correctly would reach any worse outcome on a real task. But it is at least reasonable to assume that this kind of damage to previously known facts would not be helpful, and the presentation of statistics such as "forgot 82% of previously known facts" or "reported incorrect answers confidently at a 5% rate" is intuitively meaningful in way that other divergence metrics are not.

As with most things with quantization, the amount of damage increases dramatically past a certain size, and there is a point at which the model is too damaged to be useful/reliable/trustworthy enough. There is no rule about this, and can't be: it depends too much on a user's tolerance and workload. Generally I end the quant ladders before it crosses below an 80% recall rate, which is usually around 2.5-3 bits per weight. This is a completely arbitrary line, but it needs to be drawn somewhere.


## Model Recommendations By Available VRAM

Here's a breakdown of general model recommendations at different levels of available VRAM. Not considered are much older models, or the more sketchy looking finetunes/abliterations/distillations.

One prominent gap in coverage in this list is nvidia's sprawling Nemotron family, some of which is EXL3 supported. I just personally haven't used them at all, and there may be appealing options there.

### 4-6GiB

This size category is extra-tricky: not just because it's very small, but because most cards in this category are too old of an architecture to support any of vLLM's KV cache quantizations: no fp8, no TurboQuant, no NVFP4.

This means the name of the game becomes KV management more than size of the model weights. As always, different model architectures have different levels of efficiency in KV memory requirements, and in this class, the Qwen3.5 family models are the king. Not only do they have the highest maximum context in the class (256K), they also have the lowest KiB/token requirement (12KiB/t and 32KiB/t respectively). Additionally, Qwen3.5 in this range generally have among the highest capability benchmark scores. Finally, the models are both vision and MTP enabled, so both of those features are available if desired and space allows.

So it ends up a race between [Qwen3.5-2B](docs/recipes/Qwen3.5-2B.md) and [Qwen3.5-4B](docs/recipes/Qwen3.5-4B.md): the former is obviously a less capable model, but with a more efficient KV, so if long context is an absolute requirement this may still be the better pick. My advice would be that a 2B model (or really even the 4B model) is unlikely to be able to really take advantage of a much longer context, and the VRAM would very likely be better spent on the 4B model with a correspondingly reduced context size.

The main other contender in this space is [MiniCPM5-2B](docs/recipes/MiniCPM5-2B.md), which by model size is competitive with Qwen3.5 on benchmarks, but lacks in the other ways Qwen is strong: a maximum 128K context, no vision, no MTP, and most importantly a considerably less efficient KV. For use cases where longer context or vision is not a requirement, it could be worth considering.

LiquidAI's `LFM3-1.7B` (or its vision-enabled variant `LFM3-VL-2B`) is competitive as well: good KV efficiency (22KiB/t) and impressive looking (though thin) published benchmarks. There are currently no EXL3 quants available, however. Their prior LFM2.5 series are generally not as competitive, although the tiny MoE `LFM2.5-8B-A1B` deserves mention for being a much smaller MoE than any other model listed, with a KV efficiency that ties the best alternative (`Qwen3.5-2B`). With 1B active parameters, there are real questions about capability, but some benchmark scores look acceptable.

XHToken's `Spark-X2.6-4B` is another recent impressive looking model. At 38KiB/tok, its KV efficiency is worse than other options in this class. There is no vision option.


### 8GiB

There's really currently only two reasonable contenders in this space: Gemma 4's MoE [gemma-4-26B-A4B](docs/recipes/gemma-4-26B-A4B.md), and [Qwen3.5-9B](docs/recipes/Qwen3.5-9B.md) (or finetunes such as [Ornith-1.5-9B](docs/recipes/Ornith-1.5-9B)). Gemma generally has different strengths than Qwen, so the recommendation is more about use case than anything. For conversational tasks, many people find gemma a more pleasant model to work with. For agentic work, Qwen does score much better on benchmarks; it is worth nothing however, that agentic work at the 9B model class may be fairly disappointing by modern standards.


### 12GiB

The 12GiB cards open up considerably in terms of options. [Qwen3.6-35B-A3B](docs/recipes/Qwen3.6-35B-A3B.md) is probably the best choice for agentic work: a very efficient KV, vision and MTP support, the extra speed that MoEs can bring, and solid scores on benchmarks. Finetunes of 35B Qwen MoEs are available as well (i.e. [Ornith-1.5-35B-A3B](docs/recipes/Ornith-1.5-35B-A3B.md)).

As at the 8GiB tier, for conversational work Gemma 4's 12B dense model [gemma-4-12B-it](docs/recipes/gemma-4-12B-it.md) is worth consideration. It has no vision, but otherwise offers similar capabilities as Qwen. As a dense model, it will likely run more slowly.

Two other MoEs deserve mention: poolside's [Laguna-XS-2.1](docs/recipes/Laguna-XS-2.1.md) and zai-org's `GLM-4.7-Flash`. Neither model support vision, and both have a less efficient KV than Qwen or Gemma, but they have different strengths and could better fit certain use cases.


### 16GiB

The main model that opens up here is [Qwen3.8-27B](docs/recipes/Qwen3.8-27B.md), although it is tight: KV costs more, and the model itself is much bigger in resident VRAM. However, the capability scores are potentially very worth it: it is a very strong model, especially for coding tasks. Likely it will be necessary to bump the body bitrate below 4bpw, which will affect quality; the main question is whether this degradation of a stronger model will be a fair trade as compared to a higher bitrate [Qwen3.6-35B-A3B](docs/recipes/Qwen3.6-35B-A3B.md).

The wildcard option is [Qwen3.8-Flash-Next](docs/recipes/Qwen3.8-Flash-Next.md): a much larger model that doesn't seem like it would plausibly fit, but the architecture means it can, with a combination of disk offload and host cpu offload. It does require a large amount of host RAM available. But the KV efficiency is shockingly good, and its capability scores are in a different class entirely from anything else that can be run. In order to fit on a 16GiB card, it is necessary to use a 2bpw or lower quant, and there is the very real question of how much damage that represents to the model, and whether it is better to use a less damaged smaller model on the balance. There isn't solid data on that, but there is reason to think that larger models can be considerably more resilient to aggressive quantization than smaller ones, so it's a plausibly good trade.


### 24GiB+

[Qwen3.8-Flash-Next](docs/recipes/Qwen3.8-Flash-Next.md) becomes progressively more of a no-brainer at higher VRAM systems, as it becomes possible to run at bitrates that are more classically acceptable. At 32GiB, 4bpw is possible, and there this model becomes the obvious choice.

`Nanbeige4.3-33B` looks at first blush like an interesting model: benchmark scores are very high (almost too high to believe), but the benchmark validity question becomes irrelevant once you look at the KV efficiency: this model is essentially impossible to run at long context on cards of this VRAM class.

[gemma-4-31B-it](docs/recipes/gemma-4-31B-it.md) is possible, and would be a decent pick in all the same situations that Gemma 4 at other VRAM classes would be. It's got a more expensive KV, but not to a disqualifying degree like Nanbeige, particularly given Gemma's strength lies in directions that do not demand the longest contexts.

Meta's [Muse-Glimmer-30B](docs/recipes/Muse-Glimmer-30B.md) is another prominent option here, and its main claim to fame is an incredibly efficient KV architecture: at 15KiB/token, it is possible to run a quantized KV cache on this model at the maximum context in 0.5GiB of VRAM. The caveat is that the maximum context is lower than any nearby competitor, at 128K. But this is plenty of context for many kinds of work, and a larger KV than maximum context supports greater concurrent use, so this is still a potentially significant strength for certain kinds of use.


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

vLLM offers a performance tuning setting `--performance-mode`, which allows hinting of the expected usage patterns for it to optimize for: `interactivity` favors lower latency, `throughput` favors higher overall rate across multiple sessions, and `balanced` (the default) attempts to find a good balance between the two. Testing on one particular configuration (serving Qwen3.6-35B-A3B on an RTX 5070ti) shows the following results:

|Mode|Concurrency|Prefill|Decode|Prefill vs. default|Decode vs. default|
|---|---|---|---|---|---|
|balanced|1|5098.35|101.63|||
|balanced|4|15521.81|173.26|||
|interactivity|1|5094.41|100.64|0.99x|0.99x|
|interactivity|4|15583.35|174.48|1.00x|1.01x|
|throughput|1|5784.18|99.91|1.13x|0.98x|
|throughput|4|17067.72|173.52|1.09x|1.00x|

Which is to say, on this configuration, `throughput` gives a significant boost to prefill at a minimal cost to decode, and the setting is otherwise a wash.

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


## Host RAM and disk offload

vLLM supports CPU offload of weights, and this plugin extends this mechanism to cover EXL3 quantized tensors as well. The vLLM mechanism to control this is the `cpu-offload-gb` parameter, which instructs vLLM to use up to that number of GiB of host RAM *per rank* (so on a 2 card system with `tensor-parallel 2`, the total RAM pinned and allocated will be double that number).

Normally, if vision is configured on a model and offload is desired at all, the vision encoder should be the first thing to go: performance loss is almost zero there. Telling vLLM which parts to offload is configured with the `cpu-offload-params` option, which can be a list of names. For vision, include `visual` in the list.

The following additions have been made to offload in this plugin:
 - Embeddings are automatically offloaded to host RAM by default
 - Engram tables are automatically streamed from disk via exllamav3's streaming mechanism.
 - MoE expert offload is handled very differently from vLLM's normal behavior. This is enabled by including `experts` in the `cpu-offload-params` list. See the next section for more details on how this works.


## MoE expert cache

Simply including experts in the list of offloaded targets will automatically enable the MoE expert cache, which divides expert residency into three layers: available VRAM space on the GPUs, available host RAM, and finally (if required) the disk where the tensors were loaded from. 

When `max-model-len` is `auto`, the cache will attempt to allocate enough VRAM to hit a 40% residency rate, which in benchmarks has held the performance loss at a relatively low level. KV space will be given at least a usable amount (currently 32K tokens). 

When operating, the system will actively promote and demote experts through all three tiers based on their use. The impact on performance at different residency rates at each of the three tiers will be system dependent, but expect significant slowdown with high levels of disk offload.


## A note on quant selection

Selecting a quant for a particular model involves a balancing act, starting with knowing your VRAM budget. A typical process would look like:

 - Know how much free VRAM your system has available (in total if there are multiple GPUs present) -- if it is shared with a desktop or other things, this can make a difference, and in general there is usually a small amount of VRAM that is allocated to the host system and cannot be used.
 - Know what your target context size is for the work you wish to do, and determine how much VRAM this will require: this is a model-dependent calculation, and also depends on any KV quantization that will be used. Online KV space calculators exist for this purpose and are quite helpful, although many do not cover the more aggressive KV quantization schemes. As an approximate reference, `turboquant-4bit-nc` is about 3.4x smaller than an unquantized (BF16) KV cache, so dividing the unquantized size by 3.4 will give about the right value for that option. `nvfp4` is similar.
 - Subtracting the KV space budget from the total VRAM available gives you your VRAM budget for the model itself, and armed with this information you can then visit the model page of the quant you wish to use. All of the quant model cards listed above give at least some idea of a quality vs. size comparison, though the amount of information presented varies widely. The goal is to select the highest bits per weight for which the model body fits into the VRAM budget.
 - In the case of MoE models, you may decide to consider bitrates that exceed the VRAM budget by a significant amount if you choose to use CPU expert offload. With the MoE expert cache, it is generally possible to offload 30% or more of the model's weights with a performance cost that may be relatively small (although this depends on the host system -- PCIe bus speed, as well as the model's architecture). The performance cost will increase at higher offload rates, but the degradation is generally gradual.
 - Once you have selected a specific quant and revision, put it all together into a `vllm serve` configuration, and hopefully it serves as expected. Using `--max-model-len auto` is a good idea at first run, even if a fixed context size is needed; if a calculation was off, this will give a direct indication of how much, by checking the actual KV space available once it loads.
 