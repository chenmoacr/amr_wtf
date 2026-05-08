# GHOST

**G**host-**H**ost **O**utput **S**teering **T**oolkit
A hallucination-driven behavior-vector extraction method for small language models.

---

- [English](#english)
- [中文](#中文)

---

## English

### Abstract

GHOST is a research toolkit that extracts and applies *behavior vectors* from a
small open-weight language model (Gemma 4 E2B-it) by exploiting the model's
inability to introspect its own generation process. By replacing the model's
own answer with a *teacher* answer (Claude Opus 4.6 / a better-written answer /
a hallucination-free answer), aligning the two answer token sequences positionally
where possible (or at least matching their lengths), and computing per-neuron
activation differentials, the toolkit isolates the *functional neurons* that
drive specific output behaviors (depth, structure, citation, frontend design
quality).
These vectors can then be applied via clamp injection, suppression, LoRA
training, or LoRA task-vector ensembling to steer the host model toward the
teacher's behavior at a fraction of normal fine-tuning cost.

The project name *amr_wtf* (the original directory name preserved for
historical accuracy) reflects the project's origin: a passing idea, not a
planned engineering effort. The published codename *GHOST* describes the
mechanism more faithfully.

### 1. Motivation

The principle is simple. Large language models do not know how their outputs
are generated; from the model's perspective, predicted tokens appear out of
thin air. This opens an opportunity: by replacing the model's own answer with a
*teacher* answer (Claude Opus / a higher-quality answer / a hallucination-free
answer) and differentiating per-neuron activations against the baseline, one
can locate the neurons that move the model toward the teacher's preferences.
These neurons are what the literature calls *behavior vectors*.

The project's engineering philosophy distinguishes two neuron classes:

- **Knowledge neurons** store specific facts (the location of the Golden Gate
  Bridge, the boiling point of water).
- **Functional neurons** control structural behaviors (output depth, layering,
  literary comprehension, frontend detail quality).

GHOST successfully induced Gemma 4 E2B-it to "believe it was producing
Opus-style answers" and used this confusion to locate the functional neurons
that move Gemma's output closer to Opus.

Beyond individual neurons, the model contains:

- **Functional circuits** &mdash; causal chains of neurons across layers,
  observable through attribution patching.
- **Large neuron clusters** &mdash; dense regions of functionally similar
  neurons. The existence of such clusters as a distinct structural object is
  treated in this project as an *unverified conjecture*: the toolkit does
  not contain a direct experimental confirmation that they exist as
  first-class entities. However, the *comfort zone* that Gemma 4 E2B-it
  acquires during its instruction-tuning (IT / RLHF) phase is, both
  behaviorally and functionally, very close to what one would expect a
  large functional neuron cluster to look like: an abstract concavity in
  activation space whose attractive force pulls outputs back toward
  instruction-tuned defaults. The two notions &mdash; the engineering
  concept "large functional neuron cluster" and the post-RLHF concept
  "comfort zone" &mdash; can therefore be treated as approximately
  equivalent operational descriptions of the same phenomenon. This
  near-equivalence is *suggested by, but not proven by*, the toolkit's
  observation that progressive suppression of "the strong" causes "the
  weak" to surface in the same layer set rather than in new layers.

### 2. Practical Value

For example: at the time of this project's release, there is a model
(GPT-5.5) that, after RLHF on a particular reward signal, learned to abuse
the token *goblin* to inflate its score, and ended up producing outputs
saturated with that word. GHOST can address such a problem without further
training or system-prompt patching:

1. Prepare two equal-length answers to the same prompt: one containing the
   abused token, one without.
2. Run a QA-differential on neuron activations over the aligned answer span.
3. Locate the functional neuron responsible for the abuse.
4. Suppress it by zeroing the corresponding column in `down_proj`, or merge a
   correcting bias into model weights for zero runtime overhead.

Because the differential method is structurally biased toward functional
neurons &mdash; knowledge neurons rarely show stable cross-context activation
patterns at this signal-to-noise ratio &mdash; the located neuron is, in
practice, the correct intervention point.

The same pipeline addresses:

- Hallucination repair in specific contexts.
- Single-sample preference tuning.
- Style steering without fine-tuning.

The dominant cost is a few minutes of forward-pass inference on a single
consumer GPU per intervention.

### 3. Method Overview

#### 3.1 Activation Differential

For each prompt, two sequences are constructed:

- `seq_A = prompt + base_model_own_answer`
- `seq_B = prompt + teacher_answer`

Both sequences are run through the host model. At every layer, the input to
`mlp.down_proj` (i.e. the post-`act(gate) * up` intermediate, which is the
canonical "neuron activation" in a gated-MLP transformer) is captured per token
over the answer span and averaged. The differential is

```
diff[L, n] = mean_B[L, n] - mean_A[L, n]
```

The answer-span alignment is done positionally token-by-token where the two
answers happen to share token positions, and otherwise their lengths are
equalized by truncation to `min(len(A), len(B))`. Strict one-to-one alignment
is not guaranteed in practice; the goal is to keep the per-position activation
differences meaningful, and to avoid length-bias confounds in the global
average. Neurons with large `|diff|` and a high active rate in at least one
condition become candidate behavior vectors.

#### 3.2 Region Decomposition: Global vs. Vertical Functional Neurons

The QA samples in this project distinguish two kinds of token in the answer
span:

- **Glue tokens** carry little semantic content. They exist to connect
  conclusions: transitions, connective phrases, scaffolding sentences.
- **Conclusion tokens** carry the model's actual judgment and most of the
  semantic load.

Two notions of functional neuron follow from this distinction:

- **Global functional neurons** are identified by differentiating over the
  entire answer span. They tend to capture broad style and tone: how
  "Opus-like" the overall surface looks, regardless of where in the answer
  the depth actually sits.
- **Vertical (region-specific) functional neurons** are identified by
  restricting the differential to a chosen *conclusion region*. The answer
  span is decomposed semantically into:

  - `glue` &mdash; transitional / connective sentences
  - `d1_surface` &mdash; surface critique
  - `d2_structural` &mdash; structural / narrative analysis
  - `d3_thematic` &mdash; thematic depth

  By averaging only over tokens that fall inside a chosen region (typically
  `d3_thematic` for "thinking depth" or `d2_structural` for "structural
  reasoning"), one obtains a finer-grained probe: the neurons that fire
  specifically while the model is *emitting a conclusion*, rather than while
  it is emitting glue. This region-restricted differential is the project's
  primary technique for breaking the polysemantic confusion that plagues
  whole-answer averaging, and it is what reveals which functional axis a
  given neuron actually controls. For example, the project's earliest
  "global" probe identified L27#10024 as a putative *Opus master switch*;
  region decomposition later showed it is in fact a *d1 surface-critique
  switch* and not a thematic-depth switch.

#### 3.3 Causal Verification

For chosen targets, the toolkit performs:

- *Static-graph* analysis: cosine of write direction
  (`down_proj[L_src].weight[:, n]`) against read direction
  (`gate_proj[L_sink].weight[n', :]` or `up_proj`'s analog), weighted by RMS-norm
  scale, over all known-neuron pairs.
- *Attribution patching*: per-layer gradient of a chosen metric M (e.g.
  `act[L_terminal, n_terminal]` summed over anchors) with respect to per-token
  activations, yielding `attribution_diff[L, n] = sum_t grad . (act_B - mean_t' act_A)`.
- *Causal ablation*: forward-pre-hook on `down_proj` input replacing
  `x[..., n]` with `0` (ablate), `2 * x[..., n]` (amplify), or `-x[..., n]`
  (invert), measuring change in M.

Across two independent metric chains, the *sign* of attribution and the *sign*
of measured causal effect agreed for 100% of tested nodes (12/12). This is
strong evidence that attribution-sign can be used as a causal-direction proxy.

#### 3.4 Crutch-Removal SFT

A central finding of this project is that the inventory of "verified" behavior
neurons is not a sparse set of indispensable nodes, but the strongest signals
from a deeper redundant pool. The crutch-removal protocol exploits this:

1. *Crutch 0* (default): no intervention, model produces baseline output.
2. *Crutch 1* (Mode A): all known inventory neurons are suppressed
   (`x[..., n] = 0`); LoRA SFT on five teacher-paired QA samples proceeds
   under suppression. The LoRA's gradient is forced to find substitute paths.
3. *Crutch 2* (Mode C): the inventory neurons *and* the top recruits surfaced
   by Crutch 1 are jointly suppressed; SFT reruns. New "second-pool" neurons
   surface.
4. *Crutch 3* (Mode D): all of the above plus Mode C's recruits are suppressed.
   At this depth, the output layer (L34 in particular) loses sufficient
   capacity that surface-form errors begin to appear (CJK token-table leakage,
   typos), marking the practical floor.

Each suppression round reveals a previously dominated capability layer. The
first round surfaces specific-symbol capture (cigarette glow, bunker code).
The second surfaces in-text quotation and emotional-relation judgments. The
third surfaces character-distinction and motive-depth observations comparable
to the teacher's. None of these capabilities were learned during SFT &mdash;
they were already present in the base model, gated by stronger paths.

#### 3.5 LoRA Task-Vector Ensembling

The three crutch-removal LoRA adapters each encode a direction "away from
Gemma's default and toward a particular slice of teacher behavior", but each
also carries adapter-specific noise (Mode C: narrative misreadings; Mode D:
CJK leakage). Linear combination of `Delta-W` matrices via PEFT's
`add_weighted_adapter` allows:

- *Common direction* across adapters reinforces the shared "teacher-like"
  signal.
- *Adapter-specific noise* averages toward zero.

The empirically best mixture in this project is `dl_pushD_v1`:
`A_off : C : D = 0.15 : 0.30 : 0.55`. At inference, no suppression hooks are
required &mdash; the avoidance pattern is already baked into the merged weights.

#### 3.6 Routine vs. Capability (Attention Anchor Probe)

A subsequent attention probe verified that the LoRA adapter does not learn
parametric depth; it learns a *routine* that more frequently triggers the
mid-layer attention paths (L9 / L19 / L24) which "look back at the input" and
copy concrete content into the working position, where the late-layer
transform circuit (L27 / L34) can act on it. The buff window after each
copy-trigger lasts roughly 5-15 tokens before attention drifts back to recently
generated text and the output reverts to generic phrasing.

Empirical signature on a 745-token generation:

```
mean attention-to-input ratio:
  cite tokens (N=59):  0.278
  buzz tokens (N= 7):  0.224
  Delta:               +0.054

post-cite decay:
  offset  0:  0.278
  offset 15:  0.230  (gradual single-step decay)

per-layer cite-vs-buzz gap:
  L9, L19, L24:   +0.21 to +0.27   (anchor / retrieval layers)
  L27:            -0.02            (transform, NOT anchor)
  L34:            +0.09            (transform, weak anchor)
```

The implication: GHOST's small-sample SFT changes a model's *routing strategy*
(when to retrieve from input), not its *parametric capability*. Generalization
is therefore strong across novel inputs of the same task type and weak across
task domains, exactly as observed.

#### 3.7 Dataset Construction Philosophy

The toolkit operates on five teacher-paired QA samples. This sample size is a
deliberate methodological choice rather than a resource constraint, and it
warrants explicit treatment.

The activation differential `mean_B - mean_A` treats the teacher answer as
the "destination" and the host's own answer as the "origin". The fidelity of
the located behavior vectors depends, to first order, on whether the teacher
answer is in fact a higher-quality realization of the same underlying intent.
A teacher answer that is shallow, off-task, or stylistically incoherent
relative to its prompt will register as a behavior vector pointing somewhere
other than the intended functional axis. The differential method is sensitive
to teacher quality in a way that ordinary supervised fine-tuning is not,
because there is no language-modeling loss term to absorb noise from
mis-paired samples.

For most publicly available "high-quality" datasets &mdash; including those
nominally curated from frontier-model outputs &mdash; per-sample editorial
review is loose enough that a non-trivial fraction of entries violates this
requirement. A dataset nominally labeled "Opus high-quality QA, n=10000" can
in practice contain Opus answering elementary arithmetic problems with no
structural depth, or producing terse procedural answers to prompts that
called for thematic analysis. The mean teacher quality of such a dataset is
pulled below the threshold at which the differential reliably distinguishes
functional signal from incidental signal.

This project therefore trades sample count for editorial control. All five
QA samples were authored or hand-curated by the project author, who retains
full knowledge of each sample's intent, structure, conclusion type, and depth
distribution. This is what makes downstream interpretation tractable: when a
region-restricted differential surfaces a neuron, the author can reason about
which intended axis of the teacher answer that neuron is responding to,
rather than treating the result as an unattributed statistic.

The trade-off is explicit. Findings on this dataset are stable but are not
statistically generalized at conventional sample sizes. Reproductions
intending to extend the toolkit to a new domain should not scale `n` blindly;
they should construct a small, fully controlled dataset for the target domain
and grow it only when each addition demonstrably preserves teacher-quality on
that domain's intended functional axes.

### 4. Findings Summary

| Item | Result |
| --- | --- |
| Verified functional neurons | 57 (4 tiers: verified / general / lit / math, plus thought_quality) |
| Causally verified circuit nodes | 12 (sign consistency 100%) |
| Confirmed motif | Feed-forward inhibition at L09#6039 (drives terminal L33#9054, suppresses L12#4638 which is itself a suppressor of L33#9054) |
| Hidden mid-early circuit hubs | L09 / L11 / L12 (missed by all probes with `L_LO >= 15`; L11#2089 has the strongest single-node causal effect at +43.6% on L12#4638) |
| Crutch-removal floor | ~83 suppressed neurons (Mode D) before output-layer surface forms collapse |
| Best LoRA ensemble | dl_pushD_v1 (A_off:C:D = 0.15:0.30:0.55), 12M trainable parameters, 5 paired QA samples, 4 epochs |
| Attention probe verdict | Adapter learns retrieval routing, not parametric depth |

### 5. Limitations

- **Cannot teach new capability.** A complete chain ablation on the math chain
  (`circuit_chain_per_node.py` on LeetCode 233) caused the model to enter
  repetition loops without solving the problem. The circuit gates style and
  output trajectory; it does not gate algorithmic competence.
- **Routine, not capability.** The dl_pushD_v1 adapter cites teacher-aligned
  passages and uses the resulting attention-buff window for depth, but does
  not generalize to tasks where input citation is unavailable or unhelpful.
- **Think-mode incompatible with clamp.** Across all tested configurations,
  `enable_thinking=True` combined with neuron clamping degrades output. The
  thinking-segment distribution is far enough from the answer-segment
  distribution that constant clamp offsets break the reasoning chain. Disable
  thinking when applying GHOST clamps.
- **Bias-merge model deprecated.** An earlier deliverable
  (`gemma7_E2b_OpusSAE` with merged `down_proj` biases) had downstream
  problems and has been removed. The merge mathematics
  (`x[..., n] += off` is equivalent to adding `off * down_proj.weight[:, n]`
  to the layer's output bias) is documented in `build_opus_sae_model.py` and
  remains correct in principle; the deployed merge produced unstable behavior
  and was not promoted to a production artifact.
- **Probe lower bound was wrong.** All probes used `L_LO >= 15`, missing the
  early hubs at L09-L12. Future probes should set `L_LO = 0`.
- **Hard-coded paths.** All scripts use absolute Windows paths
  (`J:/amr/amr_wtf/...`) and a specific conda environment
  (`D:\ProgramData\anaconda3\envs\gemmapreview`). The repository is a research
  record, not a portable library.

### 6. Repository Layout

```
amr_wtf/
  README.md                            this file
  NEURONS.md                           detailed neuron inventory
  NEURON_INVENTORY.md                  tier and merge-scale documentation
  CIRCUIT_FINDINGS_2026-05-01.txt      static + attribution + ablation report
  20260502_phase2_results.txt          phase-2 verification + stop-signal
  20260502_distributed_neurons.txt     distributed-representation roadmap

  data/                                paired QA samples (literary + code)
  outputs/                             per-experiment .pt and .log artifacts

  probe_*.py                           single-QA / cross-QA / region probes
  region_*.py                          region decomposition and aggregation
  circuit_static.py                    cosine write-vs-read graph
  circuit_attr_*.py                    attribution patching
  circuit_verify_*.py                  causal ablation
  circuit_chain_*.py                   chain-level ablation

  chat/
    app.py                             Tkinter chat UI with live clamp
    runtime.py
    steering.py
    neurons.json                       neuron inventory + presets

  opus47code/
    crutch_pipeline.py                 generic crutch-removal SFT pipeline
    configs/mode_*.yaml                pipeline configs (E/F/G compatible)
    crutch_ensemble_infer.py           LoRA task-vector ensembling
    crutch_ensemble_depth_sweep.py     mixture-weight grid search
    crutch_attn_anchor_probe.py        attention-routing verification
    multi_qa_intent_sft.py             weighted multi-QA SFT
```

### 7. Reproduction

This is a research record and not a one-click reproducible artifact. Key
caveats:

- Model: `J:/amr/models/gemma-4-E2B-it` (BF16 for training, INT8 for early
  probes via `bitsandbytes`).
- Environment: Python 3.12, PyTorch 2.6+ with CUDA 12.6, `transformers`
  with Gemma 4 support, `peft >= 0.10`, `bitsandbytes`, `fastapi`, `uvicorn`,
  `tkinter`.
- The `opus47code/` and `geminicode/` directories use absolute paths; reading
  them in order (probe -> region -> circuit -> crutch_pipeline -> ensemble ->
  attn_probe) reconstructs the project trajectory.
- Five teacher-paired QA samples are required. The samples used in this
  project are derived from Anthropic and Google commercial outputs and are
  not redistributed; users wishing to reproduce should generate equivalent
  paired data with the teacher model of their choice.

### 8. Acknowledgments and Citation

This project is built on Google DeepMind's Gemma 4 E2B-it and uses paired
outputs from Anthropic's Claude Opus 4.6 and Google's Gemini 3 Flash Preview
Thinking as teacher signals.

If this work is useful, cite as:

```
GHOST: Ghost-Host Output Steering Toolkit
A hallucination-driven behavior-vector extraction method for small language models
2026
```

### 9. License

The code in this repository is released for research use. The Gemma 4 E2B-it
model is governed by Google's Gemma Terms of Use; downstream users must comply
with that license. Teacher-paired QA samples in `data/` are not redistributed
and must be regenerated locally by users with access to the teacher models in
question.

---

## 中文

### 摘要

GHOST 是一套针对小型开放权重语言模型（Gemma 4 E2B-it）的研究工具集，通过利用
模型无法内省自身生成过程这一性质，抽取并应用 *行为向量*。具体方式是把模型自己的
回答替换为 *教师* 回答（Claude Opus 4.6 / 一份更好的回答 / 一份无幻觉的回答），
尽可能将两边的 token 逐位对应（或者至少让两边长度一致）后，逐神经元做激活差分，
定位驱动特定输出行为（深度、结构、引用、前端细节）的 *功能性神经元*。这些向量随后可以通过 clamp 注入、抑制、LoRA 训练，或
LoRA 任务向量集成等手段，把 host 模型导向教师的行为，成本远低于常规微调。

项目目录名 *amr_wtf* 是历史名（"a moment of random what-the-fuck"），表明它源
于一个一闪而过的主意而非有计划的工程。对外的代号 *GHOST* 更准确地刻画了机制：
让 host 模型为 teacher 输出 "代笔"。

### 1. 动机

本项目的原理其实非常简单。大语言模型不知道自己输出的结果是怎么来的，对它而
言，预测的词是凭空出现的。这给了本项目一个契机：把模型的回答替换成 Claude
Opus、或更好的回答、或非幻觉的回答，与基线回答的神经元激活进行差分，即可找到
让模型倾向于回答出教师模型偏向的功能性神经元。这正是学界所说的 *行为向量*。

本项目的工程哲学核心区分两类神经元：

- **知识性神经元**：存储具体事实（金门大桥、水的沸点等）。
- **功能性神经元**：让内容更有层次和深度、理解文学语义、给出更好的前端细节
  和设计。

GHOST 成功诱导 Gemma 4 E2B-it 误以为自己就在输出 Opus 的回答，并基于这种
"幻觉" 找到了让 Gemma 的回答更接近 Opus 的功能性神经元。

在单个神经元之外，模型内部还存在：

- **功能性神经链路**：跨层的因果链，可通过 attribution patching 观察。
- **大团神经元**：功能近似的神经元在某些层段密集分布。本项目把 "大团神经
  元" 作为独立结构对象的存在视为一个 *未验证猜想* &mdash; 工具集本身并
  不包含证明它作为一类首要实体存在的直接实验。但是，Gemma 4 E2B-it 在
  instruction-tuning（IT / RLHF）阶段获得的 *舒适圈* 约束，无论从行为还
  是功能上看，都与人们预期的 "大团功能性神经元" 表现非常接近：在激活空
  间里像一个抽象凹陷，凹陷提供的吸引力把输出拉回 instruction-tuned 默
  认。从这一意义上，工程概念 "大团功能性神经元" 与后 RLHF 概念 "舒适圈"
  可以视为同一现象的两种近似等价的操作性描述。这种近似等价 *受到本项目
  以下观察的启示但并未被严格证明*：渐进抑制 "最强者" 后，被招募来补位的
  总是同一层段内的次强者，从不出现新层段冒头。

### 2. 实用价值

例如，本项目开源时有一个模型（GPT-5.5）在某个奖励信号 RLHF 之后学会了滥用
"哥布林" 这一特定 token 来刷分，结果输出大量充斥该词。GHOST 可以在不进行
后训练、不追加系统提示词的情况下修复此类问题：

1. 准备两份等长回答：一份含哥布林，一份不含。
2. 在对齐答案 span 上做 QA 差分，得到逐神经元激活差。
3. 定位到对应该词输出的功能性神经元。
4. 抑制它（把 `down_proj` 输入端该列置零），或把对应的偏置合并进模型权重，
   从而做到零推理时开销。

由于差分方法天然偏向功能性神经元（知识性神经元在不同 prompt 上很少表现出
稳定的跨上下文激活模式），定位到的神经元在实践中就是正确的干预点。

同样的流程适用于：

- 特定情景下的幻觉修复
- 单样本偏好微调
- 不微调情况下的风格引导

主要成本仅是单卡消费级 GPU 上每次干预数分钟的前向推理。

### 3. 方法概要

#### 3.1 激活差分

对每个 prompt 构造两条序列：

- `seq_A = prompt + 基线模型自答`
- `seq_B = prompt + 教师回答`

两条序列分别前向，逐层在 `mlp.down_proj` 的输入张量（即 `act(gate) * up` 后
的中间激活，是 gated MLP transformer 中 "神经元活动" 的标准定义）上 hook 抓
取，对答案 span 上的 token 做平均。差分为

```
diff[L, n] = mean_B[L, n] - mean_A[L, n]
```

答案 span 的对齐采取 "尽可能逐位对应" 的策略：在两边答案恰好共享 token 位
置处做位置对位，否则通过 `min(len(A), len(B))` 截断让两边长度一致。实际中
并不保证严格的一一对应；目标是让逐位激活差异保持可比，并避免长度差异污染
全局平均。`|diff|` 大且至少在某一条件下高激活率（active rate）的神经元成
为候选行为向量。

#### 3.2 区域分桶：全局功能性神经元 与 区间垂直功能性神经元

本项目的 QA 样本把答案 span 中的 token 区分为两类：

- **胶水词**（glue tokens）：本身不承载太多意义，只起把结论连接起来的作
  用，例如过渡句、连接词、承启性语句。
- **结论词**（conclusion tokens）：模型实际给出的结论，承载大部分语义负载。

由此自然区分出两种功能性神经元：

- **全局功能性神经元**：在整个答案 span 上做差分得到的神经元。这类神经元
  往往捕捉广义的风格与基调 &mdash; 即整体回答 "看起来有多 Opus"，不区分
  深度实际出现在答案的哪一段。
- **区间垂直功能性神经元**：把差分限制在某个 *结论区间* 上得到的神经元。
  做法是把答案 span 按语义进一步分桶：

  - `glue` 过渡 / 连接句
  - `d1_surface` 表面批评
  - `d2_structural` 结构 / 叙事分析
  - `d3_thematic` 主题深度

  仅在所选区间（通常是 `d3_thematic`，对应 "思想深度"；或 `d2_structural`，
  对应 "结构化推理"）的 token 上做平均，即可定位到 *只在模型输出结论时*
  激活的神经元，而不是仅在输出胶水词时激活的神经元。这是本项目用来打破整
  答案平均带来的 polysemantic 混淆的主要技术，也正是它揭示了某个神经元实
  际控制的具体功能轴。举例：项目最早期的 "全局" 探针把 L27#10024 误标为
  "Opus 总开关"；区域分桶之后才发现它实际上是 "d1 表面挑细节" 开关，并
  不是主题深度开关。

#### 3.3 因果验证

对选定目标，工具集执行：

- **静态权重图**：write 方向（`down_proj[L_src].weight[:, n]`）与 read 方向
  （`gate_proj[L_sink].weight[n', :]` 或 `up_proj` 对应项）的 cos 相似度，按
  RMS-norm 加权，覆盖所有已知神经元两两组合。
- **Attribution patching**：以选定指标 M（如终端神经元在 anchor token 上的
  激活和）对逐 token 激活求梯度，得到
  `attribution_diff[L, n] = sum_t grad . (act_B - mean_t' act_A)`。
- **因果消融**：在 `down_proj` 输入处装 forward-pre-hook，把 `x[..., n]` 替
  换为 0（ablate）、`2 * x[..., n]`（amplify）、或 `-x[..., n]`（invert），
  测量 M 的相对变化。

在两条独立指标链上，attribution 的符号与因果效应符号对所有测过的节点（共
12/12）100% 一致。这一现象提供了把 attribution sign 当作因果方向代理的实
证基础。

#### 3.4 渐进抽拐杖式 SFT（Crutch-Removal SFT）

本项目的核心发现之一：所谓 "verified" 的神经元清单不是稀疏的不可替代节点，
而是一个更深的冗余池中信号最强的部分。基于这一观察的 crutch-removal 协议：

1. **拐杖 0**（默认）：不做任何干预，模型给出基线输出。
2. **拐杖 1**（Mode A）：把所有 inventory 神经元抑制（`x[..., n] = 0`）后，
   在 5 对教师配对 QA 样本上做 LoRA SFT。LoRA 梯度被迫寻找替补路径。
3. **拐杖 2**（Mode C）：把 inventory 神经元 + Mode A 招募的 top 神经元一并
   抑制，重做 SFT。新一批 "二号位" 神经元浮出。
4. **拐杖 3**（Mode D）：在 2 的基础上再加上 Mode C 招募的 top 神经元。在这
   一深度，输出层（尤其 L34）容量已不足，开始出现表层错误（CJK 词表漏字、
   错字），即触底信号。

每一轮抑制都揭开此前被支配的能力层：第一轮浮出 "具体符号捕捉"（烟草微光、
地堡密码）；第二轮浮出 "原文引用 + 情绪关系判断"；第三轮浮出与教师可比的
"角色区分 + 动机深度"。这些能力都不是 SFT 学到的 &mdash; 它们一直在基模型
里，只是被更强的路径压住了。

#### 3.5 LoRA 任务向量集成

三个 crutch-removal 阶段产出的 LoRA adapter 都各自编码 "远离 Gemma 默认 +
偏向某一切片教师行为" 的方向，但每个也都带各自的噪声（Mode C：叙事误读；
Mode D：CJK 漏字）。通过 PEFT 的 `add_weighted_adapter` 对 `Delta-W` 矩阵
做线性组合：

- 多 adapter 共享的 "更像 teacher" 方向叠加增强。
- 各 adapter 各自的噪声平均后趋零。

本项目实测最优配比为 `dl_pushD_v1`：
`A_off : C : D = 0.15 : 0.30 : 0.55`。推理时无需挂任何 suppression hook，
因为 "回避路径" 已经直接体现在合并后的权重里。

#### 3.6 Routine 还是 Capability（注意力锚点探针）

后续注意力探针验证了一个关键问题：LoRA adapter 学到的不是参数化深度，而是
一个 *routine* &mdash; 它更频繁地触发中层（L9 / L19 / L24）的注意力，让
attention "回头看输入"，把具体内容搬运到当前生成位置；末层（L27 / L34）随
后用搬到的材料做转换。每次搬运触发后的 buff 窗口约 5&ndash;15 token，之后
注意力漂回最近生成的 token，输出又退回通用模板。

745 token 生成上的实证特征：

```
平均 input-attention ratio：
  cite tokens (N=59):  0.278
  buzz tokens (N= 7):  0.224
  Delta:               +0.054

cite 后衰减：
  offset  0:  0.278
  offset 15:  0.230  （单调缓降）

逐层 cite-vs-buzz gap：
  L9, L19, L24:   +0.21 ~ +0.27   （anchor / 检索层）
  L27:            -0.02            （转换层，非 anchor）
  L34:            +0.09            （转换层，弱 anchor）
```

含义：GHOST 的小样本 SFT 改的是模型的 *路由策略*（何时回输入检索），不是它
的 *参数化能力*。因此泛化在同任务类型的新输入上较强，跨任务域较弱，与项目
后期跨样本测试的结果完全一致。

#### 3.7 数据集构建哲学

工具集运行在 5 对教师配对 QA 样本上。这一样本量是有意为之的方法论选择，
不是资源限制，因此值得单独说明。

激活差分 `mean_B - mean_A` 把教师回答视为 "目标"，把基模型自答视为 "起
点"。所定位行为向量的可靠性，一阶取决于教师回答是否确实是同一底层意图的
更高质量实现。一份相对其 prompt 而言浅薄、跑题、或风格不连贯的教师回
答，会被记录为指向预期功能轴之外某处的 "行为向量"。差分方法对教师质量
的敏感度高于常规监督微调，因为这里没有语言建模损失项来吸收错配样本带来
的噪声。

对大多数公开可得的 "高质量" 数据集 &mdash; 包括那些标称从 frontier 模型
输出整理而来的 &mdash; 逐条编辑审查的密度都低到不足以满足这一要求。一份
名义上叫 "Opus 高质量 QA, n=10000" 的数据集，实际中可能包含 Opus 在做
小学算术且毫无结构深度的样本，或者对应当做主题分析的 prompt 给出简短的
流程化回答。这种数据集的平均教师质量会被拉到差分方法能可靠区分功能性信
号与偶然信号的阈值以下。

本项目因此用样本数量换编辑控制权。5 个 QA 样本全部由作者自写或经手审
定，作者完整掌握每条样本的意图、结构、结论类型与深度分布。这正是使下
游解释变得可处理的前提：当某次区间限定差分浮出一个神经元时，作者能够
判断它响应的是教师回答中预期的哪一条功能轴，而不是把结果当作一个无归
属的统计量。

这一权衡是明确的。结论在该数据集上稳定，但在常规样本规模意义下并未做
统计意义上的泛化。希望把工具集扩展到新领域的复现工作不应盲目扩大
`n`；应当为目标领域构建一份小而完全可控的数据集，并仅当每次扩充都可证
明地保持目标领域预期功能轴上的教师质量时才增长它。

### 4. 主要发现汇总

| 项目 | 结果 |
| --- | --- |
| 已验证的功能性神经元 | 57 个（4 个主 tier：verified / general / lit / math，外加 thought_quality） |
| 因果验证的链上节点 | 12 个（attribution 符号与因果方向一致率 100%） |
| 确认的 motif | L09#6039 上的前馈抑制（同时正向驱动终端 L33#9054 并压制 L12#4638，而 L12#4638 本身是 L33#9054 的抑制者） |
| 隐藏的早中段电路 hub | L09 / L11 / L12（被所有 `L_LO >= 15` 的探针漏掉；L11#2089 单点因果效应最强，对 L12#4638 invert +43.6%） |
| Crutch-removal 物理底 | 抑制约 83 个神经元（Mode D）后输出层表层崩塌 |
| 最优 LoRA 集成 | dl_pushD_v1（A_off:C:D = 0.15:0.30:0.55），12M 可训参数，5 对 QA 样本，4 epoch |
| 注意力探针结论 | adapter 学到检索路由，非参数化深度 |

### 5. 局限

- **不能教会新能力**。在数学链上做完整链路 ablation（LeetCode 233 上跑
  `circuit_chain_per_node.py`）会让模型陷入复读，无法解题。该电路控制的是
  风格与输出走向，不是算法能力。
- **Routine 而非 Capability**。dl_pushD_v1 通过 cite 教师一致段落、利用搬
  运后的 attention-buff 窗口产出深度内容；在没有可 cite 输入或 cite 无意义
  的任务上，它不会自动迁移。
- **思考模式与 clamp 不兼容**。在所有测试配置下，`enable_thinking=True` 配
  合神经元 clamp 都会让输出退化。思考段分布与回答段分布相距甚远，常数 clamp
  偏移会破坏思考链。使用 GHOST clamp 时关闭思考模式。
- **Bias-merge 模型已弃用**。一个早期产物 `gemma7_E2b_OpusSAE`（把
  `down_proj` 偏置合并入权重）在下游使用中表现不稳定，已删除。合并的数学
  （`x[..., n] += off` 等价于在 `down_proj` 输出上加
  `off * down_proj.weight[:, n]`）记录在 `build_opus_sae_model.py` 中，原理
  上仍正确，但实际部署的合并产物未达到生产可用状态。
- **探针下界设置错了**。所有探针都用 `L_LO >= 15` 起步，漏掉了 L09&ndash;
  L12 的早期 hub。后续探针应设为 `L_LO = 0`。
- **路径硬编码**。所有脚本使用绝对 Windows 路径（`J:/amr/amr_wtf/...`）和
  指定 conda 环境（`D:\ProgramData\anaconda3\envs\gemmapreview`）。本仓库
  是研究记录，不是可移植的库。

### 6. 仓库结构

```
amr_wtf/
  README.md                            本文件
  NEURONS.md                           神经元清单（详）
  NEURON_INVENTORY.md                  tier 与合并 scale 文档
  CIRCUIT_FINDINGS_2026-05-01.txt      静态图 + attribution + 消融报告
  20260502_phase2_results.txt          phase-2 验证 + stop-signal 实验
  20260502_distributed_neurons.txt     分布式表征研究路线图

  data/                                配对 QA 样本（文学 + 代码）
  outputs/                             各实验 .pt 与 .log 产物

  probe_*.py                           单 QA / 跨 QA / 区域探针
  region_*.py                          区域分桶与跨 QA 聚合
  circuit_static.py                    cosine write-vs-read 图
  circuit_attr_*.py                    attribution patching
  circuit_verify_*.py                  因果消融
  circuit_chain_*.py                   链路级消融

  chat/
    app.py                             带 clamp 的 Tkinter 聊天界面
    runtime.py
    steering.py
    neurons.json                       神经元清单 + 预设组合

  opus47code/
    crutch_pipeline.py                 通用 crutch-removal SFT 流水线
    configs/mode_*.yaml                流水线配置（E/F/G 兼容）
    crutch_ensemble_infer.py           LoRA 任务向量集成
    crutch_ensemble_depth_sweep.py     混合权重网格搜索
    crutch_attn_anchor_probe.py        注意力路由验证
    multi_qa_intent_sft.py             加权多 QA SFT
```

### 7. 复现

本仓库是研究记录，不是一键可复现产物。关键说明：

- 模型：`J:/amr/models/gemma-4-E2B-it`（训练用 BF16，早期探针经
  `bitsandbytes` INT8）。
- 环境：Python 3.12，PyTorch 2.6+ 配 CUDA 12.6，含 Gemma 4 支持的
  `transformers`，`peft >= 0.10`，`bitsandbytes`，`fastapi`，`uvicorn`，
  `tkinter`。
- `opus47code/` 与 `geminicode/` 内脚本使用绝对路径；按 probe -> region
  -> circuit -> crutch_pipeline -> ensemble -> attn_probe 的顺序阅读可
  完整还原项目轨迹。
- 需要 5 对教师配对 QA 样本。本项目使用的样本基于 Anthropic 与 Google 商业
  模型输出，未随仓库分发；希望复现的用户应自行用所选教师模型生成等价的配对
  数据。

### 8. 致谢与引用

本项目基于 Google DeepMind 的 Gemma 4 E2B-it，并使用 Anthropic Claude Opus
4.6 与 Google Gemini 3 Flash Preview Thinking 的输出作为教师信号。

如本工作有用，可引用为：

```
GHOST: Ghost-Host Output Steering Toolkit
A hallucination-driven behavior-vector extraction method for small language models
2026
```

### 9. 许可

本仓库代码用于研究用途。Gemma 4 E2B-it 模型受 Google Gemma Terms of Use 约
束，下游使用者须遵守该许可。`data/` 中的教师配对 QA 样本未随仓库分发，需要
复现的用户应在本地使用对应教师模型重新生成。
