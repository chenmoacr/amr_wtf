# SAE 神经元清点（截至 2026-04-29）

来源：`J:\amr\amr_wtf\chat\neurons.json`

## 总览

| Tier | 数量 | 合并 scale | 说明 |
|---|---|---|---|
| verified | 3 | **1.0** | 实机验证有效（V1 三件套） |
| thought_quality | 14 | **1.0** | thought 段锚定（QA01 探针） |
| general | 2 | **1.0** | 跨域有效 |
| lit | 11 | **0.5** | 文学域风格类，减弱合并（"比原本稍强"） |
| math | 27 | **0.5** | QA02 + GB01 region/blockwise，减弱合并 |
| candidate | 10 | **0.0** | n=1 未验证，**跳过** |
| **合并总数** | **57** | | |
| **跳过** | **10** | | candidate tier |

## 各 tier 神经元详细列表

### verified（3，scale=1.0）

| id | layer | gain | region | 用途 |
|---|---|---|---|---|
| L15#1511 | 15 | +2.0 | answer | 代码结构布局 pos |
| L26#449 | 26 | +2.0 | answer | 代码脚手架放大 pos |
| L16#1298 | 16 | -2.0 | always | anti-lazy（双段位） |

### thought_quality（14，scale=1.0）

T_d2（结构层）：
- L15#4995 +2.0 thought, L15#4270 +2.0 thought, L25#10262 +2.0 thought
- L34#6819 -1.5, L34#384 -1.5, L34#1614 -1.5, L34#6608 -1.5（SOP 模板抑制）

T_d1（具体知识写入电路，BF16 探针）：
- L15#11278 +2.5 thought（最强正向）
- L17#11943 +2.0 thought
- L23#8588 -2.0, L19#8648 -2.0, L18#8398 -2.0, L24#7991 -2.0, L16#6159 -2.0

### general（2，scale=1.0）

- L26#5430 -1.0 answer  general/tone_neg
- L27#2890 +1.5 answer  general/long_structured_output_pos（注意：thought 段反向）

### lit（11，scale=0.5）

- L27#10024 -3.0, L27#6644 +2.0, L32#9383 -2.0, L32#8474 +1.0, L27#8968 +1.0
- L34#8522 +1.0, L31#604 -1.0, L15#8146 +1.0, L28#7686 -1.0, L27#5590 -1.0, L23#287 -1.0

### math（27，scale=0.5）

QA02 region 探针（12）：
- L26#10136 -2.5 always, L16#8035 -2.0 always, L15#11943 +1.5 always
- L25#4879 -2.5, L25#3178 -2.0, L34#3696 -1.5, L34#3169 -1.5, L17#7479 -1.5
- L16#8358 +2.0, L15#258 +2.0, L17#10314 +1.5
- L27#755 +1.5 thought

GB01 region 探针（5）：
- L33#9054 +2.0, L17#2838 -1.5 thought, L33#9511 -1.5 always
- L15#150 +1.5, L15#2067 -1.5 always

GB01 blockwise 探针（10）：
- L27#8010 +2.5（最强 pos）, L17#7444 -2.0, L20#2810 -2.0 thought, L28#3943 +1.5
- L23#1867 -1.5 thought, L34#9243 -1.5, L21#4744 -1.5 thought, L15#5746 -1.5 thought
- L17#7873 +1.5, L19#1010 +1.5

### candidate（10，**跳过**）

L26#12083, L15#7439, L24#5158, L17#5551, L17#4784, L34#3650, L34#3966, L34#1227, L25#11331, L15#8289

n=1 单次发现未验证，合并风险大于收益。**保留在 neurons.json 供未来 probe 复用。**

## 合并策略说明

1. **数学等价**：clamp `x[..., n] += offset` 等同于在 down_proj 输出加常向量 `offset * W[:, n]`。多个神经元同层叠加为 `bias_vec = Σ offset_n * W[:, n]`。
2. **实现**：将受影响的 down_proj 替换为 `bias=True` 版本，bias 设为该向量。**runtime 0 hook 开销，纯 weight 修改，等价于原 clamp 行为**。
3. **符号守恒**：负向 gain 保持负向（`final_offset = global_alpha * gain * tier_scale`，`tier_scale > 0`，所以符号由 gain 决定）。
4. **region 失效**：合并后 thought/answer/always 区分失效，所有 bias 恒定生效。这是接受的代价。
5. **lit/math 减弱**：scale=0.5 让风格/数学神经元贡献减半（避免过推）。

## 跨域 sign-flip 警告

以下神经元在不同任务域方向不一致，合并后可能在某些任务上反作用：
- L34#8522（lit/d2_marker_pos）：QA02 散文域 +，GB01 块级偏 A 侧
- L34#6819 / L34#6608 / L34#1614（thought SOP_template_neg）：UI thought 域 -，GB01 代码输出域 +
- L17#4784（candidate，已 skip）：thought 段 sign-flip

合并后这些会**始终偏向其 default_gain 方向**。如果运行表现异常，可考虑手动从合并集合排除。
