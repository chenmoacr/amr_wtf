# Gemini Code Steering 研究目录

本目录旨在通过对比 QA 差异来定位和研究 Gemma-4-E2B 的 SAE 神经元，特别关注代码生成任务。

## 目录结构
- `qa_neuron_probe.py`: 核心探测脚本。支持 BF16 加载，区分 Think/Output 阶段。
- `outputs/`: 存储实验结果。
    - `analysis_report.txt`: 自动生成的神经元差异排名。
    - `diff_stats.pt`: 完整的激活均值与激活率数据（PyTorch 格式）。
- `REPORT.md`: 实验结论与人工解读。

## 运行方法
确保当前环境有 `transformers`, `torch`, `accelerate`。
```bash
python qa_neuron_probe.py
```

## 协作指南
1. **获取神经元列表**: 读取 `outputs/analysis_report.txt` 获取 top diffs。
2. **进一步实验**: 使用 `J:/amr/amr_wtf/clamp_experiment.py` 对本报告中发现的神经元进行 Clamp 验证。
3. **数据更新**: 如果有新的 LeetCode 失败/成功对，请更新 `J:/amr/amr_wtf/data/gemma_TargetedDrugs/gemma_code_GB01.json` 并重新运行探测。
