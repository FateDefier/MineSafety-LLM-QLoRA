以下参数来自 **deepseek-v4-pro** 的解释：

---

## 一、基础运行设置

| 参数 | 含义 | 选择建议 |
|------|------|----------|
| `model_name_or_path` | 预训练模型路径（本地或 HuggingFace 模型名） | 填写实际路径或官方名称，确保有 `qwen2.5-7b-instruct` |
| `stage: sft` | 训练阶段：监督微调（SFT） | QLoRA 通常用于 SFT，无需改动 |
| `do_train: true` | 执行训练（否则仅推理/评估） | 必须为 `true` |
| `finetuning_type: lora` | 微调类型：LoRA | QLoRa 本质是 `lora` + 量化，这里固定 |
| `trust_remote_code: true` | 允许执行自定义模型代码（如 Qwen 的自定义建模文件） | 非官方 Hub 模型或 Qwen 旧版需要设为 `true`，安全前提 |
| `template: qwen` | 对话模板（用于格式化输入输出） | 必须与模型匹配（如 qwen、llama 3、chatglm 3 等） |
| `output_dir` | 保存 checkpoint 和最终模型的目录 | 自定义，建议有空余磁盘空间 |
| `report_to: none` | 训练日志上报目标（wandb / tensorboard / none） | 测试或不想用外部工具时选 `none`，否则 `tensorboard` 方便本地查看 |

---

## 二、数据相关参数

| 参数 | 含义 | 选择建议 |
|------|------|----------|
| `dataset: mine-safety-data` | 使用的数据集名称（需在 `dataset_info.json` 中定义） | 根据实际任务命名，确保格式符合 SFT 要求（如 instruction / input / output 字段） |
| `dataset_dir: data` | 数据集存放目录 | 默认 `data` 即可，可修改 |
| `max_samples: 100000` | 最多使用的样本数（所有数据会随机截取） | 若数据量极大（如百万级），可设较小值加速试验；完整训练时去掉或设很大 |
| `val_size: 0.1` | 从训练集中分割 10% 作为验证集 | 数据量大时 0.05~0.1 均可；若已有独立验证集则设为 0，并用 `eval_dataset` 指定 |
| `preprocessing_num_workers: 16` | 数据预处理并行进程数 | 根据 CPU 核心数设置（如 8~16），过多可能 OOM(`Out Of Memory`)，建议 4~8 大部分情况够用 |
| `cutoff_len: 2048` | 输入序列的最大长度（token 数） | 根据显存和任务调整：7 B 模型 + 4-bit + 2048 长度占用约 12~16 GB VRAM；若数据含长文档可增至 4096，但量化后仍可能 OOM |
| `packing: false` | 是否将多个短样本拼接打包，提高训练效率 | 建议 `false`（传统方式更稳定）；若序列都很短（<256）且想提升速度可开 `true`，但需要框架支持且 loss 计算会忽略 padding |

---

## 三、QLoRA 量化核心参数

| 参数 | 含义 | 选择建议 |
|------|------|----------|
| `quantization_bit: 4` | 量化位数（4-bit） | QLoRA 标配 4-bit，也可 8-bit（但显存节省不明显）。**4** 是最常用的 |
| `quantization_method: bnb` | 量化方法，使用 bitsandbytes 库 | 固定为 `bnb`（目前 QLoRA 仅此一种成熟方案） |
| `double_quantization: true` | 是否在 4-bit 基础上再对量化常数做二次量化（节省约 0.5% 显存） | 建议 `true`，基本无损失但省一点显存 |
| `bf16: true` | 使用 bfloat 16 精度进行计算（需要 Ampere 及以上架构 GPU） | 若 GPU 支持（A 100/RTX 3090/4090/H 100 等）且数据含大数值，开启可提高稳定性；否则用 `fp16: true`。注意同时开启 `bf16` 和 `fp16` 会冲突 |

---

## 四、LoRA 适配器参数

| 参数 | 含义 | 选择建议 |
|------|------|----------|
| `lora_rank: 16` | LoRA 低秩矩阵的秩 `r` | 常见值 8, 16, 32。**16** 是良好起点，任务复杂可到 32，简单任务 8 即可。**秩越大参数量越多，效果上限略高但过拟合风险增加** |
| `lora_alpha: 32` | LoRA 缩放系数 `α` | 通常设为 `2 × r`（即 32）或 `r` 的一倍多。`α` 越大，LoRA 分支影响越强，可微调时动态调整。32 是稳妥选择 |
| `lora_dropout: 0.1` | LoRA 层的 dropout 概率 | 小数据集（<1 k）可设 0.1 **防过拟合**；大数据集可设 0.0；一般 0.05~0.1 |
| `lora_target` | 要应用 LoRA 的模块名称（线性层） | 对于 Qwen/Llama 系模型，一般包括 `q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj`（即**所有 attention 和 FFN 投影层**）。列全可最大化可训练参数，若显存紧张可只选 `q_proj, v_proj` |

---

## 五、训练超参数

| 参数 | 含义 | 选择建议 |
|------|------|----------|
| `num_train_epochs: 3.0` | 训练总轮数 | 2~5 轮常见。数据集大（>10 k）可 2~3 轮；小数据集（<1 k）可 5 轮但注意过拟合 |
| `per_device_train_batch_size: 2` | 每张 GPU 的 batch size（实际样本数） | 因量化 + 长序列显存受限，2 是常见值。若显存有余可增至 4 或 8 |
| `per_device_eval_batch_size: 2` | 评估时的 batch size | 可与训练一致，评估更省显存，2 即可 |
| `gradient_accumulation_steps: 4` | 梯度累积步数（有效 batch size = `per_device_batch × accum × GPU数`） | 设 4 可使有效 batch 达到 2×4=8，稳定训练。若显存极小可增大累积步数 |
| `learning_rate: 0.0002` | 学习率 | LoRA 常用 1 e-4 ~ 5 e-4。**0.0002（2 e-4）** 是较好起点；若发现 loss 震荡可降低至 1 e-4 |
| `lr_scheduler_type: cosine` | 学习率调度策略 | `cosine` 适合中等长度训练（3~10 epochs）；训练很短（1 epoch）可用 `linear`；`constant` 很少用 |
| `warmup_steps: 0` | 预热步数（学习率从 0 线性上升到目标值的步数） | 数据集较大（>10 k steps）可设 100~500；小数据集或 `cosine` 调度也可设 0。这里为 0 表示无预热 |
| `optim: adamw_torch` | 优化器 | `adamw_torch`（PyTorch 原生）稳定；也可 `paged_adamw_8bit`（bitsandbytes 节省显存），但一般 4-bit 模型下 `adamw_torch` 足够 |
| `max_grad_norm: 1.0` | 梯度裁剪的最大范数 | 1.0 是常用值，防止梯度爆炸。可保持 |
| `include_num_input_tokens_seen: true` | 在日志中记录已处理的 token 总数 | 辅助监控训练量，建议 `true` |

---

## 六、训练流程与日志

| 参数 | 含义 | 选择建议 |
|------|------|----------|
| `eval_strategy: steps` | 评估时机：按步数 | 推荐 `steps` |
| `eval_steps: 100` | 每隔 100 步在验证集上评估一次 | 根据数据集大小：若每 epoch 步数很多（如 2000 步），100 步可频繁监控；若步数少（<500），可改为 `epoch` |
| `save_steps: 100` | 每隔 100 步保存一次 checkpoint | 可与 `eval_steps` 相同或倍数关系；注意磁盘空间 |
| `logging_steps: 5` | 每 5 步打印一次 loss 等信息 | 5~10 步即可，太密集日志过多 |
| `plot_loss: true` | 自动绘制 loss 曲线（保存到输出目录） | 方便可视化，建议 `true` |
| `ddp_timeout: 180000000` | 分布式训练超时（秒） | 单卡训练可忽略；多卡时若训练时间长，设大值（如 180000 秒 = 50 小时） |
| `flash_attn: auto` | 是否使用 FlashAttention-2 加速 | `auto` 表示自动检测可用性。若 GPU 支持（A 100/3090/H 100 等）推荐开启，可显著降低显存并加速长序列 |

---

## 七、其他

| 参数                      | 含义        | 选择建议                                        |
| ----------------------- | --------- | ------------------------------------------- |
| `enable_thinking: true` | 控制是否输出思考链 | 若任务为推理题可开 `true`，一般聊天指令微调可能无关，保持 `true` 无影响 |

---
