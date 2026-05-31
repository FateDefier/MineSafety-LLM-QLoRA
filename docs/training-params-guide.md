> **注意**：本文是一份学习笔记，记录了参数选择的思考过程和经验总结。文中使用的示例参数（rank=8, lr=5e-5, cutoff=4096）来自早期实验，与本项目最终采用的参数（rank=16, lr=2e-4, cutoff=2048）有所不同。具体参数对比和选择原因请参考项目 README 的”训练配置”章节。

## 调整微调参数

在模型微调中，各类参数就像是你在给模型 “补课” 之前制定的教学计划和策略。它们决定了你如何教学、教学的强度以及教学的方向。如果你选择的教学计划不合适（比如补课时间太短、讲解速度太快或复习策略不合理），可能会导致学生学习效果不好。同样，如果你选择的超参数不合适，模型的性能也可能不理想。

过去经常收到很多同学的问题："在微调过程中这些参数到底要怎么设置效果才最好？" ，这肯定不是一个标准答案，参数的设置和你选择的模型、用于微调的数据集，以及你微调机器的硬件配置都有关系，并不存在 “最佳参数” 这个说法，下面我会结合我个人的经验，尽可能的告诉大家一些关键参数的作用和影响，以及在实际微调任务中调整的思路是什么。没有在下面讲到的参数，大家都直接保持默认值即可。

---

### 学习率（Learning Rate）

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUl0o4fdHMDJNbQtTx2bDBeQS6GDUibObF1b3OL2dvgdamokjxOC4tU49w/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

|**分类**|**说明**|
|---|---|
|**核心概念**|**学习率（Learning Rate）**<br><br> 决定了模型在每次更新时参数调整的幅度，通常在 (0, 1) 之间。也就是告诉模型在训练过程中 “学习” 的速度有多快。学习率越大，模型每次调整的幅度就越大；学习率越小，调整的幅度就越小。|
|**通俗理解**|通俗来说，学习率可以用来控制复习的“深度”，确保不会因为调整幅度过大而走偏，也不会因为调整幅度过小而进步太慢。如果你每次复习完一道题后，你会根据答案和解析调整自己的理解和方法。  <br>+ 学习率大（比如 0.1）：每次做完一道题后，你会对解题方法进行很大的调整。比如，你可能会完全改变解题思路。优点是进步可能很快，因为你每次都在进行较大的调整。缺点就是可能会因为调整幅度过大而“走偏”，比如突然改变了一个已经掌握得很好的方法，导致之前学的东西都忘了。  <br>+ 学习率小（比如 0.0001）：每次做完一道题后，你只对解题方法进行非常细微的调整。比如，你发现某个步骤有点小错误，就只调整那个小错误。优点是非常稳定，不会因为一次错误而“走偏”，适合需要精细调整的场景。缺点就是进步会很慢，因为你每次只调整一点点。|
|**个人经验**|一般保持在 5 e-5（0.00005） 和 4 e-5（0.00004）之间，小数据集尽量不要用大学习率，不要担心速度慢，速度快了反而会影响微调跳过（如果是全参数微调，由于所有参数都会参与更新，为了防止原始知识被破坏，需要采用更小的学习率，一般比 Lora 的学习率小一个数量级）。|
|**显存影响**|几乎无影响。|
|**本次取值**|5 e-5 (即 0.00005)|

---

### 训练轮数（Number of Epochs）

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlPmegROcrXRXiaXdICp7eX6h5QrZjSUTOCgfStX8AvLe8Oqvanv3icrLw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

|**分类**|**说明**|
|---|---|
|**核心概念**|**训练轮数（Number of Epochs）**<br><br> Epoch 是机器学习中用于描述模型训练过程的一个术语，指的是模型完整地遍历一次整个训练数据集的次数。换句话说，一个 Epoch 表示模型已经看到了所有训练样本一次。|
|**通俗理解**|通俗来说，训练轮数就是我们从头到尾复习一本教材的次数。  <br>+ 轮数少：比如你只复习一遍，可能对书里的内容还不是很熟悉，考试成绩可能不会太理想。  <br>+ 轮数多：比如你复习了 10 遍，对书里的内容就很熟悉了，但可能会出现一个问题——你对书里的内容背得很熟，但遇到新的、类似的问题就不会解答了，简单讲就是 “学傻了“，只记住这本书里的内容了，稍微变一变就不会了（**过拟合**）。|
|**个人经验**|一般情况下 3 轮就可以，在实际开始训练后，只要 LOSS 没有趋于平缓，就可以再继续加大训练轮数，反之如果 LOSS 开始提前收敛，就可以减小 Epoch 或者提前结束。注意不要把 LOSS 降的过低（趋近于零），训练轮数尽量不要超过 10 ，这两种情况大概率会导致过拟合（把模型练傻）。一般情况下，数据集越小，需越多 Epoch；数据集越大，需越少 Epoch，LOSS 控制在 0.5 -> 1.5 之间。|
|**显存影响**|几乎无影响。|
|**本次取值**|3|

---

### 批量大小（Batch Size）

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlt2ibmUChK4g5TkaRiaveXZb1mTjIFnblJIc9YDKw4sEsTqJib2f6yHJNQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

|**分类**|**说明**|
|---|---|
|**核心概念**|批量大小（Batch Size） 是指在模型训练过程中，每次更新模型参数时所使用的样本数量。它是训练数据被分割成的小块，模型每次处理一个小块的数据来更新参数。|
|**通俗理解**|通俗来说，批量大小可以用来平衡复习速度和专注度，确保既能快速推进复习进度，又能专注细节。假设你决定每次复习时集中精力做一定数量的题目，而不是一次只做一道题。  <br>+ 批量大（比如 100）：每次复习时，你集中精力做 100 道题。优点是复习速度很快，因为你每次处理很多题目，能快速了解整体情况（训练更稳定，易收敛到全局最优）。缺点是可能会因为一次处理太多题目而感到压力过大，甚至错过一些细节（耗显存，泛化能力差，易过拟合）。  <br>+ 批量小（比如 1）：每次复习时，你只做一道题，做完后再做下一道。优点是可以非常专注，能仔细分析每道题的细节，适合需要深入理解的场景（省显存，易捕捉数据细节，泛化能力强）。缺点就是复习速度很慢，因为每次只处理一道题（训练不稳定，易陷入局部最优）。**|
|**显存影响**|影响明显，批量大小越大，对显存消耗越大。|
|**批量计算**|在实际微调任务中，**批量大小（Batch Size）**一般是由（单 CPU）**批处理大小（Per Device Train Batch Size）**和**梯度累计步数（Gradient Accumulation Steps）**共同决定的。（实际的批量大小：`per_device_train_batch_size * gradient_accumulation_steps`）|
||批处理大小（Per Device Train Batch Size），就是每个 GPU 一次处理的样本数（比如 per_evice_train_batch_size=8，就是 1 块 GPU 处理 8 条数据），在你只有一块 GPU，没有设定**梯度累计步数**的时候其实和**批量大小**就是一样的。|
||**梯度累计步数（Gradient Accumulation Steps）**<br><br>实际上是一个降低显存的优化手段，当我们显存不够用，但是又想采用大批量的时候，就可以加大梯度累积步数，从而实现分批次计算梯度然后累积进行更新。比如我们像设定**批量大小为 6**，但是我们的 **CPU 显存只能支持到 2** 了，这时候就可以把 **梯度累计步数 设置为 3 **，实际的步骤就是：  <br>+ 1. 先算 2 个样本的梯度，不更新参数；   <br>+ 2. 再算 2 个样本的梯度，不更新参数；   <br>+ 3. 再算 2 个样本的梯度，与前两次累积后一起更新参数。   <br>这样就能实现用小显存实现大 batch_size 的效果，类似于 “分期付款” 的效果。|
|**个人经验**|一般情况下，大 batch_size 需搭配大学习率，小 batch_size 需搭配小学习率，对于一些小参数模型、小数据集的微调，单 GPU 批处理大小（Per Device Train Batch Size）建议从 2 甚至是 1 开始（要想省显存就直接设置 1），通过调大梯度累积步数（比如设置为 4 或者 8）来加快模型微调速度。|
|**本次取值**|批处理大小：1、梯度累计步数：8|

---

### 截断长度（Cutoff length）

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlQkgfWwBV5DtqN9NuAcWn0UhsLy7SMdIjs85wfVND1JGZP4ZuEY7GtQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

|**分类**|**说明**|
|---|---|
|**核心概念**|截断长度（Max Length）决定了模型处理文本时能接收的最大 token 数量（token 是文本分块后的单元，如词语、子词）。它直接影响模型对上下文信息的捕获能力，同时制约计算资源的消耗。|
|**显存影响**|影响显著：截断长度每增加 1 倍，模型处理每个样本的显存消耗近似线性增长。例如：  <br>+ 截断长度 1024 时，1 块 32 GB 显存可能支持批量大小 16；  <br>+ 截断长度增至 2048，同显存下批量大小可能需减半至 8，否则会因显存不足报错。|
|**个人经验**|为啥一定要设置截断长度呢？假定你的显存处理的单条数据的极限大小为 4096 个 Token，你大部分的训练数据都小于 4096 个 Token，但是其中有一条是 5000 Token，那么训练到这条数据时你的显存一定会爆掉，从而中断整个训练，所以设定一个合理的截断长度是非常有必要的。  <br>在实际训练中，不能为了省显存而把截断长度设定的过小，比如你的训练数据平均大小为 2048 个 Token，那你把截断长度也设定为了 2048 个 Token，那有接近一半的训练数据都将是被截断、不完整的，这会大大影响训练效果。在显存条件允许的情况下，最好的截断长度当然是设定为数据集中单条数据的最大 Token 数，但现实可能不允许，你可以去计算你的数据集 Token 长度的 P 99、P 95 的值，来根据你的显存大小逐步调整，当然这也就意味着有 1%、5% 的训练数据是不完整的，你也可以在训练前把这部分较长的数据集剔除出去，这样可以减小对训练效果的影响。|
|**本次取值**|4096|

目前市面上有比较多成熟的工具来计算文本的 Token 数，比如 （[https://tiktokenizer.vercel.app/](https://tiktokenizer.vercel.app/)）：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlwkxRsCYiasSvz7y9QMwyySjd0Hr5DTLL2nmlNlcnjZmNz0zyBt6osicg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

如果使用的是 Easy Dataset 构造的数据集，我们也可以看到每条数据集的 Token 数（默认以 GPT 4 计算）：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlufaxQqcThddiawsfWa2GXILicN5XpuOAlfm1RQmWmwnYtoKovR1BffLw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

> 注意，不同模型在计算 Token 时的算法是不一样的，例如 Qwen、ChatGLM、Doubao 等，会针对中文语义特点优化分词逻辑，采用类似 BPE（字节对编码）的动态分词，但针对中文语境优化了 “语义块” 分割逻辑。例如，“人工智能” 会被识别为一个整体 Token（而非拆成 “人工”+“智能” 或单个字符），减少冗余 Token 数量。传统英文模型（如 GPT-3.5）：字符级或粗粒度分词，对中文缺乏优化，可能将汉字拆分为单个字符（如 “今”“天”“天”“气”...），所以在中文数据集的分词中，不同模型的差距还是挺大的。

在 LLaMA-Factory 中也提供了一个帮助批量计算数据集 Token 长度分布的脚本（scripts/stat_utils/length_cdf.py ），我们可以通过如下命令运行，注意把 model_name_or_path 、dataset 替换你自己的信息：

`torchrun --nproc_per_node=1        scripts/stat_utils/length_cdf.py        --model_name_or_path /root/autodl-tmp/Qwen2.5-7B-Instruct        --dataset security        --dataset_dir data        --template qwen   `

对于本地微调的数据集运行结果如下，可以发现 99.83% 的数据都是 < 4000 Token 的，所以我们本次设定的截断长度为 4096 ，如果你的设备性能足够电话，也可以直接设置为 5000：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlYIE4bHeiaPLByib3sm2GZ5UCr2wibmutkeOotQIFCdR48RpxvPfYiaGp7A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

---

### LoRA 秩（LoRA rank）

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlJicKQGIr87tE2rDtX6ZiaRichPRWn4Ebg0M0z2qDdgw05w3wBXwOsf3wQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

|**分类**|**说明**|
|---|---|
|**核心概念**|LoRA（低秩适应）中的秩（Rank）是决定模型微调时参数更新 “表达能力” 的关键参数。它通过低秩矩阵分解的方式，控制可训练参数的规模与模型调整的灵活程度。秩的数值越小，模型微调时的参数更新越 “保守”；秩的数值越大，模型能捕捉的特征复杂度越高，但也会消耗更多计算资源。|
|**通俗理解**|+ 秩低（如 4）：相当于用固定的几种 “思维模板” 学习新知识。比如学数学时，只记 3 种解题套路，遇到新题只能用这 3 种方法套。优点是学习过程很稳定，不容易学偏（过拟合风险低），且 “脑子”（显存）不费力；缺点是遇到复杂问题可能束手无策，因为模板太少（表达能力有限）。  <br>+ 秩高（如 64）：相当于掌握了 100 种解题思路，遇到新题可以灵活组合方法。优点是能处理更复杂的任务（比如生成风格更细腻的内容），模型微调的 “潜力” 更大；缺点是容易学 “乱”—— 可能把无关的知识强行关联（过拟合），而且太费 “脑子”（显存消耗显著增加）。|
|**个人经验**|+ 日常微调建议从 8-16 开始尝试，这是平衡效果与效率的常用区间。一般就是从 8 开始，如果微调完觉得模型没学会就调大，建议最低不要 < 8，小数据集不要调的过大。|
|**显存影响**|影响有限，例如对于 7 B 模型，秩从 8 提升到 64，大概仅会增加 2 GB 显存。|
|**本次取值**|8|

---

### 验证集比例（Val size）

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlbqgFgscWuUFQTkWOjdicuP57UlfXvSET221eQUh6NNzXCAPongjEibow/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

|**分类**|**说明**|
|---|---|
|**核心概念**|验证集比例是指从原始训练数据中划分出用于模型性能评估的子集比例（通常用 0-1 之间的小数表示）。这部分数据不参与模型训练，仅用于在训练过程中实时监控模型的泛化能力，避免过拟合。例如：验证集比例设为 0.2，表示将 20% 的数据作为验证集，80% 作为训练集。|
|**通俗理解**|一般来说，设不设置验证集是不影响模型的训练效果的，但是可以额外增加一条验证集的 LOSS 曲线，方便更好的观察模型微调的效果。|
|**个人经验**|+ 小数据（<1000 样本）：0.1-0.2，建议验证集样本数 ≥ 100（如 500 样本用 100 做验证）；  <br>+ 大数据（>10000 样本）：0.05-0.1，建议验证集样本数 ≥ 1000 即可（如 10 万样本用 5000 做验证）。  <br>比较复杂的任务可适当提高比例，因需要更严格监控各类别的拟合情况；比较简单单任务可适当降低比例，避免浪费训练数据。|
|**本次取值**|0.15（431 条数据）|

---

## 启动微调任务

### 预览命令

在以上配置完成后，我们配置一个最终微调后 Lora 适配器的输出目录，然后点击预览命令：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlbVHicTNic01Fbdrvdy80aWeJwfTD6jrrw3Ng7MNFicGQBt0NSrYFkEnDw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

点击后可以输出拼接好的 llamafactory-cli train 的各项参数：

`llamafactory-cli train \       --stage sft \       --do_train True \       --model_name_or_path /root/autodl-tmp/Qwen/Qwen2.5-7B-Instruct \       --preprocessing_num_workers 16 \       --finetuning_type lora \       --template qwen \       --flash_attn auto \       --dataset_dir data \       --dataset security \       --cutoff_len 4096 \       --learning_rate 5e-05 \       --num_train_epochs 3.0 \       --max_samples 100000 \       --per_device_train_batch_size 1 \       --gradient_accumulation_steps 8 \       --lr_scheduler_type cosine \       --max_grad_norm 1.0 \       --logging_steps 5 \       --save_steps 100 \       --warmup_steps 0 \       --packing False \       --report_to none \       --use_swanlab True \       --output_dir /root/autodl-tmp/models/security007 \       --bf16 True \       --plot_loss True \       --trust_remote_code True \       --ddp_timeout 180000000 \       --include_num_input_tokens_seen True \       --optim adamw_torch \       --lora_rank 8 \       --lora_alpha 16 \       --lora_dropout 0 \       --lora_target all \       --swanlab_project security007 \       --swanlab_mode cloud \       --val_size 0.15 \       --eval_strategy steps \       --eval_steps 100 \       --per_device_eval_batch_size 1   `

本次使用的微调参数的总结：

|**字段名**|**含义**|**具体值**|
|---|---|---|
|model|基座模型|Qwen 2.5-7 B-Instruct|
|stage|训练阶段|sft|
|template|模板类型|qwen|
|finetuning_type|微调类型|lora|
|learning_rate|学习率|5 e-5 (即 0.00005)|
|num_train_epochs|训练轮数|3.0|
|per_device_train_batch_size|每个设备训练批次大小|1|
|gradient_accumulation_steps|梯度累积步数|8|
|cutoff_len|截断长度|4096|
|lora_rank|LoRA 低秩矩阵秩数|8|
|dataset|数据集名称|security（2876 条数据）|
|val_size|验证集比例|0.15（431 条数据）|
|bf 16|半精度浮点数支持|true|
|flash_attn|Flash 注意力机制|auto|
|lora_alpha|LoRA 缩放因子|16|
|lora_dropout|LoRA 丢弃率|0|
|lora_target|LoRA 目标参数|all|
|lr_scheduler_type|学习率调度器类型|cosine|
|max_grad_norm|梯度裁剪范数|1.0|
|max_samples|最大样本数|100000|
|optim|优化器类型|adamw_torch|
|packing|是否进行序列打包|false|
|plot_loss|是否绘制损失曲线|true|
|preprocessing_num_workers|预处理工作线程数|16|
|save_steps|模型保存间隔步数|100|
|warmup_steps|预热步数|0|
|use_swanlab|是否使用 SwanLab|true|
|swanlab_mode|SwanLab 模式|cloud|

---

### 训练过程

然后我们点击开始训练：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlhZkyIvibz0miaJYq10Vkf8iabtLKB1dPaShFezLfUeHOXojER77CHQIYA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

回到终端可以看到整个训练过程：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlW90u9aoEYQFbibTiaichtPqojejhpA1OMdwTG3ibnWcTAic4NE7l6ZW4a1g/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

回到 LLaMA Board 页面，我们也可以看到具体的进度和 LOSS 曲线：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUloJicrcE3kCcc6Pq09dsgw8ffG6QDdg5icdfibCqpTkut16nrTBic1BeT7A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)

---

这里我们看到总的进度为 456 步，这个可以根据我们之前的微调参数计算出来：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlhPWPjeTY0gtBwMhJJwpyZak50Xd3hqCaicoNDer5WrefuxwhsrodhKw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)

> 总步数由“每轮处理次数”和“轮数”决定，而前者取决于“数据量”“批次大小”“梯度累积”的组合。 向下取整意味着最后一次若凑不满完整的更新批次，就跳过不处理。

---

### 显存消耗估算

本次微调使用的硬件配置如下：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlGvicznxvpOUn3NCWaQsOnkHHQa6DtibaOeajccqcX4BLIKVkQ6C4hPGQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)

在微调过程中，我们可以通过 nvidia-smi 命令查看 GPU 和显存的占用：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlibxsDRjFibnXSFTCug5wlpxZiba6C95ZhHBuibGFMQw7TicyQWA5dgjAibGA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)

这里我用了两张 48 G 显存的卡，我们看到在微调过程中，两张卡分别使用了 G 的显存。如果你不知道一个微调任务需要什么样的硬件配置，我们可以通过当前选择的微调参数来对可能的显存占用做个简单的估算（估算方法与实际严谨的计算方法可能存在偏差，但大体接近）。在模型微调（Lora）过程中，显存占用主要由**基础模型权重、激活值、框架开销、LoRA 适配器等**几部分构成：

---

**基础模型权重（Base Model Weights）**

预训练模型的参数矩阵，可以简单理解为就是选择的预训练模型占用的显存大小。

- **计算方法：显存占用 = 模型参数数量 × 单个参数的字节数**
    

- FP 32：32 个二进制位（4 字节，1 字节 = 8 位）
    
- FP 16：16 个二进制位（2 字节）
    
- BF 16：16 个二进制位（2 字节，指数位同 FP 32 ）
    
- INT 8：8 个二进制位（1 字节）
    
- INT 4：4 个二进制位（0.5 字节）
    
- INT 2：2 个二进制位（0.25 字节）
    

- 上面我们提到过常见的模型精度下，单个参数的显存占用：
    
- 在本次微调中，我们选择的模型是 Qwen 2.5-7 B-Instruct ，参数量为 70 亿，计算精度为 BF 16 ，所以预估的显存占用为 **70 亿 × 2 byte ≈ 140 亿字节（14 GB）。**
    

---

**框架开销（Framework Overhead）**

LLaMA Factory 底层使用的深度学习框架（如 PyTorch）本身的显存占用，包括：张量缓存、线程资源、内核调度开销、自动微分图结构等等。

- **计算方法：比较难精确计算；**
    
- **估算方法：一般占用不会太大，我们可以先默认估算消耗 1 G**
    

---

**Lora 适配器（LoRA Adapters）**

在 Lora 微调中，不会直接修改原始模型的庞大权重，而是通过插入轻量级的 “Lora 适配器模块” 来学习模型微调所需的变化，这部分可以理解为就是这个小模块的显存占用。

- **计算方法：显存占用 = LoRA 层数 × 秩（Rank）×（输入维度 + 输出维度）× 2 B**
    
- **估算方法：和我们上面提到的 LoRA 秩的大小呈正相关，一般占用不会太大，在常规配置下不会超过 0.5 G，保守估计 0.5 G**
    

---

**激活值（Activations）**

前向传播过程中各层的输出张量（如隐藏层状态、注意力矩阵等）。简单理解就是模型 “处理数据时产生的所有中间结果”，这些结果需要临时存在显存里，主要受训练时同时处理的数据量大小的影响。

- **计算方法：显存占用 = 批量大小 × 序列长度 × 隐藏层维度 × 模型层数 × 单个元素字节数**
    
- **估算方法：单次处理的 Token 量每增加 1 K，显存约增加 2.5 G**
    

激活值的显存占用和我们上面提到的单 GPU 批量大小和数据集的截断长度成正相关，可以说除去以上三个值，剩下的就是激活值的占用，为了找到估算规律，我们做个简单的实验，在以上微调配置不变的情况下，我们设置一下不同的阶段长度和批量大小（单 GPU）：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUltKkHnwJFIBlUJwA8IiblgaPtibQsBaT79ZtrL3Qb17bRId3jdx7Wkxcg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUl0zee6t0zoJzZkbP9EyKU6SVLXoLZnAibQiaCyTpHaKUPia3yslNaEnVqQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)

可以发现，当截断长度、批量大小分别设定为 1024 * 2 和 2048 * 1 时显存消耗是是完全相同的，激活值当显存占用就是和单个 GPU 单次处理的 Token 量是正相关的，**单次处理的 Token 量每增加 1 K，显存约增加 2.5 G。**

---

根据我们前面的微调配置，显存占用估算如下：

|**显存消耗分布**|**对应设置**|**估算方式**|
|---|---|---|
|**基础模型权重**|Qwen 2.5-7 B-Instruct|+ 模型：（70 亿参数），精度：BF 16（2 字节/参数）   <br>+ 计算：70 亿 × 2 Byte = 140 亿字节 = **14 GB**|
|**框架开销**|-|+ 简单估算：**1 GB**|
|**LoRA 适配器**|LoRA 秩=8|+ 简单估算：**0.5 GB**|
|**激活值**|单设备批量大小=1  <br>截断长度 = 4096 Token|+ 单次处理 Token 量：1 × 4096 = 4096 Token（4 K）   <br>+ 估算：每 1 K Token 约增加 2.5 GB 显存   <br>+ 计算：4 × 2.5 GB = **10 GB**|

`基础模型权重：14 GB     框架开销：1 GB     LoRA适配器：0.5 GB     激活值：10 GB     ---------------------     总计：14 + 1 + 0.5 + 10 = **25.5 GB**   `

这里可以使用我开发的这个智能体进行简单估算（后续 LLaMA Factory 也会支持在 webui 中显示预估显存消耗）：

[https://www.coze.cn/s/3d3maKRCIlk/](https://www.coze.cn/s/3d3maKRCIlk/)

注意这里的消耗建立在我们没有主动启用其他额外的优化手段的情况下，如果你的硬件配置有限，可以尝试开启以下两个优化手段。

---

### 显存优化技巧：liger_kernel

以上显存计算方式中，模型权重、Lora 适配器、框架消耗这些已经无法再进行优化了（不考虑量化的情况下），只有激活值的消耗还有一定的优化空间，这里我们可以启用 liger_kernel，在前面的加速方式章节，我们简单提到过，这里就派上用场了。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlZcUJDGvDL4zTlcibBgECQ21co5MFd8RUypWVibXGwsTMjW1rBHIibdwicQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=18)

Liger Kernel 将 Transformer 中的关键操作（如 RMSNorm、RoPE、SwiGLU、CrossEntropy）重写为 Triton 内核，并通过内核融合将多个操作合并为一个计算步骤，避免了传统方法中多次内存读写带来的冗余存储，在模型微调过程中，可以显著降低数据处理相关的内存占用。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlVZPvr5zzKicR3g09NVsWnbLZicicNKK5AWhCO8Emj51h9kar2DNib8d46A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=19)

下面是我针对先前的几组配置，在开启 Liger Kernel 后的显存消耗实验：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlU6Fbg9wDKrVLev5xEVicGia241eDZgthicTSewSrgEic6VXJpkuOzBkZXQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=20)

可以发现，开启 liger_kernel 后，在 2 K Token 和 4 K Token 下的消耗分别为 16.6 G 和 17.9 G，每增加 1 K Token，显存约增长 0.6 G，相比先前的 1 K Token 2.5 G ，下降了 76% 。

注意：在默认情况下，LLaMA Factory 会启用 flash attention 进行加速，而开启 liger_kernel 后 flash attention 依然是会开启的，两者并不是互斥的关系。

---

### 分布式显存优化：DeepSpeed

以上的显存消耗我们都是基于单卡（单 GPU）的维度计算的，在实际任务中，单张消费级显卡是很难支撑大参数或者大批量的微调任务的，所以一般我们都会采用多卡分布式训练。

假定在以上的微调任务中，我们将截断长度改为 2048、批量大小改为 3，那么预估的显存消耗（ liger_kernel 未开启）为 14+1.5+15=30.5 G，假定我们现在有两张 RTX 4090 D(24 GB) 的卡，总显存是 48 G，理论上是完全可以 Cover 这个任务的，但如果你还是使用以上的配置，一定会爆显存：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUl6ibmeGS51twJHn8aCN7kX18VRtz1ZWnW3Rt5u6G2qibZEjxsUqfAvBfQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=21)

这里可能会涉及到大家的一个常见物误解：我们虽然使用了多张卡进行训练，但却并不是真正意义上的内存平摊，模型参数和优化器状态仍需完整存储在单张卡上，无法突破单卡内存限制，可以理解为每张卡上的任务还是互相独立的，互不干扰，这样可以有效提升微调的速度，但每张卡都需要承担最大的显存消耗。想要真正的实现多卡的显存平摊，可以启用 DeepSpeed Stage 3。

DeepSpeed 是由微软研发的一款深度学习优化库，旨在简化分布式训练与推理过程，DeepSpeed 通过 ZeRO 技术将模型状态分片到多卡，消除显存冗余，并结合混合精度训练和并行策略（张量/流水线并行），使多卡训练时单卡显存占用随 GPU 数量显著降低，支持更大规模模型训练。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUlgiaJhO5CfyicR4YiaicbdMck6icaiaL3RecD2shjZ0icfIWHVicvG5XqFp5Fgw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=22)

在 LLaMA Factory 中需要配置 DeepSpeed Stage：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUluHdkDzCiaic5CCYJEgCnTT1Q51hweodxCdPKiaTazvuuibhoFdh3wfNaxQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=23)

我们发现 DeepSpeed Stage 配置里有四个选项：0（默认）、1、2、3，这里我们用一个通俗的例子来让大家更容易理解：如果把多卡训练比作「多人包饺子」，那 DeepSpeed Stage 就是「分工方案」，假设用 4 张卡训练大模型，对比不同 Stage 的「分工模式」：

1. **Stage 0（不开启 Stage）：每张卡独立全包**
    

- **做法**：每张卡都存完整的模型参数、算前向、算反向、更新参数。
    
- **类比**：4 个人各自包 4 盘饺子，每人都要准备 1 份馅、1 份面皮。
    

---

1. **Stage 1：参数共享，计算独立**
    

- **做法**： 参数只存 1 份，放在所有卡上（共享馅），但每张卡各自算前向、反向、更新参数。
    
- **类比**：4 个人共用 1 份馅，但各自擀面皮、包饺子。
    

---

1. **Stage 2：参数和优化器分离**
    

- **做法**： 参数存 1 份（共享馅），优化器状态（如 Adam 的动量、方差）单独放在部分卡上（比如 2 张卡存优化器，2 张卡算前向）。
    
- **类比**：2 个人专门管馅和调料（优化器），2 个人负责擀面皮、包饺子。
    

---

1. **Stage 3：极致分工，参数「流动」**
    

- **做法**： 把模型按层拆成多段（比如 4 张卡各管 1 / 4 层），参数不再固定在卡上，而是按计算流程「流动」：
    

- 卡 1 算前 1-4 层，用完参数就传给卡 2；
    
- 卡 2 算 5-8 层，用完再传给卡 3，以此类推。
    
- 优化器状态和参数都只存 1 份，放在「参数服务器」卡上，计算卡用完就还回去。
    

- **类比**： 包饺子流水线：卡 1 负责剁馅，卡 2 负责擀面皮，卡 3 负责包饺子，卡 4 负责煮饺子，材料（馅、面皮）在流水线上传递。
    

---

- **不开启 Stage（Stage 0）**： **优点**：简单，卡之间不用传数据，速度快。 **缺点**：显存占用大。
    
- **开启 Stage 1/2/3**： **优点**：用「分工」换显存，让更大的模型能在有限显卡上训练。 **缺点**：Stage 越高，卡之间通信越多，可能降低训练速度。
    

下面是我们在开启 DeepSpeed Stage 3 的情况下两张卡的显存消耗：分别是 16.3 G，总共消耗 32.6 G，比我们预估的 30.5 G 要多出 2 G，这些就是多卡通信的额外开销：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/e5Dzv8p9XdTibGOAw56Af6iafZjz1icymUl1bFrXV8Y3jic208untiapQ3EBCKVPiccIIWibUepjAplBZ4qJ0At3WHTibA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=24)