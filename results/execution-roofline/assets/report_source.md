# Driving-VLA：从一次更新到 Roofline

Forward / backward 耗时、延迟隐藏、通信与计算强度的实测证据｜2026-10-28

## 01 从直觉开始：为什么工作量减半，时间没有减半

把一次训练更新想成一条流水线：CPU 准备并下发工作，GPU 执行算子，反向传播产生梯度，各卡同步梯度，最后更新参数。**这条流水线的速度取决于最长的依赖路径。** 增加 GPU 可以减少每张卡处理的样本数，但不会自动减少 Python 调度次数、算子数量、每次启动的固定成本，也不会消除等其他卡的时间。

本项目 global batch 固定为 64。16 卡时每卡处理 4 个样本，32 卡时每卡处理 2 个样本。实测平均 step 从 **260.82 增到 285.42 ms**，吞吐从 **245.38 降到 224.23 samples/s，下降 8.62%**。这里的吞吐是已归档的一小时训练中、去掉 warmup 的 step 统计口径；含尾部同步的更新墙钟约为 261.53 / 286.15 ms。[E1]

单卡对照给出关键线索：checkpoint 开启时，microbatch 2→4 的 step p50 只从 **210.66 增到 221.28 ms，增加 5.04%**，样本数却翻倍。GPU 上矩阵计算确实变多了，但大量固定的执行成本可以与新增计算部分重叠。反过来把 batch 从 4 减到 2，这部分固定成本仍然存在。[E2]

[[CHART:scaling|图 1 · 扩卡后的快步与慢步|一小时未插桩训练。慢步阈值为 250 ms；分类是描述性统计，不是因果归因。]]

更有解释力的是分布：32 卡的快步均值其实更短，**223.66→215.16 ms**；慢步比例却从 **37.75%→71.58%**。按相同阈值做对称分解，频率变化贡献 +33.26 ms，组内时间变化贡献 −8.66 ms，两者恰好组成 +24.60 ms 的均值差。**这是一个统计恒等式，不能把 +33.26 ms 直接称为“GC 成本”。** 它指明接下来应追踪尾延迟与同步，而不只是网络带宽。

## 02 先把“时间”说清楚，才能读懂 trace

同一个 forward 可以有多种时间。CPU 范围表示主机侧从进入到退出的跨度；CUDA event 表示同一 stream 上两处标记之间经过的时间；GPU kernel duration 才表示某个 kernel 在设备上执行多久。CPU 不断给 GPU 下发工作时，这三种时间会重叠。

| 测量口径 | 包含什么 | 不能据此声称什么 |
|---|---|---|
| 完整更新墙钟 | 主机调度、GPU 完成、通信、更新，以及纳入口径的尾部工作 | 不能当作纯计算时间 |
| CPU annotation span | Python/C++ 路径、launch、可能的阻塞或 GC | 不能当作 CPU 忙碌时长 |
| CUDA event elapsed | stream 进度，包含下发不及时形成的空隙 | 不能当作 kernel duration |
| Kernel 区间并集 | 至少有一个已记录 kernel 执行的时长 | 不能当作 SM 利用率或 Tensor Core 利用率 |
| NCCL 区间 | collective kernel 驻留，包括等 peer 到达 | 不能全部算作数据传输成本 |
| `max_rank(phase)` | 每个阶段在所有 rank 中各自的最大值 | 不同阶段的最大值不能相加成同一 rank 的 step |

原训练代码对 `[step, data_wait, forward, backward, optimizer, H2D]` **逐字段做 MAX all-reduce**。最慢 forward 和最慢 backward 可能属于不同 rank；因此 16 卡的 forward 平均 122.12 ms 与 backward 平均 190.66 ms，不能相加并与 260.82 ms 比较。[E1,E7]

本报告重新解析历史 16/32 卡各 16 步 rank0 trace，以及单卡 6 个配置各 4 步 trace。GPU kernel 通过 CUDA runtime/driver 的 **correlation ID** 找到 host launch，再用 launch 时刻归入 forward、backward、optimizer。关联覆盖率均为 **100%**。归属由下发来源确定，而不是用 GPU 执行时刻猜测；已下发的 forward 工作可能在 CPU 进入 backward 后仍继续执行。

[[CHART:timeline32|图 2 · 32 卡 rank0 的一条真实更新时间线|Profiler 插桩下的代表性迭代；CPU 是范围跨度，GPU 是 kernel 区间。仅为显示合并不超过 30 μs 的小间隙，统计值仍使用精确并集。]]

这条时间线不是无插桩的平均 step。历史 trace 的 kernel 并集/跨度为 **16 卡 2.593/4.895 s，32 卡 1.480/4.901 s**。32 卡每卡 batch 更小，设备上执行的工作明显缩短，但整个 trace 跨度相近。剩余区间里可能有主机下发、同步、设备 memcpy 和 profiler 成本，不能统称为“网络等待”或“GPU 闲置”。

## 03 Forward：大矩阵算得不错，整条链路仍有空隙

此训练已经缓存相机与文本特征，实际输入是 `inputs_embeds`。计时路径包括语言骨干、LoRA、策略 head 和 loss；**不包含在线视觉编码、teacher 生成、token embedding 查表或词表 lm_head**。模型配置为 36 层，hidden 2048，MLP intermediate 11008，16 个 query heads / 2 个 KV heads，head dimension 128。[E2,E6,E7]

缓存有 173 个有效 token，但实际张量长度为 **256**。普通 Linear 接收到 `[B,256,H]` 后处理的是 `M=B×256` 行，不能把 FLOPs 按 173 行直接折算；attention 的 mask、因果三角和后端融合需要另行分析。只看“有效 token 数”容易低估 padding 成本。

| 算子 | GEMM：`[M,K]×[K,N]` | B=2 的逻辑 FLOPs | 主要工作 |
|---|---|---:|---|
| Q / O projection | M=512, K=N=2048 | 4.295 GF | 隐藏维度变换 |
| K / V projection，各一个 | M=512, K=2048, N=256 | 0.537 GF | GQA 压缩 KV 维度 |
| MLP gate / up，各一个 | M=512, K=2048, N=11008 | 23.086 GF | 扩展中间维度 |
| MLP down | M=512, K=11008, N=2048 | 23.086 GF | 压回 hidden |
| LoRA A，Q 分支 | M=512, K=2048, N=8 | 0.0168 GF | rank-8 降维 |
| LoRA B，Q 分支 | M=512, K=8, N=2048 | 0.0168 GF | rank-8 升维 |

本轮新微测只使用 **1 张 H20，78 SM，PyTorch 2.9.1**，按实际项目的 shape 和转置布局构造随机 BF16 矩阵；这些值验证性能，不验证驾驶质量。测量了 B=2/4/8 的 24 个 shape，加一个 8192³ dense GEMM，共 25 组。每组包含 3×24 次 warm CUDA-event 计时、3×12 次冷缓存近似计时，以及 warm/cold 各 4 次带 launch 关联的 CUPTI trace。[E5]

$matrix_table

**MLP up 的设备执行时间随 batch 几乎线性增加，而完整 checkpoint-on step 没有。** B=2→4 时，MLP up 的 kernel 时间约 **$mlp2_us→$mlp4_us μs**；同一个大 GEMM 已达到 **$mlp2_tf / $mlp4_tf TFLOP/s**，接近本机实测参考 **$ref_tf TFLOP/s**。因此“每卡 batch 小，所以所有矩阵乘法都很差”并不符合证据。更准确的描述是：**大 MLP GEMM 计算效率较高；薄矩阵、逐元素操作和主机调度构成另一层成本。**

[[CHART:dispatch|图 3 · 设备执行与 event 窗口的差距|B=2，warm。设备值是相关 kernel 区间并集的中位数；event 是 3 个连续调用窗口的均值。两组独立采样，不应把差值全部归因于某一函数。]]

以 LoRA A 为例，设备执行中位数 **$lora_a_us μs**，连续 event 平均约 **$lora_a_event μs**。这里还有 dispatch 间隙和 API/stream 排程的影响。冷缓存近似 event 又比 warm 增加约十几微秒，但关联 kernel 的冷/热差异通常小得多。**CPU 在 `start.record()` 后下发不够快，也会被 event 捕捉到；不能把这段差值解释成 HBM 读取变慢。**

冷缓存近似使用 192 MiB 的流式读写清理缓存，清理本身不计入 GEMM timer；但没有 cache hit / DRAM counters，不能证明完全冷缓存。输入与权重反复复用的 warm 场景也比真实 36 层轮换的权重更容易命中缓存。

## 04 Backward：冻结权重为什么仍然昂贵

对 `Y=XW`，反向传播通常包含两次乘法：

```text
dX = dY × Wᵀ
dW = Xᵀ × dY
```

本项目基座权重冻结，所以省掉的是 **dW**。但是更早层的 LoRA 参数要学习，梯度还需要穿过后续冻结层，**dX 不能全部省掉**。只有位于不需要输入梯度的图边界处，某些 dX 才可能被跳过。故“只训练 238 万参数”不意味着反向只处理 238 万参数量的计算。

LoRA 的形式是 `Y = XW + s·(XAᵀ)Bᵀ`。A/B 的梯度需要 `dA,dB`；输入梯度还要汇合基座分支与 LoRA 分支。Q/V 每层各有 rank-8 adapter。梯度、head 与 optimizer state 多为 FP32，而基座/autocast 为 BF16。**FP32 参数不等于该矩阵乘法在 FP32 Tensor 路径执行**：autocast 可将 GEMM 的输入转换为 BF16。源码中的 `values.to(float32)`、随后 autocast、输出回转，以及梯度累积，会产生实际 copy/cast 和 autograd 节点；性能不能只按参数存储 dtype 推断。[E7]

checkpoint 还给 backward 加了一段 **重跑 forward** 的工作，以恢复没有保留的中间激活。项目使用 `use_reentrant=False`，默认 early stop 在需要的激活恢复后停止；不应该机械地把整层完整 forward 都算一遍。[R2]

[[CHART:kernels|图 4 · 关闭 checkpoint 去掉了哪里的 kernel|同机单卡 B=2；每组 trace 4 步，所有 kernel 都关联到 host 下发阶段。数量包含 optimizer 和阶段外 kernel。]]

单卡 B=2 的实测最容易核对：

| 阶段 | CP on：kernel/步 | CP off：kernel/步 | CP on：GPU kernel 并集/步 | CP off：GPU kernel 并集/步 |
|---|---:|---:|---:|---:|
| Forward | 2246 | 2246 | 30.69 ms | 30.66 ms |
| Backward | 5254 | 3166 | 59.97 ms | 35.91 ms |
| 全部 kernel | 7541 | 5453 | 90.95 ms | 66.85 ms |

重算增加约 **2088 个 kernel/步、24.06 ms 的 backward kernel 执行**。Forward 的设备工作基本不变，但 profiler 中 CPU forward span 从 107.94 降到 77.16 ms，CPU backward span 从 211.07 降到 102.04 ms。CPU span 包括插桩与可能的等待，不能拿这两个范围的差值当作未插桩的“Python 独占成本”。它们与 kernel 数、独立无插桩吞吐对照一起，支持 checkpoint 引入大量主机和反向重算工作。[E2,E5]

无插桩单卡对照中，关闭 CP 在 B=2/4/8 的吞吐收益分别为 **2.035× / 1.737× / 1.379×**，显存峰值分别 **7.85 / 9.78 / 13.59 GiB**。同样本梯度与轨迹对照通过了原有数值检查；这些是局部执行等价与性能证据，不是收敛或 ADE/FDE 的新测量。[E2]

## 05 延迟隐藏有三个层次，要分别判断

**第一层是 SM 内部：一个 warp 等数据，scheduler 换另一个已就绪的 warp 执行。** GEMM 常把下一块 A/B 的搬运与当前块 Tensor 运算流水化。Hopper 的异步拷贝、TMA、shared-memory buffering 可帮助构造这种流水线，但“硬件支持”不能证明当前 kernel 已有效使用。要判断，应看 eligible warps、issue active、Tensor pipe 与长记分牌等待等指标；本环境没有 NCU，暂缺这些 counters。[R3,R4,R6]

本轮 CUPTI 给出了 launch 的资源参数：代表性 nvjet GEMM 用 **384 threads/block、168 registers/thread、约 160–226 KiB shared memory/block**。结合设备实际 65536 registers/SM、233472 B shared memory/SM，寄存器和 shared memory 都把并驻限制在大约一个 block/SM。384 threads 是 12 warps，按 64 warp 容量算，资源上限约 **18.75%**。寄存器分配粒度等细节还会影响严格 occupancy 计算。[E5,R4]

**低 occupancy 并不自动等于低 Tensor Core 利用率。** 本机 8192³ GEMM 的主要 grid 为 `[1,78,1]`，同样使用这类资源配置，仍达到实测参考 $ref_tf TFLOP/s；持续的片内流水线和每 warp 的工作量同样重要。CUPTI 字段 `est. achieved occupancy %` 在这些 kernel 中甚至显示 0，它是工具根据资源/grid 估算的元数据，不是实测 scheduler 利用率，不能拿来宣布“GPU 利用率为零”。

$resource_table

B=2 的 LoRA A 主要 grid 为 64 个 CTA，小于 78 SM；B=4 的实现变成 16 个 CTA，说明库会换 kernel/拆分策略，不能把 CTA 数简单当作 M×N tile 数的固定函数。窄维度 N=8 或 K=8 也会降低有用算术量。**这些是资源/形状约束证据；究竟是内存延迟、访存吞吐还是指令依赖，需要 counters 才能进一步定量。**

**第二层是 CPU 与 GPU 之间：CPU 下发后续工作时，GPU 执行队列里的当前工作。** Batch 增大，大 GEMM 执行时间增加，CPU 的部分固定调度就更容易被覆盖；小 batch 时，设备更快消耗队列，dispatch/解释器/GC 停顿更容易露出。本轮 kernel 微测与原单卡 batch 对照分别提供了设备时间和完整 step 的证据。

**第三层是 DDP：后面层的梯度先准备好并开始 all-reduce，同时继续计算前面层的梯度。** 是否能隐藏通信由 bucket 准备时刻、剩余计算、stream 依赖和资源竞争共同决定。画出两条同时运行的 stream 还不够：要看它们的区间交集，以及通信完成是否推迟 step。[R1]

```text
资源内部流水线：等待加载(next tile) ↔ 计算(current tile)
主机/设备流水线：下发(next kernel) ↔ 执行(current kernel)
分布式流水线：all-reduce(ready bucket) ↔ backward(remaining layers)
```

## 06 通信 overhead：传得慢、到得晚、没有重叠是不同问题

所有 trainable 参数共 **2,382,640** 个，按 FP32 梯度约 **9,530,560 B = 9.089 MiB**；这不是几十亿基座权重的同步量。DDP 配置 `bucket_cap_mb=25`，但 dtype、首轮桶与重建会影响实际分桶，最终以 trace 为准。[E7]

在 16/32 卡各 16 步 rank0 trace 中，**每步都恰好出现一个在 backward 内下发的 NCCL gradient collective**。设备执行区间中，它与所有非 NCCL kernel 的交集均为 **0 ms**。这证明在这些采样里，梯度通信没有被本 rank 的计算 kernel 覆盖；大部分参数分布于全部 36 层，最后一个必需梯度准备好时，能剩下的计算很少。

$communication_table

数值上，gradient NCCL 平均约 **0.810 / 0.800 ms/步**；所有 NCCL 区间平均约 **0.854 / 0.863 ms/步**。NCCL 总量还包含尾部小统计和 broadcast，少量在 forward 期间的初始化/桶相关 collective 不能称为梯度数据量。这里按下发阶段分类，并未用缺失的 payload 元数据臆测每次调用的字节数。

**0.8 ms 是“该 rank 上 collective kernel 驻留多久”，并不是“跨机传 9 MiB 的纯时间”。** 更早到达的 rank 可能多等，晚到的 rank 可能几乎不等。单看 rank0 和 16 个插桩步骤，无法排除整小时其他 rank 的 arrival skew 或拥塞，也不能把原 +24.60 ms 精确切成网络、GC 和计算比例。

前一轮强制跨物理机的 2-rank、9.089 MiB all-reduce 均值约 **0.346 ms**，后端由 NCCL 日志确认是 IB/RoCE、4 channels。它证明这条小负载跨机路径可以很快，但不是 32-rank ring 的上界。历史 16 卡为 HL/2 节点、IB/RoCE/8 channels；32 卡为 LF/4 节点、soft_bonding_v8/16 channels，环境与规模同时变化，不能单独判定 plugin 优劣。[E3,E4]

对理想 ring，简单模型为：

```text
t_ring ≈ 2(P−1)·α + [2(P−1)/P]·G/β
```

P 是 rank 数，G 是梯度字节，α 是每阶段启动/同步延迟，β 是有效传输带宽。**这是解释小消息为什么敏感于延迟的模型，不是已确认的实际算法。** 多 channel、分层算法和 fabric 拓扑都可能改变它。梯度量保持 9 MiB 时，扩卡不会让 G 同比例变大，但 rank 数、arrival skew 与启动阶段可能增加。

CPU 并未在每个阶段显式 barrier。可是下一次 gradient collective 仍需全体 rank 参与；某个 rank 的 forward GC 可以表现为别的 rank 的 backward 完成更晚。此前一小时的逐字段 MAX 统计中，**98.67% / 99.48% 的慢步伴随 forward >120 ms**。这与单卡 GC 控制吻合，但历史 trace 缺逐 rank GC 时间戳，尚不能把每一次分布式慢步逐一配对。[E1,E8]

[[CHART:gc|图 5 · 相同 checkpoint 配置，只改变 GC|单卡真实样本，repeat 0 的 96 步；叠加 generation-2 GC 所在步的 forward GC 时间。完整三窗口数据见证据 CSV。]]

单卡独立因果试验中，CP on 的 288 步产生 **18 次 generation-2 GC**，每次 **102.8–159.1 ms**，18 次全部与 forward 尖峰配对；thread CPU 约等于 GC 墙钟，支持它是本地 GC 执行而不是等待网络。仅关闭自动 GC，尖峰 **18→0**，step p95 **324.03→217.89 ms**，吞吐 +6.02%。进一步关闭 CP，吞吐约 **2.087×**，显存仅 7.84 GiB。[E8]

这给出了**本地停顿机制的直接因果证据**。在 DDP 中，最快的 collective 也不能绕过尚未产生梯度的 rank；通信本身快与系统存在同步长尾可以同时成立。长期直接禁 GC 还可能积累循环引用，应测内存并在明确窗口做回收；当前优先级更高的方案是减少产生大量重算图的 CP。

## 07 Roofline：先理解坐标，再读实测曲线

Roofline 把性能问题压到两个坐标。横轴是 **计算强度 I：搬运每 byte 数据完成多少 FLOP**；纵轴是 **吞吐 P：每秒完成多少 FLOP**。一种情况是算术单元已很忙，继续增加数据复用帮助有限；另一种情况是搬数据限制了运算，要提高复用或减少流量。[R5]

```text
P ≤ min(P_peak, BW × I)
I_HBM = FLOPs / 实测 HBM bytes
拐点 I* = P_peak / BW
```

本环境没有 NCU 可执行文件，**并未测得实际 DRAM bytes、Tensor pipe 利用率、eligible warp 或 memory stall counters**。也没有采用未经一手资料确认的 H20 厂商峰值。本图用 8192³ dense BF16 GEMM 的 **$ref_tf TFLOP/s** 和 256 MiB streaming copy 的 **$ref_bw TB/s 有效逻辑读写带宽**画一条**经验参考线**，拐点约 **$ridge FLOP/byte**。[E5]

这条线不是架构上限。不同 shape 的 kernel 可能更快，copy 的逻辑读写量也不等于直接计数的 HBM traffic。**本图不能计算“利用了多少理论峰值”**；它的价值在于把本机可达性能、项目算子 shape 和主机/设备差距放到同一尺度。

[[CHART:roofline|图 6 · 本机经验 Roofline 与项目 GEMM 实测点|横轴为单遍逻辑计算强度，不是真实 HBM intensity；纵轴为相关 GPU kernel 区间并集推算的逻辑 TFLOP/s。warm/cold 为独立采样中位数。]]

图中横轴计算采用 BF16 输入/输出、一次读 A/B 与一次写 C：

```text
F = 2 M N K                         [乘加计 2 FLOP]
D_alg = 2(MK + KN + MN) byte
I_alg = MNK / (MK + KN + MN)
P_kernel = F / t_kernel
```

实际 HBM 数据可能因 cache 命中而少于 D_alg，也可能因重复加载、workspace/归并、输出重访而更多。尤其 thin GEMM 与 split-K 可能有不同实现。**I_alg 是 shape 的属性，I_HBM 是测量和缓存条件的属性；两者不能替换。** 这里还存在 cold_kernel TF/s 比 warm 略高的点，这是短采样波动，不能宣称冷缓存更优。

| B=2 算子 | I_alg，FLOP/byte | 相对经验拐点 | 设备实测表现 |
|---|---:|---|---|
| Q forward | 341.33 | 位于算术参考平台右侧 | $q2_tf TFLOP/s |
| MLP up | 394.91 | 位于算术参考平台右侧 | $mlp2_tf TFLOP/s |
| KV forward | 157.54 | 仍位于平台右侧 | $kv2_tf TFLOP/s；shape 更薄 |
| LoRA A / B，Q 分支 | 7.85 | 位于带宽参考斜线一侧 | $lora_a_tf / $lora_b_tf TFLOP/s |

MLP up 的点接近经验平台，说明大 Linear 值得保持但不应首先把它当作低效元凶。LoRA 点远低于平台、也低于逻辑带宽参考，表明薄 shape 的有用算术吞吐很低；**这并不能仅凭图确认 HBM 饱和**。短 kernel 的启动与尾部、warp/CTA 供给、窄 K/N、cache 和未计入的真实 traffic 都可能让点位低于曲线。

“整个 step 达不到峰值”还要加一层。kernel 级 Roofline 不包含 Python/GC，也不包含没有 kernel 在执行的下发间隙。完整 step 的逻辑吞吐可写为 `F_step / T_wall`；它通常低于 GEMM 吞吐，原因可能是无算术工作时间与其他算子，并不意味着 Tensor Core 的每次执行都很慢。

## 08 从配置推导算力预算：本项目不能直接套 6NT

通常全量训练的 `6NT` 近似把 forward、dX、dW 都包括了。这里 N 若取模型标称参数量，还会把没有执行的 embedding/词表部分混进去；基座冻结又省掉 dW。更好的预算是逐项求和。

```text
每层 active Linear 权重
= 2H² + 2H·KV + 3H·F
= 77,070,336
36 层合计 = 2,774,532,096

F_linear_forward = 2 · B · S_physical · Σ|W_linear|
F_linear_dx ≈ F_linear_forward
F_train_noCP ≈ 2F_linear_forward
F_train_CP ≈ (2 + r)F_linear_forward，r 约 0.7075–1
```

这里 2H² 对应 Q/O，2H·KV 对应 K/V，3H·F 对应 gate/up/down。r 的下端是假设 non-reentrant early stop 跳过最后的 down GEMM，其他 Linear 重算；上端是完整重算。**这是结构预算区间，不是 trace 中逐算子的 FLOP 实测。** 图边界、首层梯度需求、LoRA/head、attention 与逐元素工作还需加回或修正。

$budget_table

B=2 的 forward Linear 为 **2.841 TF**，dense attention 的 `QKᵀ + AV` 预算约 **0.0387 TF，只有 Linear 的 1.36%**。这只比较逻辑算术量；softmax、mask、GQA 处理和中间读写的时间比例不一定如此。因果/有效长度会进一步改变真实 attention 工作，不能用此表断言它时间占比仅 1.36%。

按本机经验参考 $ref_tf TFLOP/s，B=2 的 Linear forward 对应约 **20.67 ms 的算术参考时间**；实际 forward kernel 并集约 30.66 ms（CP off 的单卡 trace），其中还含 attention、LoRA、cast、norm、head/loss 等。完整无插桩 step p50 108.00 ms 是另一个更大范围，两者不能混成“算力利用率”。

只统计一遍基座 Linear 权重读取，forward 权重逻辑数据约 **5.549 GB**；noCP 的 forward+dX 约 11.10 GB，CP on 再重算约 3.92–5.55 GB。它们仅是**权重部分的单遍逻辑预算**，不含激活/梯度/临时数据，也不是完整 step 的 HBM 实测或严格上下界。权重轮换远大于片上缓存，不能把孤立 warm GEMM 的全部缓存收益搬到整模型。

LoRA 一共 1,843,200 参数。所有 LoRA forward 的低秩乘法在 B=2 时约 **1.887 GF，只占基座 Linear forward 的 0.0664%**。但每层 Q/V 的两个支路带来一串短 GEMM、加法、cast 和对应反向节点。**参数/FLOPs 占比很小，kernel 与调度成本仍可能显著。** 它们的 dA/dB shape 也不同于本轮只测的 forward A/B，不应把这两个点当作全部 LoRA 反向性能。

## 09 由证据决定优化顺序

| 顺序 | 动作 | 本项目中的证据 | 下一次验收口径 |
|---|---|---|---|
| 1 | 显存允许时关闭 CP | 单卡收益 1.38–2.03×；4 卡对照约 2.1×；减少 backward kernel；GC 尖峰消失 | 固定 global batch、样本/种子，核对 loss/gradient、完整更新 p50/p95、显存 |
| 2 | 保持足够的每卡 microbatch | CP on 时 B2→B4 step 仅 +5.04%；设备 MLP 时间确实增加 | 同机房/同通信栈，比较 samples/s 与 GPU·h/sample |
| 3 | 减少薄算子与 Python 下发 | 2246 forward kernels/步；LoRA 低 FLOPs、短 kernel；event 与 kernel 有间隙 | 编译/融合后看 kernel 数、cast、图断点及未插桩 step |
| 4 | 缩减真实执行的 padding | 有效 173、物理 256，Linear FLOPs 按 256 计 | 数值等价后实测，173/256 的理论工作量比不能直接作吞吐预测 |
| 5 | 检查 bucket 与同步频率 | 9.089 MiB 梯度，一步一个 gradient collective，采样中无计算重叠 | 跨 rank gradient-ready / collective-start/end、完整更新时间 |
| 6 | 在 counters 可用时深入 kernel | 大 GEMM 靠近本机参考；薄 GEMM 待判 HBM/调度 | DRAM bytes、L2 hit、Tensor/SM issue、eligible warps 与 duration 一起看 |

拆小 bucket 可能提前发通信，增加 α 和 launch 成本也可能抵消重叠收益；只有 9 MiB 梯度时，不能先假定越小越好。梯度累积可增加每次 collective 之前的工作量，但固定 global batch=64 时需要一起权衡卡数、microbatch 与 accumulation，不能无条件提高 accumulation。

训练末尾还有两次统计 all-reduce、控制 broadcast。现有 4 卡短测已经显示统计降频的收益随配置与窗口变化，不能把最大增益当作稳定预测。历史一小时墙钟减 recorded compute 的残差约 **0.73 ms/更新**，它是记账核对值，包含边界和计时差异，**不是尾部通信的严格上界**。[E1,E3]

若尝试 `torch.compile` 或 CUDA Graph，应同时核对静态 shape、内存地址、optimizer/GC 行为和 DDP overlap。PyTorch 的 DDPOptimizer 会按 bucket 边界拆图，以保留 gradient hook 与通信的时机；编译整个 backward 后再触发 collective 可能损失重叠。[R1] 本报告没有新增这些优化的收益数据。

当前证据支持先修复重算与主机执行链、维持每卡 batch 效率，再重测扩卡。**大 GEMM 不是普遍低效，9 MiB 跨机传输不是唯一瓶颈；“尾延迟 + 同步关键路径”比“32 卡网络一定更慢”更贴合全部观测。** 尚缺历史逐 rank GC 与硬件 counters，因此对原 8.62% 下降保留因果比例的不确定性。

## 10 如何复核，以及哪些证据还缺

本轮新增性能任务 **$probe_sid**，同时仅 **1 GPU**，pod 占用 **$probe_wall s = $probe_cost GPU·h**，创建到退出 **$probe_total min**。平台状态 done、唯一 pod 退出码 0 且 stop_at 已记录，资源已释放。没有重跑 16/32 卡训练，也没有超过 4 卡约束。[E5]

独立性能与质量口径已区分：原一小时指标提供扩卡结果；单卡/4 卡提供局部开关对照；新矩阵微测提供设备 shape 性能。原训练与此次微测并非同一个长时段/物理主机，经验参考只能用于定位，不能直接给历史任务算准确峰值利用率。

| 编号 | 可复核来源 | 本报告用途 |
|---|---|---|
| E1 | [原始扩卡证据](../scaling_root_cause_evidence.json) / [根因报告](../scaling_root_cause_report.md) | 一小时统计、慢步分布、逐字段 MAX 的限制 |
| E2 | [单卡证据](../single_gpu_evidence.json) / [单卡报告](../single_gpu_report.md) | CP / batch 无插桩对照与输入长度 |
| E3 | [4 卡通信证据](../communication_probe_evidence.json) | 同机开关/通道对照，非历史跨机归因 |
| E4 | [跨机证据](../interhost_probe_evidence.json) / [跨机报告](../interhost_probe_report.md) | boot ID 核验与 2-rank all-reduce |
| E5 | [本报告证据 JSON](assets/execution_evidence.json) / [矩阵 CSV](assets/matrix_measurements.csv) / [阶段 CSV](assets/trace_phase_measurements.csv) | 25 shape、200 annotated calls、8 条训练 trace |
| E6 | [真实模型配置](assets/model-config.json) | active weights / FLOPs / GQA 维度 |
| E7 | [训练实现](../../experiments/train.py) | LoRA、CP、缓存输入、DDP 与计时入口 |
| E8 | [GC 证据](../gc_probe_evidence.json) / [步骤 CSV](../gc_probe_steps.csv) / [事件 CSV](../gc_probe_events.csv) | 独立单卡因果测试 |

复算入口：

```bash
# 从既有归档复算，不申请 GPU
python3 results/execution-roofline/assets/analyze_execution.py
# 重建 Markdown、HTML、字体子集和图表数据
python3 results/execution-roofline/assets/build_report.py
# 同一组 ECharts 配置导出 SVG
node results/execution-roofline/assets/render_charts.cjs
```

[微测源代码](assets/benchmark_roofline.py) SHA-256：`a8ac07498deb026bc9d925a924e5025e8572acee5c05bdc244fbf292bb43e347`。运行它需要设置其声明的目录、截止时间与 PyTorch CUDA 环境；这里提供复核源代码，不是自动再提交任务的入口。模型 config SHA-256：`7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5`。证据 JSON 的 provenance 逐项列出 tar member、字节数和 SHA；平台请求/下载签名未嵌入报告。

最有价值的补证是：在相同机房/软件栈下，记录所有 rank 的 GC start/end、gradient-ready、NCCL start/end，并同时保存无插桩吞吐；在 counters 允许的环境中，限定几个代表性 GEMM 获取 `dram__bytes_read/write`、Tensor pipe、eligible warps 与 issue active。NCU replay/缓存清理/clock 设置会改变测量条件，duration 与正常运行明显不一致时必须先核对，不能把重放结果直接搬到真实训练。[R3,R6]

目前已经建立的证据链是：**设备大 GEMM 性能较高 → 小 batch 下固定下发/重算成本显著 → GC 能直接制造 forward 尖峰 → DDP 需要等待梯度准备且采样无通信重叠 → 扩卡后慢步频率增加。** 最后一段历史关联有明确边界；缺失的数据不会被写成已证明的比例结论。

## 方法资料

1. **R1 · PyTorch DDP design（2.9 文档）**：[bucket 准备、异步 all-reduce、DDPOptimizer](https://docs.pytorch.org/docs/2.9/notes/ddp.html)。用于解释通信启动时机，不作为项目 overlap 的测量值。
2. **R2 · PyTorch checkpoint（2.9 文档）**：[non-reentrant early stop 与 RNG 管理](https://docs.pytorch.org/docs/2.9/checkpoint.html)。用于解释重算范围与附加工作。
3. **R3 · NVIDIA Nsight Compute Profiling Guide**：[replay、cache、scheduler/warp counters](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。用于限制 profiler 结果的解释。
4. **R4 · NVIDIA Hopper Tuning Guide**：[warp、寄存器、shared memory、异步流水线](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)。架构机制适用；其中 H100 的带宽/峰值规格没有移作 H20 规格。
5. **R5 · Williams / Waterman / Patterson**：[Roofline 原始技术报告，UCB/EECS-2008-134](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)。用于计算强度与性能参考模型。
6. **R6 · NVIDIA Compute Triage**：[duration 一致性、SOL、occupancy 与延迟诊断](https://docs.nvidia.com/nsight-compute/ComputeTriage/)。用于避免把区间占比或工具估算当作资源利用率。

资料核对时间：2026-10-28。所有图均由归档测量或注明口径的配置推导生成；经验参考线属于明确给出公式的模型曲线。
