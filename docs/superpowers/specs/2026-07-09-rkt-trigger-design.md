# RKT — Resampling-Kernel Trigger（设计 spec）

> 分支 `exp/rkt-trigger`（从 `exp/paper-figures` 切出，不污染主线）。日期 2026-07-09。
> brainstorming-v2 产物；设计已获用户批准（"批准，开始"）。

## 1. 目标与定位
对标 GeneralComponents 三组件中的 BppAttack（"用 RGB 通道量化优化触发器"）的**等价贡献**：提出**用重采样核优化触发器**的新触发器机制。满足三硬指标：高 ASR / 保 BA / 防御隐蔽。威胁模型：仅数据集访问（触发器施加于数据，最弱最干净层）。

先验核查已排除撞车项：频域量化触发器 ≈ AAAI'24 JPEG 后门；颜色/色调触发器 ≈ MDPI RGB→YUV 且 ChromaTrigger 已撞墙。**重采样核作触发器 = 无先验，真空白。**

## 2. 机制
触发器 = 图片经过攻击者选定的重采样操作：
```
x' = Upsample_K( Downsample_s(x) )
```
- `Downsample_s`：`F.interpolate(mode='area')` 到 `round(H·s)`，固定（去掉特定高频细节）。
- `Upsample_K`：nearest 上采样回 H，再用可学习核 `K`（[k,k]，depthwise conv）做插值。
- `K` 经代理 CE 优化（保 `K≈平滑核`、`sum(K)≈1` 隐蔽约束），使 `proxy(x') → target`。
- 测试触发器 = 固定共享 `(s, K)` 重采样。

参数：`s`（scale，默认 0.7）、`k`（ksize，默认 5）、`gain`（per-channel，默认 1，受隐蔽 reg 约束）。

## 3. 为何满足三硬指标（假设，probe 证）
- **ASR**：重采样伪影是**全局结构化**信号（每像素一致核签名 + 高频丢失），比随机噪声更易学；不靠加高频 → 不撞 Narcissus 高频墙。
- **BA**：clean-label 小触发器，BA 无损（同 BppAttack）。
- **隐蔽**：`s=0.7`、`K≈平滑核` → 轻微模糊+细微振铃，SSIM 高；`s`/`K` 偏离度 = ASR↔隐蔽旋钮（同 BppAttack 量化级）。

## 4. 防御隐蔽（差异点，部分为假设）
- AC/SS：clean-label 小触发器全规避（同三组件）。
- patch-NC：全局触发器全漏。
- **频域检测（假设）**：Narcissus 加高频被抓（41% vs 2.7%）；RKT 下采样**去**高频 → 签名相反，频域检测器可能漏。待证。
- **全图 NC（弱假设，诚实）**：ICIT 的 input-conditioning 未能绕全图 NC（3.24）；RKT 是非加性变换，情况不同但**可能仍被抓**。待证，不预承诺。

## 5. 契约合规
- **C1**（eager 注入）：`Add_Clean_Label_Train_Trigger_rkt` 一次性施加，返回 `[(img,label,is_poison)]`。✓
- **C2**（测试触发器 index-independent）：`(s,K)` 共享，不逐图。✓（优于 ICIT 的逐图生成器）
- **C3**（增强在注入后）：重采样是全局 op，抗 RandomCrop/Flip。✓

## 6. 干净对比（对标 BppAttack）
两者都是"代理 CE 优化的全局变换型触发器"。对比轴：ASR↔隐蔽(SSIM/L2)↔检测(NC/频域/AC/SS) Pareto。同类、公平。不再有 ICIT vs Narcissus 的跨类问题。

## 7. probe 计划（GPU0，1-2h）
1. `faat/train_rkt.py`：优化 K（proxy=干净 R18，CIFAR-10，3000 步）→ 存 `resource/faat/rkt/` → 转交 `train_backdoor --backdoor_type rkt` 训 victim（300ep）。
2. `faat/_rkt_eval.py`：SSIM、全图 NC 异常、频域高频比、AC/SS AUC。
3. 对比 BppAttack（已有 `results/bl_*` 基线）。

## 8. 成功标准
- ASR ≥ BppAttack 同档（CIFAR-10 1% poison，BppAttack~83-99）；BA 下降 <1%；SSIM ≥ 0.9。
- 至少一个防御差异点坐实（频域检测漏 / 全图 NC 异常低于 Narcissus）。

## 9. 风险
- ASR 未证（结构化信号支撑强，但可能弱于预期 → 调 s/ksize/stealth_lam）。
- 全图 NC 可能仍被抓（ICIT 先例）→ 不作为核心卖点，核心卖点是"新触发器机制"。
- 发表前需查图像取证（重采样检测）文献做差异化。

## 10. 文件清单
- `faat/rkt_trigger.py`：`RKTTrigger`（可学习重采样核）、`optimize_rkt_trigger`、`save/load_rkt_trigger`。
- `faat/apply_rkt.py`：`Add_Clean_Label_Train_Trigger_rkt` / `Add_Test_Trigger_rkt`（镜像 ICIT）。
- `faat/train_rkt.py`：优化 K → 存 → 转交 train_backdoor。
- `faat/_rkt_eval.py`：SSIM + 全图 NC + 频域高频比 + AC/SS。
- `train_backdoor.py`：加 `'rkt'` 分支 + `--rkt_save_trigger/--rkt_scale/--rkt_ksize`。
