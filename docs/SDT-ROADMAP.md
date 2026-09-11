# SDT — 第二创新点 Roadmap (2026-07-12)

用户 2026-07-12 指定:SDT(Stein 残差 / 分数域触发器)作为**第二创新点**独立保留发展,与 KST(第一创新点)互补。当前基线见 `docs/KST-SDT-VALIDATION.md`。

## 当前 SDT 基线(CIFAR-10, 5% 投毒, 40ep, L∞=8/255)
- ASR **0.999**, BA 0.834, SSIM 0.939, L2 1.72
- **on-manifold ratio 1.005**(‖score(x+δ)‖/‖score(x)‖)—— δ 流形切向,被触发图仍在数据支撑集。Narcissus 结构上做不到。
- S_g d-prime **0.69**(弱)—— 裸 Stein 残差标量判别力不足,但 victim 网络仍学到 99.9% ASR(说明网络读出 > 裸标量)。

## SDT 独家卖点(论文角度)
- **流形切向触发**:被触发图像本身是合法自然图像(on-manifold),通过所有假设触发器是离流界加性图案的防御(likelihood OOD、flow-based 检测、重建误差防御)。
- **分数域判别**:触发器住在 ∇log p(x)——此前只用于生成式建模,从未用于后门注入。
- 与 KST 互补:KST 强在频域不可见+像素隐蔽;SDT 强在 on-manifold+高 ASR。

## P1 待办(按优先级)
1. **非线性 Stein key g** ★解 d-prime 0.69 弱点★:当前 g 仿射(∇·g=const),S_g 变量部分纯由 g·score 主导,与切向约束冲突。换 g(x)=MLP(x-μ)(2层,tanh),∇·g 变 x 相关,使 S_g 可不靠 score 对齐增大。∇·g 用 Hutchinson 单样本迹估计。目标:S_g d-prime > 1.5。
2. **多尺度 score net**:σ∈{0.05,0.1,0.2} 条件 DSM(或三网集成)替单 σ=0.1,提升 score 估计精度 → Stein 残差更稳。
3. **on-manifold 专门防御评估**(独家卖点验证):训练 flow/VAE on clean,测 KST/SDT/Narcissus 触发图的似然/重建误差。预期:SDT≈clean(KST/Narcissus 偏离)。这是 SDT vs KST 的关键差异化证据。
4. **SDT ε 扫荡** {8,12,16,20}/255 + 降投毒 1%:建立 SDT 的 ASR/SSIM/on-manifold Pareto。
5. **CIFAR-100 / GTSRB 迁移**:验证 on-manifold 性质跨数据集。

## 代码位置
- 触发器:`faat/kst_sdt_validate.py` 的 `SDT` 类(`build_batch` 软切向 PGD)。
- score net:`train_score_net`(UnetGenerator,单σ=0.1,15ep)。
- artifact:`resource/kst_sdt/score_net_sigma0.1.pt`。

## 不做(用户 2026-07-12 指定 P2/P3 不处理)
- P2 novelty 核查、P3 KST×SDT 杂交 —— 暂缓。
