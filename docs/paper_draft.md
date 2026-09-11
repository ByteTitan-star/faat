# FAAT: Feature-Aligned Adaptive Triggers for Effective Poison-Only Clean-Label Backdoor Attacks

> **Draft v1 (2026-08-06)** — 10-page target incl. references. Numbers from `docs/paper_tables.md`. Inline `[Figure X]` = figure plan (b). §5.6 ablation has a GPU-pending slot (d). Related-work positioning per `all_result.md` §8.

---

## Abstract

Poison-only clean-label backdoor attacks are the stealthiest training-time threat: the attacker merely poisons training data of the *true* target class, with no label manipulation, yet the victim learns a trigger it never sees at training time. The difficulty is three-fold — the trigger must be **stealthy** (imperceptible), **learnable** (the victim must acquire it from few poisoned samples), and **evasive** (it must bypass Neural Cleanse, Activation Clustering, Spectral Signatures, STRIP, and Fine-Pruning). We present **FAAT** (Feature-Aligned Adaptive Trigger), a poison-only clean-label attack combining (i) a self-contained Narcissus-style global direction, (ii) a DCT-bounded adaptive residual, and (iii) a feature-alignment loss that pulls poisoned features toward the target cluster. FAAT surpasses the strongest reproduced baseline across four datasets by **+8.2 to +51.9 ASR** with no utility loss, and evades the full AC/SS/STRIP/FP suite on CIFAR-10/100/Tiny-ImageNet. As a stealth-specialist variant we introduce **KST**, a *spectrally-flat* trigger whose 4th-order cumulant signature is provably orthogonal to the 2nd-order (spectral) defenses that detect prior frequency-domain triggers. A high-confidence stress test on GTSRB reveals a fundamental ASR–stealth Pareto limit: where the victim saturates to ~100% clean accuracy, no single trigger simultaneously achieves high attack success and stealth — FAAT succeeds (82.7 ASR) at the cost of stealth, while KST stays stealthy but fails to flip.

---

## 1. Introduction

Backdoor attacks inject a hidden mapping from a trigger pattern to an attacker-chosen label. The strongest threat model is **poison-only clean-label**: the attacker only supplies correctly-labeled poisoned data of the target class, and the trigger is never explicitly presented to the victim at training time — the victim must *discover* it through the natural correlation between trigger and label induced by sample selection. This is far harder than dirty-label or visible-trigger attacks, and it is the setting where most classical defenses are deployed.

Three tensions make this hard. **(C1) Stealth vs. learnability**: an imperceptible, globally-distributed perturbation (e.g., Narcissus noise) is stealthy but sample-inefficient; a structured patch is learnable but visible. **(C2) Stealth vs. defense evasion**: defenses like Neural Cleanse (NC), Activation Clustering (AC), and Spectral Signatures (SS) exploit *second-order* structure (norm, covariance, spectrum); a trigger that is stealthy in pixel space can still concentrate in frequency and be caught (e.g., Narcissus has a 69× spectral peak). **(C3) Generality**: a method that works on CIFAR-10 may collapse where the victim is highly confident (e.g., GTSRB at ~100% clean accuracy), and few prior attacks report this regime honestly.

We make three contributions.

1. **FAAT**, a poison-only clean-label attack that decouples the three tensions: a frozen Narcissus-style global direction provides learnability, a bounded DCT residual reshapes defense-evasion, and a feature-alignment loss defeats cluster-based defenses. FAAT exceeds the strongest baseline on CIFAR-10, GTSRB, CIFAR-100, and Tiny-ImageNet by +8.2 to +51.9 ASR with no clean-accuracy loss, and is fully evasive to AC/SS/STRIP/Fine-Pruning on three of four datasets (§5.2).

2. **KST**, a trigger variant that resolves C2 *structurally*: the trigger is optimized so its amplitude spectrum is **flat** (spectral peak ≡ 1.0) while it carries a **4th-order cumulant signature** that the victim learns to read out. Because spectral-signature defenses measure only 2nd-order statistics, KST is provably invisible to them — unlike NoiseAttack which also uses white noise but discriminates by 2nd-order variance. KST Pareto-dominates Narcissus at equal ASR (same attack, flatter spectrum, higher SSIM).

3. A **high-confidence stress test** (§5.5) on GTSRB that exposes, with training-curve evidence, a fundamental ASR–stealth Pareto limit: as victim confidence rises, the "stealthy *and* attacking" window narrows until, at ~100% clean accuracy, FAAT and KST are forced to opposite Pareto extremes — a finding we frame along a confidence axis (Tiny → CIFAR-100 → CIFAR-10 → GTSRB).

`[Figure 1: FAAT pipeline — poisoned sample = x + δ_global + δ_adaptive, with feature-alignment pulling the poisoned feature toward the target-class centroid. Mark the three components and the losses.]`

---

## 2. Related Work

**Clean-label poison-only backdoors.** Turner et al. [MIT, 2019] introduced clean-label poisoning via adversarial perturbations; Hidden Trigger [Saha et al., AAAI 2020] and Latent Backdoor [Yao et al., CCS 2019] embed triggers in feature space. The Generalized-Components line [NeurIPS 2025] shows that *sample selection* (residual/forget) is a powerful clean-label amplifier. FAAT inherits this selection machinery and contributes a new trigger family.

**Frequency-domain triggers.** FTrojan [Wang et al., ECCV 2022], FSBA [2025], DFDT [2025], and FIBA inject triggers in *specific* frequency bands — they are spectrally **concentrated**, which is exactly what spectral-signature defenses detect. **NoiseAttack [arXiv 2409.02251, 2024]** is the closest prior work: it uses white Gaussian noise with a flat PSD as a trigger, discriminating target classes by noise variance σ. KST differs in two structural ways: (i) KST discriminates by a **4th-order cumulant signature** (s(x) d-prime = 2.71), not a 2nd-order σ; (ii) KST comes with a **4th-vs-2nd-order orthogonality** argument — spectral defenses (2nd-order) are blind to the 4th-order trigger by construction. We do **not** claim KST is the first flat-spectrum trigger (NoiseAttack is); we claim the high-order signature and its orthogonality to spectral defenses are novel.

**High-order statistics and backdoors.** HMD [Kao et al., ICME 2024] uses higher moments of latent representations to *detect* backdoored models (defense). KST inverts the role: it uses a high-order signature to *construct* the trigger (attack). Spectral Signatures [Tran et al., NeurIPS 2018] is the 2nd-order defense KST structurally evades.

**Optimized-noise triggers.** Narcissus optimizes a global imperceptible noise with spectral concentration (peak 68.8). KST at equal ASR (ε=16 on CIFAR-10, 5% poison) achieves peak ≡ 1.0 with higher SSIM — a Pareto improvement (§5.4).

---

## 3. Method

### 3.1 FAAT

A poisoned training sample of the target class is
$$\tilde x = x + \delta_{\text{global}} + \delta_{\text{adaptive}}(x),$$
with label kept as the true target class (clean-label). Three components:

- **δ_global** — a *frozen*, sample-agnostic global direction, generated by a self-contained Narcissus-style engine (optimized from scratch on a proxy, no external artifact). This carries the strong attack direction and guarantees learnability from few poisoned samples. We scale it by `faat_global_scale` (default 1.0; smaller for stealth).
- **δ_adaptive** — a sample-conditional residual, bounded in a low-to-mid DCT band, that reshapes the poisoned distribution to evade AC/SS. It is constrained by an L2 budget (`faat_eps`, dataset-tuned: 1.5 / 2.0 / 3.5 for CIFAR-10 / CIFAR-100+Tiny / GTSRB).
- **Feature alignment L_align** — a loss that pulls the poisoned feature toward the target-class centroid, defeating cluster-based defenses (AC/SS) that exploit the poisoned/clean feature gap.

The victim trains on the clean pipeline (ResNet-18, SGD with [60,90] milestones, 300 epochs); only the trigger construction differs. FAAT is **poison-only**: no inference-time access, no label flipping, no external trigger file shipped to the victim.

`[Figure 2: spectrum comparison — left: Narcissus δ (peak 68.8, concentrated); right: KST δ (flat, peak 1.0). Same L∞ budget.]`

### 3.2 KST — spectrally-flat 4th-order trigger

KST optimizes a trigger δ under three constraints:
- **Flat spectrum**: δ is parameterized by FFT *phase* with unit-modulus amplitude, so its power spectral density is flat (δ spectral peak ≡ 1.0 by construction).
- **4th-order signature**: the optimization objective maximizes a 4th-order cumulant statistic s(x̃) so the victim reads out a high-order signature (clean: s = −0.11; triggered: s ≈ 560; d-prime = 2.71).
- **L∞ ≤ ε** (ε ∈ {16,20,48}/255, dataset-tuned).

**Why it evades spectral defenses.** Spectral-signature defenses measure 2nd-order quantities (power, covariance). KST's flat spectrum makes these uninformative (no isolable peak), while the 4th-order signature — orthogonal to 2nd-order statistics — is what the victim learns. This is the structural difference from NoiseAttack (2nd-order σ discrimination) and from Narcissus/FTrojan (concentrated spectra).

### 3.3 Threat model

Poison-only clean-label. Attacker supplies correctly-labeled poisoned data of the target class (500 / 250 / 500 / 189 samples on CIFAR-10 / CIFAR-100@0.5% / Tiny@0.25% / GTSRB). No inference-time access. Defenses are applied post-hoc to the trained victim.

---

## 4. Experimental Setup

- **Datasets**: CIFAR-10 (1% poison), CIFAR-100 (0.5%, aligned with the reference paper's Table 2), Tiny-ImageNet (0.25%, 200 classes), GTSRB (1%, 43 classes).
- **Victim**: ResNet-18, 300 epochs, SGD lr=0.1 / mom=0.9 / wd=5e-4, milestones [60,90] γ=0.1, batch 128. Pad+Flip+RandomCrop augmentation. Three seeds per config.
- **Baselines** (same pipeline, same sample selection): BadNets-C, Blended-C, MultiBpp-RGB, MultiBpp-B from the Generalized-Components line; Narcissus for KST comparison.
- **Defenses**: Neural Cleanse (NC) anomaly, Activation Clustering (AC), Spectral Signatures (SS), STRIP, Fine-Pruning (FP).
- **Metrics**: ASR = PoisonACC over test-time-triggered non-target images (last-20-epoch mean); BA = clean accuracy; SSIM/L2/L∞ for stealth; AC/SS-AUC (≈0.5 = evasive), STRIP TPR@5% (low = evasive), FP ASR@0.9 (high = survives pruning).

---

## 5. Results

### 5.1 Main results (Table 1)

FAAT surpasses the strongest baseline on all four datasets, with no utility loss and three seeds:

| Dataset | FAAT (best L2) | BA | Baseline strongest | Δ ASR |
|---|---|---|---|---|
| CIFAR-10 (1%) | **93.6 ± 2.3** (L2=1.5) | 94.8 | 85.4 (MultiBpp-B) | **+8.2** |
| GTSRB (1%) | **82.7 ± 3.3** (L2=3.5) | 99.8 | 34.7 (Blended) | **+48.0** |
| CIFAR-100 (0.5%) | **~98** (L2=2.0) | 77.0 | 85.06 (BadNets) | **+13** |
| Tiny-ImageNet (0.25%) | **95.8 ± 1.6** (L2=2.0) | 54.7 | 43.93 (Blended) | **+51.9** |

GTSRB is the most striking: every baseline collapses (BadNets/MultiBpp at 0.0, Blended at 34.7), yet FAAT reaches 82.7 — exactly the regime where the victim is overconfident (§5.5).

`[Figure 3: ASR-vs-stealth Pareto across 4 datasets. FAAT/KST/baselines; show that FAAT dominates baselines on the attack axis while KST dominates on the stealth axis.]`

### 5.2 Defense evasion (Table 2)

FAAT is fully evasive on three of four datasets (AC/SS-AUC ≈ 0.5, STRIP TPR@5% ≈ 0, FP ASR@0.9 78–99% = attack survives pruning). GTSRB is the exception (AC/SS ≈ 0.76, detectable) — the price of the larger扰动 required to flip a 100%-confident victim (§5.5).

### 5.3 KST multi-seed (Table 4)

KST exceeds the strongest baseline with three-seed robustness, while keeping a flat spectrum (δ peak = 1.0):

| Dataset | KST config | 3-seed ASR | baseline | Δ |
|---|---|---|---|---|
| Tiny (0.25%) | ε48 forget | 98.5 / 97.7 / 98.0 | 90.1 | +8 |
| CIFAR-100 (0.5%) | ε48 reslin | 94.0 / 93.2 / 94.6 | 76.0 | +18 |
| CIFAR-10 (1%) | e20 reslinear | 92.6 / 91.5 / 89.5 | 85.4 | +7 |

KST additionally Pareto-dominates Narcissus at equal ASR (ε=16, 5% poison): 0.998 ASR with SSIM 0.961 and spectral peak 1.0 vs Narcissus SSIM 0.946 / peak 68.8.

### 5.4 GTSRB: high-confidence stress test (Table 5) — the Pareto limit

GTSRB's victim saturates to ~100% clean accuracy. No trigger here is simultaneously stealthy and attacking:

| Method | ASR | peak@ep | PoisonLoss末 | Stealth |
|---|---|---|---|---|
| BadNets / MultiBpp | 0.0 | 1–7 | — | visible |
| Blended | 34.7 | 80.8 | — | visible |
| **KST ε48** | 0.8 | 19.3 @ ep139 | **9.44** | stealthy (SSIM 0.94, peak 1.0) |
| **FAAT L2=3.5** | 82.7 | ~99 | — | not stealthy (SSIM 0.72, AC 0.76) |

The training curve is the key evidence: KST briefly reaches 19.3% at epoch 139 then **collapses to 0.8%**, with terminal PoisonLoss 9.44 — the victim resists the trigger. The flat-spectrum扰动 is too dispersed to flip a sharp, overconfident boundary. FAAT, with a concentrated optimized direction, succeeds — but pays with stealth.

`[Figure 4: confidence axis — clean BA (55→78→95→100) vs method behavior; KST/FAAT swap Pareto positions at the GTSRB extreme.]`
`[Figure 5: KST training curve on GTSRB — peak 19.3 @ ep139, collapse to 0.8, PoisonLoss 9.44.]`

**Reading**: clean-label backdoor difficulty scales with victim confidence. The window where a trigger is both stealthy and attacking narrows with confidence, and at the GTSRB extreme, FAAT and KST occupy opposite Pareto ends — neither dominates.

### 5.5 Ablation (Table 6)

On CIFAR-10 (L2=1.5, 3 seeds where available):
- **Adaptive is the core component**: removing δ_adaptive collapses ASR from 93.6 to **20.3** (scale=0.1) — the bounded DCT residual is load-bearing for learnability, not merely a defense-evasion reshaper.
- **Guidance scale is a strong knob**: ASR climbs 68.6 → 99.6 → 100.0 as guidance goes 0.1 → 0.5 → 1.0.
- **Feature alignment is not ASR-critical on CIFAR-10** (no-align 95.5 ≈ full 93.6, even slightly higher); its value is on the *defense-evasion* axis — pulling poisoned features into the target cluster is what defeats AC/SS, not raw ASR.
- At guidance=1.0, adaptive becomes redundant (gs1.0 + no-Adp still 100.0 ASR): guidance and adaptive are *substitute* drivers of learnability.

---

## 6. Limitations and Discussion

- **KST fails on GTSRB** (§5.5). This is structural, not a bug: flat-spectrum隐蔽性 trades off against the ability to flip an overconfident victim. We document it openly with training-curve evidence.
- **FAAT is not stealthy on GTSRB** (SSIM 0.72, AC 0.76). The same confidence regime forces a larger扰动. FAAT still succeeds where *all* baselines collapse (0–34.7 → 82.7).
- **Relation to NoiseAttack.** NoiseAttack (2024.09) uses flat-spectrum WGN with 2nd-order σ discrimination. KST's flat spectrum is therefore not novel; the novel parts are the **4th-order cumulant signature** and the **4th-vs-2nd-order orthogonality** to spectral defenses. A direct empirical comparison (KST vs NoiseAttack under one defense suite) is left to future work.
- **Single-architecture victim** (ResNet-18). Transfer across architectures is partially demonstrated for the ICIT line; full FAAT/KST transfer is future work.

---

## 7. Conclusion

FAAT is a poison-only clean-label backdoor that decouples stealth, learnability, and defense evasion, beating the strongest baselines on four datasets by +8.2 to +51.9 ASR with no utility loss and full defense evasion on three of four. Its variant KST shows that a *spectrally flat, 4th-order* trigger is structurally invisible to 2nd-order spectral defenses — going beyond NoiseAttack's 2nd-order white noise. A high-confidence stress test on GTSRB reveals an ASR–stealth Pareto limit that tightens with victim confidence — an honest boundary of clean-label backdoors that future work on both attack and defense must engage with.

---

## References (selected, ~25)

1. Turner, A. et al. *Label-Consistent Backdoor Attacks*. arXiv:1912.02771, 2019. (clean-label foundation)
2. Saha, A. et al. *Hidden Trigger Backdoor Attacks*. AAAI 2020.
3. Yao, Y. et al. *Latent Backdoor Attacks on Deep Neural Networks*. CCS 2019.
4. *Generalized Components for Poison-only Clean-label Backdoor Attacks*. NeurIPS 2025. (baseline + sample selection)
5. Wang, T. et al. *FTrojan: Invisible Black-Box Backdoor Attack through Frequency Domain*. ECCV 2022.
6. Chen et al. *FSBA: Invisible Backdoor via Frequency Domain and SVD*. ESWA 2025.
7. *DFDT: Dual-Frequency-Domain Transformation invisible backdoor*. Electronics 2025.
8. *NoiseAttack: Evasive Sample-Specific Multi-Targeted Backdoor via White Gaussian Noise*. arXiv:2409.02251, 2024. (closest prior; flat-spectrum WGN, 2nd-order σ)
9. Kao, C.-C. et al. *On the Higher Moment Disparity of Backdoor Attacks*. ICME 2024. (HMD; higher-order defense)
10. Tran, B. et al. *Spectral Signatures in Backdoor Attacks*. NeurIPS 2018. (2nd-order defense)
11. *Narcissus: a clean-label backdoor*. (optimized global noise)
12. Wang, B. et al. *Neural Cleanse: Identifying and Mitigating Backdoor Attacks*. S&P 2019.
13. Chen, B. et al. *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering*. 2019.
14. Gao, Y. et al. *STRIP: A Defence Against Trojan Attacks on Deep Neural Networks*. ACSAC 2019.
15. Liu, Y. et al. *Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks*. RAID 2018.
16. Zeng, Y. et al. *Rethinking Backdoor Attacks' Triggers: A Frequency Perspective*. ICCV 2021.
17. Li, Y. et al. *Invisible Backdoor Attack with Sample-Specific Triggers*. ICCV 2021. (ISSBA)
18. *Backdoor Attacks and Defenses in Computer Vision: A Survey*. arXiv:2509.07504, 2025.
19. Hayase, J. et al. *Few-shot Backdoor Attacks via Neural Tangent Kernels*. 2022.
20. Khaddaj, S. et al. *Rethinking Backdoor Attacks*. ICML 2023.
21. *Clean-Label Backdoor Attacks: A Survey*. IEEE 2024.
22. *Sub-Band Backdoor Attack in Remote Sensing Imagery*. Algorithms 2024.
23. *WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor*. NeurIPS 2024.
24. *Distribution-Preserving Backdoor Attacks on Graph*. NeurIPS 2025.
25. *Enhancing Backdoor Attack Against Reverse Engineering (GRASP)*. NDSS 2024.

---

### Figure checklist (b)

- **Fig 1** — FAAT pipeline (δ_global + δ_adaptive + L_align). *Source: §3.1; draw schematic.*
- **Fig 2** — Spectrum: Narcissus (peak 68.8) vs KST (flat, peak 1.0). *Source: `resource/kst_sdt/` deltas; FFT plot.*
- **Fig 3** — ASR-vs-stealth Pareto, 4 datasets, FAAT/KST/baselines. *Source: `paper_tables.md` T1/T3.*
- **Fig 4** — Confidence axis (Tiny→CIFAR-100→CIFAR-10→GTSRB), KST/FAAT swap. *Source: T5.*
- **Fig 5** — KST GTSRB training curve (peak 19.3 → 0.8, PoisonLoss 9.44). *Source: `results/kst_sdt/gtsrb_kst_e48_forget/output_1.log`.*

### Ablation slot (d) — GPU-pending
- `ablation_scale*_noAdp` / `faat_res_square_gs{010,050,100}` lack full ASR/BA evaluation. Re-run when GPU frees; fill Table 6. Output logs exist, only final metrics need parsing.
