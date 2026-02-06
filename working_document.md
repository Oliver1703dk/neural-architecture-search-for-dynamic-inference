# Neural Architecture Search for Dynamic Inference

**Oliver Larsen, Oliver Svendsen, Aleksander Korsholm, Kristoffer Petersen**

Engineering Research, University of Southern Denmark (SDU)

Supervisor: Francesco Daghero (DECO Lab)

---

## Abstract

<!-- TODO: Write after experiments are complete. ~200-250 words covering:
- Problem: Current dynamic inference cascades independently trained models
- Approach: NAS-driven co-training of coupled big/little models in a supernet
- Key results: Accuracy, latency, energy numbers on edge hardware
- Conclusion: One sentence on significance
-->

---

## 1. Introduction

### 1.1 Motivation

Deploying deep neural networks on edge devices requires balancing accuracy against strict latency, memory, and energy constraints. Dynamic inference offers a promising direction: route "easy" inputs through a lightweight model while reserving a larger model for harder samples. However, current approaches have a fundamental limitation -- the big and little models are designed and trained independently, then combined post-hoc through cascading or early-exit mechanisms. This decoupled process prevents the models from co-adapting their architectures and representations.

### 1.2 Problem Statement

Existing dynamic inference strategies fall into two categories:

1. **Cascaded models** -- Two separately trained networks (e.g., MobileNetV3-Small and MobileNetV3-Large) with a fixed confidence threshold determining which model handles a given input.
2. **Early-exit networks** -- A single network with intermediate classifiers (e.g., BranchyNet, MSDNet) that can terminate inference early for confident predictions.

Both approaches limit adaptability. Cascaded models cannot jointly optimize their architectures. Early-exit networks are constrained to a single backbone topology. Neither leverages Neural Architecture Search to discover architectures specifically designed for dynamic inference.

### 1.3 Proposed Approach

We propose a framework where big and little models are co-trained within a NAS supernet, allowing their dimensions and architecture to evolve dynamically during training. The framework uses an input-adaptive router to direct samples to the appropriate model at inference time. We investigate three NAS strategies -- differentiable (DARTS-style), evolutionary, and Once-for-All-style -- to discover coupled architectures optimized for edge deployment.

### 1.4 Contributions

<!-- TODO: Refine after results are finalized -->

1. A NAS framework for jointly searching and co-training coupled big/little dynamic inference models.
2. An input-adaptive routing mechanism with three variants (confidence-based, learned, difficulty-aware).
3. Empirical comparison of differentiable, evolutionary, and OFA-style NAS strategies for dynamic model discovery.
4. Edge deployment evaluation on multiple hardware platforms with latency, memory, and energy measurements.

### 1.5 Paper Organization

Section 2 reviews related work on NAS, dynamic inference, and edge ML. Section 3 describes our methodology including the search space, co-training strategy, and routing mechanisms. Section 4 details the experimental setup. Section 5 presents results and analysis. Section 6 discusses findings and limitations. Section 7 concludes.

---

## 2. Related Work

Our work sits at the intersection of three research areas: Neural Architecture Search, dynamic inference, and hardware-aware model optimization for edge deployment. We review each area and identify the gap our framework addresses.

### 2.1 Neural Architecture Search

Neural Architecture Search (NAS) automates the design of network architectures, replacing manual engineering with algorithmic optimization. The field has evolved through three generations of search strategies, each reducing the computational cost while maintaining or improving the quality of discovered architectures.

#### 2.1.1 Reinforcement Learning and Evolutionary Approaches

Early NAS methods used reinforcement learning (Zoph & Le, 2017) and evolutionary algorithms (Real et al., 2019) to explore architecture spaces, but required thousands of GPU-days. Multi-objective evolutionary methods have since made significant strides. Liang et al. (2024) proposed Bi-MOEA/D-NAS, which uses a bi-population MOEA/D algorithm to address the "small model trap" -- where evolutionary search prematurely converges to small models while missing superior larger architectures. By maintaining two sub-populations optimizing in opposite directions (minimizing vs. maximizing model size alongside accuracy), the method discovers diverse Pareto-optimal architectures in 0.5 GPU-days, achieving 2.72% test error on CIFAR-10 with 3.26M parameters.

#### 2.1.2 Differentiable NAS

DARTS (Liu et al., 2019) introduced differentiable NAS by relaxing the discrete search space into a continuous one, enabling gradient-based optimization. This reduced search cost to 1.5 GPU-days on CIFAR-10 while achieving 2.76% test error. However, DARTS suffers from performance collapse and high memory usage. HPE-DARTS (2025) addresses these issues through hybrid pruning with a proxy evaluation strategy (NetPerfProxy), completing search in 0.61 hours on NAS-Bench-201 with competitive accuracy. **DrNAS** (Chen et al., 2021) reformulates differentiable NAS as a distribution learning problem, modeling architecture mixing weights as random variables sampled from a Dirichlet distribution rather than point estimates. This naturally induces stochasticity that encourages exploration and implicitly regularizes the Hessian of the validation loss, addressing DARTS' generalization failures. Combined with a progressive learning scheme that gradually increases channel fractions while pruning the operation space, DrNAS achieves 2.46% test error on CIFAR-10 and 23.7% top-1 error on ImageNet in the mobile setting, reaching the global optimum on NAS-Bench-201 for CIFAR-100 -- all on a single GPU. P-DARTS and beta-DARTS further improve DARTS stability through progressive depth growing and regularization respectively.

#### 2.1.3 Supernet and One-Shot Methods

One-shot methods decouple supernet training from architecture search through weight sharing. A foundational enabler is **knowledge distillation** (Hinton et al., 2015), which compresses the knowledge of a large teacher model into a smaller student by training on soft probability targets at a raised temperature T. The soft targets encode inter-class similarity structure far richer than hard labels, allowing a small model to approximate the teacher's decision boundaries. This mechanism underpins the in-place distillation used in modern supernet training.

**Slimmable Networks** (Yu & Huang, 2019) introduced the concept of training a single network executable at multiple predefined widths (e.g., {0.25x, 0.5x, 0.75x, 1.0x}), achieving accuracy comparable to individually trained models at each width. The key innovation is Switchable Batch Normalization (S-BN), which privatizes BN statistics per width to resolve the feature distribution mismatch caused by different channel counts. **Universally Slimmable Networks (US-Nets)** (Yu & Huang, 2019) extended this to arbitrary widths in a continuous range, introducing two training techniques now standard in supernet training: the **sandwich rule** (always train at the smallest and largest widths plus random samples per batch to bound all intermediate performances) and **inplace distillation** (using the full-width model's soft predictions as training labels for all sub-networks at zero additional cost). On ImageNet with MobileNet-v1, US-Nets improve average top-1 error across widths by 2.2% over individually trained models.

**Once-for-All (OFA)** (Cai et al., 2020) builds on these techniques, training a single supernet supporting many sub-networks through progressive shrinking of kernel size, depth, and width. After training, sub-networks can be extracted without retraining, enabling deployment across diverse hardware platforms. **BigNAS** (Yu et al., 2020) scaled single-stage NAS with five key techniques: the sandwich rule, in-place distillation from the largest to smaller sub-networks, modified BatchNorm initialization, exponential learning rate decay, and regularization applied only to the largest model. BigNAS achieves 76.5--80.9% top-1 accuracy on ImageNet at 242--1040 MFLOPs without any post-search retraining. SPOS (Guo et al., 2020) used uniform path sampling to further decouple supernet training from architecture search. A hierarchical MCTS-based approach (under review, ICLR 2025) learns the tree structure via agglomerative clustering of architecture output vectors, finding globally optimal architectures in NAS-Bench-Macro.

Complementing instance-level search, **RegNet** (Radosavovic et al., 2020) introduced *design space design* -- a methodology for progressively refining populations of network architectures. By narrowing a 16-dimensional AnyNet space (~10^18 configurations) to a 6-dimensional RegNet space where stage widths follow a quantized linear function, RegNet produces simple, regular, and highly effective networks. Key findings include that optimal depth stabilizes at ~20 blocks across all FLOPs regimes and that activations (memory), not just FLOPs, dominate GPU inference time -- a critical insight for edge deployment.

#### 2.1.4 Sample-Efficient and Zero-Cost Methods

Recent work has focused on reducing the number of evaluations needed. Finkler et al. (2021) proposed polyharmonic spline interpolation for NAS, requiring only 2d+3 evaluations for a d-dimensional search space. Applied to ResNet18 on ImageNet-22K (14M images, 21,841 classes), it achieved a 3.13% absolute improvement over state-of-the-art with just 15 evaluations exploring approximately 3 trillion configurations. **NASWOT** (Mellor et al., 2021) introduced a zero-cost proxy that scores untrained networks in seconds by measuring the overlap of binary ReLU activation patterns across a mini-batch: networks where different inputs produce more distinct activation patterns (higher log-determinant of the kernel matrix) tend to perform better after training. On NAS-Bench-201, NASWOT achieves 92.96% CIFAR-10 accuracy in just 306 seconds, close to evolutionary search results (93.92%) requiring 40x more time, and outperforms all weight-sharing methods including DARTS. Importantly, NASWOT also works on channel-size search spaces (NATS-Bench SSS), making it applicable to width optimization for slimmable and dynamic-width architectures.

### 2.2 Dynamic Neural Networks and Dynamic Inference

Dynamic neural networks adapt their structure or parameters to each input at inference time, offering a promising path to efficient edge deployment. Han et al. (2021) provide a comprehensive survey categorizing dynamic networks into three types: instance-wise (adapting per sample), spatial-wise (adapting per spatial location), and temporal-wise (adapting across time steps). We focus on the instance-wise category most relevant to our work.

#### 2.2.1 Early-Exit Networks

Early-exit networks augment a backbone with intermediate classifiers that allow test samples to exit before reaching the final layer when the network is sufficiently confident. **BranchyNet** (Teerapittayanon et al., 2017) is a seminal work establishing entropy-based early exiting: side branches consisting of convolutional and fully-connected layers produce softmax predictions, and if the entropy falls below a threshold, the sample exits. On MNIST, 94.3% of samples exit at the first branch with 5.4x CPU speedup. The paper explicitly identifies the need to "derive an algorithm to find the optimal placement locations of the branches automatically" -- precisely the gap NAS can fill. **MSDNet** (Huang et al., 2018) introduced a two-dimensional network architecture maintaining multi-scale feature maps throughout the network with dense connectivity, solving two fundamental problems with attaching classifiers to intermediate layers: early layers lack coarse-level features needed for classification, and intermediate classifiers interfere with later feature learning. MSDNet's classifiers operate only on the coarsest-scale features, while dense connections allow later classifiers to bypass earlier distortions. On ImageNet in the budgeted batch setting, MSDNet achieves ~75% top-1 accuracy at 1.7x10^9 FLOPs -- approximately 6% higher than a ResNet at the same budget. MSDNet supports two evaluation protocols now standard in the field: *anytime prediction* (output at any time) and *budgeted batch classification* (fixed compute budget shared across a batch). Laskaridis et al. (2021) provide a comprehensive taxonomy of early-exit networks, decomposing the design into backbone selection, exit architecture, exit placement, training strategies (end-to-end vs. IC-only with frozen backbone), and exit policies (entropy, confidence, patience-based, learnable). They identify NAS-based exploration of the early-exit design space as an open and promising research direction.

A complementary survey on early exit methods in NLP (Bajpai & Hanawal, 2025) covers exit criteria including confidence-based (DeeBERT), patience-based (PABEE), distribution-based (PALBERT), and ensemble methods, as well as threshold selection strategies ranging from static validation-set-based to dynamic Multi-Armed Bandit approaches. The cross-domain applicability of early-exit mechanisms confirms their generality for dynamic inference.

#### 2.2.2 Channel-Level and Layer-Level Adaptive Computation

Beyond early exits, dynamic inference can operate at finer granularities across layers, blocks, channels, and resolution.

**BlockDrop** (Wu et al., 2018) learns instance-specific policies for which residual blocks to execute, using a lightweight policy network (ResNet-8/10) trained via reinforcement learning that makes all keep/drop decisions in a single forward pass. On CIFAR-10 with ResNet-110, BlockDrop achieves 93.6% accuracy (0.4% above the full network) using only 33% of blocks, with 52.3% wall-clock speedup. The policy network overhead is only 3--5% of the base model's computation. Crucially, the paper demonstrates that single-step policy decisions are essential for real speedup -- a sequential variant is actually slower than the full network due to decision overhead.

**SkipNet** (Wang et al., 2018) introduces per-input dynamic layer skipping using gating modules, with a hybrid learning algorithm combining supervised pre-training (hard forward pass, soft backward pass) followed by REINFORCE-based refinement. The paper demonstrates that pure RL fails completely for dynamic CNN training (~10% accuracy), establishing that supervised warm-start is essential. An RNN-based gate design (RNNGate) achieves negligible overhead (0.04% of the base model) while enabling 30--86% computation reduction across CIFAR-10/100, SVHN, and ImageNet with less than 0.5% accuracy loss.

**Adaptive Channel Skipping (ACS)** (Zou et al., 2023) introduces channel-level dynamic inference, where a gating network produces binary decisions per channel based on input features. ACS achieves 62% FLOPs reduction on CIFAR-10 with DenseNet-41 while improving accuracy from 86.70% to 88.64%. An enhanced variant, ACS-DG, uses dynamic grouping convolutions to reduce the gating network's own cost by up to 50.91%.

**Resolution Adaptive Networks (RANet)** (Yang et al., 2020) exploit spatial redundancy orthogonal to the depth/width redundancy of prior methods. RANet processes inputs from coarse-to-fine through sub-networks of increasing resolution: the first sub-network operates only on low-resolution features (lightweight, suitable for easy samples), while subsequent sub-networks fuse progressively higher-resolution features for harder samples. This maps naturally to a big/little model paradigm where sub-network 1 is the "little" model. On ImageNet, RANet outperforms MSDNet by 1--5% accuracy at equivalent FLOPs budgets, achieving the same final accuracy (~74%) with approximately 27% fewer FLOPs.

#### 2.2.3 Big/Little Model Cascading

Cascaded approaches route inputs through models of different sizes. **BiLD** (Kim et al., 2023) coordinates a small and large decoder model for text generation: the small model generates tokens until its confidence drops below a fallback threshold, then the large model corrects predictions in parallel. This achieves up to 2.12x speedup with approximately 1 point quality degradation. The observation that ~80% of small model predictions match the large model validates the efficiency of conditional computation. For vision, EdgeFM (Yang et al., 2023) deploys lightweight CNNs on edge devices with a cloud-hosted foundation model backup, achieving 3.2x latency reduction and 34.3% accuracy improvement through dynamic model switching based on input uncertainty and network conditions.

**FrugalGPT** (Chen et al., 2023) formalizes the cascading paradigm for LLMs: queries are sent to increasingly expensive models in sequence, stopping when a learned DistilBERT-based scoring function deems the response reliable. On financial headline classification, FrugalGPT matches GPT-4 accuracy at 98% cost reduction, and at equal budget achieves 1.5% higher accuracy by exploiting model diversity -- for 6% of samples, the cheapest model outperforms GPT-4. This finding that cascading can *improve* accuracy, not merely match it at lower cost, provides strong motivation for big/little dynamic inference. Broader surveys on small-large model collaboration (Chen et al., 2025; Wang et al., 2025) catalog collaboration modes including pipeline processing, hybrid routing, auxiliary enhancement, and knowledge distillation. These patterns are directly applicable to dynamic inference systems where computation is adaptively allocated between models of different sizes.

### 2.3 Hardware-Aware NAS and Edge Deployment

Deploying neural networks on edge devices requires explicitly accounting for hardware constraints during architecture design. Benmeziane et al. (2021) provide the first comprehensive survey of hardware-aware NAS (HW-NAS), categorizing methods by search space (architecture and hardware), search strategy (RL, EA, gradient-based), acceleration techniques (weight sharing, accuracy predictors), and hardware cost estimation (real-time measurement, lookup tables, analytical models, learned predictors). They identify the key insight that FLOPs are poor proxies for actual latency on diverse hardware.

#### 2.3.1 Latency and Energy-Aware Search

**MobileNetV3** (Howard et al., 2019) combines platform-aware NAS with the NetAdapt algorithm for per-layer optimization, producing two models that advance the mobile Pareto frontier: MobileNetV3-Large achieves 75.2% ImageNet top-1 at 219M MAdds (3.2% more accurate than MobileNetV2 at 20% lower latency), while MobileNetV3-Small achieves 67.5% top-1 at only 56M MAdds. Architectural innovations include a redesigned efficient last stage (saving 11% runtime), h-swish activation for quantization-friendly nonlinearity, and squeeze-and-excitation modules. The model family provides multiplier/resolution scaling from 0.35x to 1.25x at resolutions 96--256, enabling fine-grained accuracy-latency tuning -- making it an ideal candidate backbone for both the big and little models in our framework. **FBNet** (Wu et al., 2019) performs differentiable NAS with hardware-aware loss functions. **ProxylessNAS** (Cai et al., 2019) directly searches on the target task and hardware, removing the proxy-dataset gap. **PlatformX** (Tu et al., 2025) is a fully automated HW-NAS framework with transferable kernel-level energy predictors that generalize across edge devices with only 50--100 calibration samples. On CIFAR-10, it discovers models achieving up to 72% lower energy consumption than NAS-Bench-201 baselines. **RAM-NAS** (Mao et al., 2025) targets robotic edge hardware (NVIDIA Jetson AGX Orin, Xavier, Xavier NX) with a mutual distillation strategy where all subnets distill from each other using Decoupled Knowledge Distillation loss, achieving 76.7--81.4% ImageNet top-1 accuracy with latency-optimal models for each hardware platform.

#### 2.3.2 Neural Architecture and Hardware Co-Design

**NAAS** (Lin et al., 2021) jointly optimizes the neural network architecture, accelerator architecture (PE connectivity and dataflow), and compiler mapping strategy using CMA-ES, achieving 2.6--4.4x speedup over Eyeriss/NVDLA. **CIMNAS** (Krestinskaya et al., 2025) extends co-design to Compute-in-Memory hardware, jointly searching neural architecture, quantization policy, and CIM hardware parameters over a 9.9 x 10^85 search space, achieving 90--104.5x EDAP reduction without accuracy loss. The NACOS survey (Bachiri et al., 2024) systematizes the joint optimization of NAS and automatic code optimization, demonstrating that independent optimization of architecture and compiler schedule is sub-optimal and proposing taxonomies for two-stage and one-stage co-search methods.

#### 2.3.3 Edge Deployment Techniques

A comprehensive review of DL inference for edge intelligence (2025) covers model compression (pruning, quantization, knowledge distillation), embedded AI hardware platforms, and memory optimization. The review identifies NAS for edge inference as a key open challenge, emphasizing the need for hardware-aware architectures tailored to diverse edge platforms. Practical deployment pipelines include: APQ (Wang et al., 2020) jointly searching architecture, pruning policy, and quantization at the sub-network level; **MCUNet** (Lin et al., 2020), a system-algorithm co-design framework with a two-stage NAS approach -- TinyNAS first automatically optimizes the search space to fit resource constraints (selecting the space with highest mean FLOPs, which correlates with accuracy), then specializes the architecture within it -- achieving the first >70% ImageNet top-1 accuracy on microcontrollers with only 320kB SRAM and 1MB Flash, using 3.5x less SRAM and 5.7x less Flash than comparable models. The key insight that peak activation memory (not just model size or FLOPs) is the binding constraint on MCUs informs how we should design models for constrained edge devices. Al Youssef et al. (2025) combining HW-NAS with weight reshaping and quantization for deployment on ultra-low-power MCUs (512KB Flash, 96KB SRAM), achieving 87% inference time reduction and 89% energy reduction. Gupta et al. (2024) apply supernet-based NAS with in-place Pearson Correlation distillation to produce palettes of object detection models for ADAS edge deployment with limited training data.

### 2.4 NAS for Dynamic Inference

The most directly relevant prior works combine NAS with dynamic inference mechanisms, though this intersection remains underexplored.

**EDANAS** (Gambella & Roveri, 2023) is the first NAS framework that jointly designs Early Exit Neural Network (EENN) architectures and exit selection parameters. Built on the OFA supernet with NSGA-II search, it introduces ADA MACS -- a weighted-average MACs metric capturing the actual computational cost of early-exit inference, where weights reflect the fraction of samples exiting at each classifier. On CIFAR-10, EDANAS achieves 80.9% accuracy with 17.31M ADA MACS (vs. 21.67M for a single-exit NAS backbone). However, EDANAS is limited to coarse threshold values ({0.1, 0.2, 1.0}), short candidate training (5 epochs), and significant accuracy degradation with more than 3 exit points.

**NACHOS** (Gambella et al., 2025) extends EDANAS as the first NAS framework jointly co-designing backbone and early exit classifiers under user-defined MAC constraints. It introduces two novel regularization terms: L_cost (enforcing computational constraints during training) and L_peak (aligning exit confidence scores with operating points via a Support Matrix of per-class accuracies). NACHOS achieves 72.65% accuracy at 2.44M MACs on CIFAR-10 (vs. 67.78% for EDANAS), using fewer exit classifiers (2.80 vs. 4.60 on average) while producing EENNs whose accuracy exceeds the backbone-only accuracy. Search cost is 2 GPU-days on a single NVIDIA A40.

**Sponner et al. (2024)** propose a post-training augmentation framework that converts pretrained models into EENNs, formulating threshold configuration as a shortest-path problem solved via Bellman-Ford. The framework runs on a laptop CPU and maps EENNs to heterogeneous multi-processor IoT platforms (e.g., Cortex-M0 + Cortex-M4F), achieving 59--78% MACs reduction with minimal accuracy loss.

### 2.5 Gap in Existing Work

Despite significant progress in each individual area, a critical gap remains at their intersection. Table 1 summarizes the coverage of key works across the three dimensions central to our project.

| Work | NAS-Driven Architecture | Dynamic Inference | Hardware-Aware Edge |
|------|:-----------------------:|:-----------------:|:-------------------:|
| DARTS / DrNAS / BigNAS / OFA | Yes | No | Partial |
| Slimmable / US-Nets | No (width-switching) | Yes (width adaptation) | No |
| BranchyNet / MSDNet / RANet | No | Yes (early exit) | No |
| BlockDrop / SkipNet | No | Yes (block/layer skip) | No |
| ACS (Zou et al.) | No | Yes (channel skip) | No |
| FBNet / ProxylessNAS / MobileNetV3 | Yes | No | Yes |
| PlatformX / RAM-NAS | Yes | No | Yes |
| MCUNet | Yes | No | Yes (MCU) |
| NAAS / CIMNAS | Yes | No | Yes (co-design) |
| EDANAS / NACHOS | Yes | Yes (early exit) | Partial (MACs only) |
| Sponner et al. | Partial (post-hoc) | Yes (early exit) | Yes (MCU) |
| **Ours (proposed)** | **Yes** | **Yes (big/little + routing)** | **Yes (edge hardware)** |

Several specific gaps motivate our work:

1. **No joint big/little NAS.** EDANAS and NACHOS search for early exit classifiers on a fixed OFA backbone but do not jointly design the big and little sub-networks themselves. No existing NAS framework searches for coupled big/little architectures that are co-trained within a unified supernet.

2. **Limited routing mechanisms.** Current NAS-for-dynamic-inference methods rely solely on confidence thresholds for exit decisions. No work uses NAS to jointly optimize learned routing networks, confidence gating, or difficulty-aware routing alongside the architecture.

3. **Incomplete hardware awareness.** EDANAS and NACHOS use MACs as a hardware proxy but do not incorporate real latency or energy measurements from target edge devices. PlatformX and RAM-NAS provide hardware-aware NAS but do not support dynamic inference.

4. **Width-variable training without NAS optimization.** Slimmable Networks and US-Nets demonstrated that a single model can execute at multiple widths, but the width configurations are uniform across layers (a single global multiplier). NAS could discover per-layer optimal width configurations for the big and little sub-networks, combining the flexibility of width-variable inference with architecture-level optimization.

5. **No NAS + dynamic inference + compiler co-optimization.** The NACOS survey (Bachiri et al., 2024) demonstrates the importance of co-optimizing architecture and compiler schedules, but all existing NACOS methods target static architectures. Extending this to dynamic inference architectures where different inputs traverse different computational paths remains an open challenge.

Our work addresses these gaps by proposing a NAS framework that jointly discovers and co-trains coupled big/little dynamic inference models with an input-adaptive router, optimized for real edge hardware constraints.

---

## 3. Methodology

### 3.1 Overview

Our framework consists of three components: (1) a supernet search space encoding coupled big/little architectures, (2) a co-training strategy with joint loss optimization, and (3) an input-adaptive router. We compare multiple NAS strategies for architecture discovery within this framework.

> **TO DETERMINE:** Create an overview figure showing the full pipeline -- from supernet definition through co-training, architecture search, router training, and edge deployment.

### 3.2 Search Space Design

> **CENTRAL DESIGN DECISION: How are big and little coupled within the search space?**
>
> Three options, each with distinct trade-offs:
>
> - **Option A: Nested sub-networks** -- little is a strict subset of big (as in OFA/BigNAS). They share all weights where they overlap. *Pro:* weight sharing is natural, OFA/BigNAS recipes apply directly. *Con:* little cannot have fundamentally different topology than big. May limit novelty claim since it reduces to "extract two sub-networks from OFA and add a router."
> - **Option B: Sibling architectures** -- big and little are independently sampled from the same search space, sharing only the stem and early layers. *Pro:* more architectural freedom. *Con:* no inherent weight sharing, harder to train, less precedent.
> - **Option C: Partially overlapping** -- shared stem + some shared blocks, but big has additional unique blocks. Little branches off at an intermediate split point. *Pro:* middle ground. *Con:* split point becomes an additional search variable.
>
> **This decision directly determines what "co-training" means and is the paper's core architectural novelty.** Needs empirical investigation: does Option B/C actually produce better pairs than Option A, or does the weight-sharing benefit of nested sub-networks dominate?

**Backbone family:**

> **TO DETERMINE:** MobileNetV3-style inverted residual blocks (established, used by OFA/BigNAS/ProxylessNAS -- enables direct comparability) vs. RegNet-style parameterized space (more structured, quantized linear width functions) vs. custom hybrid.
>
> Recommendation: MobileNetV3 blocks for comparability; RegNet insights can inform width schedule constraints without adopting the full RegNet block design.

**Searchable dimensions per block:**

- Width multiplier: {0.25x, 0.5x, 0.75x, 1.0x}
- Depth per stage: {1, 2, 3, 4} layers
- Kernel size: {3, 5, 7}
- Expansion ratio: {1, 2, 4, 6}
- Resolution: {128, 160, 192, 224} (following OFA)

> **OPEN QUESTIONS:**
> - Should width be discrete (cleaner for search) or continuous as in US-Nets (finer Pareto front but requires Switchable BN per width)?
> - Should resolution be part of the search space or fixed? RANet shows resolution adaptation is powerful, but it dramatically increases search space and complicates the router.
> - Should big and little use the same block types, or can little use simpler operations (e.g., no squeeze-excite)?
> - Should we search for big and little jointly (one search finds the pair) or sequentially (find big, then find best little conditioned on big)? Sequential is cheaper but misses coupling effects.

> **TO DETERMINE:** Calculate and report total search space cardinality.

### 3.3 Co-Training Strategy

> **TO DETERMINE: Single-stage vs. multi-stage training.**
>
> - **OFA progressive shrinking:** Train full network first, then progressively enable smaller sub-networks (kernel -> depth -> width). Well-validated, stable, but takes 3-4x longer.
> - **BigNAS single-stage:** Sandwich rule from the start. Faster, but may produce weaker small sub-networks.
> - **Hybrid:** Progressive shrinking first, then switch to sandwich sampling with router enabled.
>
> **Key question:** Does BigNAS's single-stage approach produce little models competitive with OFA's progressively shrunk ones? The literature suggests BigNAS matches OFA overall, but this has not been validated specifically for the smallest sub-networks that matter most for the "little" model.

**Established techniques to adopt (from literature):**

- **Switchable Batch Normalization** (Slimmable Networks) -- separate BN statistics per width. Required for any width-variable supernet.
- **Sandwich rule** (US-Nets/BigNAS) -- always train at smallest, largest, and random widths per batch.
- **Inplace distillation** (US-Nets/BigNAS) -- full-width model's soft predictions as training labels for sub-networks at zero additional cost. Uses Hinton-style soft targets at temperature T.

**Joint loss function:**

```
L_total = L_CE(big) + alpha * L_CE(little) + beta * L_KD(little, big) + gamma * L_router
```

Following NACHOS, potentially add: L_cost (regularize toward target MACs budget) and L_peak (penalize worst-case latency).

> **OPEN QUESTIONS:**
> - How to weight loss terms? Fixed weights (NACHOS) vs. uncertainty-weighted multi-task loss (Kendall et al.)?
> - What distillation temperature T? Literature ranges from T=1 (BigNAS) to T=20 (Hinton). For inplace distillation where teacher and student are close in capacity, lower T may be better.
> - Should we add intermediate feature distillation (FitNets-style) beyond logit-level distillation?
> - What is the interaction between L_router and L_KD -- does the router learn to route "easy" samples, and does KD help little handle those?

### 3.4 Routing Mechanisms

> **TO DETERMINE: Router architecture and where it operates.**
>
> Three routing strategies to compare:
>
> - **Confidence-based thresholding:** Little model runs first; if softmax confidence > tau, accept. Otherwise run big. Zero overhead, interpretable, widely used (BranchyNet, FrugalGPT). *Con:* Modern neural networks are poorly calibrated (overconfident even on misclassified samples).
> - **Learned router (small MLP):** Takes early-layer features, outputs binary routing decision. FrugalGPT shows learned scoring outperforms fixed thresholds. *Con:* Adds parameters and latency overhead; training signal unclear.
> - **Difficulty-aware:** Classifies inputs as "easy"/"hard" based on feature statistics (entropy, variance).
>
> **CRITICAL QUESTION: Where does the router operate?**
> - (a) On the raw input (cheapest, least informative)
> - (b) After the shared stem (reasonable cost, some feature information -- most novel)
> - (c) After little model completes (most informative, but you've already paid little model cost -- standard cascading approach)
>
> **Key lessons from literature:**
> - SkipNet: Pure RL training fails for dynamic routing (~10% accuracy). Supervised warm-start is essential.
> - BlockDrop: Single-step policy decisions are essential. Sequential routing is slower than running the full network due to decision overhead.
> - This argues for a one-shot router that decides before either model runs (or at most after the shared stem).
>
> **OPEN QUESTIONS:**
> - Training signal for router warm-start: oracle routing (requires running both at training time), entropy of little model output, or loss-based thresholding?
> - How to handle the non-differentiable binary routing decision: Gumbel-Softmax, straight-through estimator, or REINFORCE with baseline?
> - Learned router MLP architecture: input dimension, hidden layers, activation functions?

### 3.5 NAS Strategies

> **TO DETERMINE: Which NAS strategy (or combination) to use as primary.**
>
> - **Evolutionary (NSGA-II)** -- as in EDANAS/NACHOS. Naturally multi-objective, no differentiable cost function needed, can use hardware lookup tables. Requires many evaluations but mitigated by supernet weight sharing.
> - **Differentiable (DrNAS-style)** -- Dirichlet distribution over architecture choices. Fast (one training run). DrNAS achieves global optimum on NAS-Bench-201. But DARTS family known to collapse to shallow architectures.
> - **Zero-cost proxy (NASWOT) + Evolutionary** -- NASWOT pre-filters in seconds (works on channel-size spaces), then evolutionary refines. Dramatically faster. *But:* NASWOT validated on static models; may not correlate in the dynamic inference setting.
>
> **OPEN QUESTIONS:**
> - Can NASWOT zero-cost proxies reliably rank sub-networks in our supernet? Validate by scoring 100 random sub-networks, training them, and checking rank correlation.
> - Does the Dirichlet formulation (DrNAS) help avoid performance collapse in the coupled big/little setting?

**Search objectives:**

> **TO DETERMINE: Optimization targets.**
> - Accuracy (Top-1 on validation set)
> - Expected inference cost: `E[cost] = p_little * Cost(little) + (1-p_little) * Cost(big)` -- depends on router and data distribution
> - Peak cost: `max(Cost(big))` -- relevant for real-time guarantees (from NACHOS L_peak)
> - Memory: `max(Mem(big), Mem(little))` -- since only one runs at a time
>
> **OPEN QUESTION:** FLOPs/MACs as cost proxy, or hardware-measured latency? EDANAS's ADA MACs accounts for early-exit savings; we need an analogous metric for routing. Hardware latency is more realistic but requires lookup tables per target device.

### 3.6 Edge-Aware Optimization

> **TO DETERMINE: Hardware cost model.**
> - FLOPs/MACs proxy (hardware-agnostic, fast during search)
> - Latency lookup table per device (as in OFA/ProxylessNAS -- measure each operation on target hardware)
> - MCUNet-style two-stage (first optimize search space for target constraints, then search within it)
>
> FLOPs are poor proxies for actual latency (Benmeziane et al., 2021). For Jetson Nano (GPU), FLOPs-latency correlation is reasonable; for Raspberry Pi (CPU), less so.
>
> **OPEN QUESTION:** Should we also constrain model size (flash storage) for MCU-class targets, or focus on Jetson/RPi?

---

## 4. Experimental Setup

### 4.1 Datasets

> **TO DETERMINE: Dataset selection and prioritization.**
>
> - **CIFAR-10** (60K, 10 classes, 32x32): Fast iteration, universal baseline. *Concern:* 32x32 limits multi-resolution experiments; even small models achieve >95%, which may not stress the router. Use primarily for debugging and ablations.
> - **CIFAR-100** (60K, 100 classes, 32x32): Same compute cost but much harder. "Easy" classes route to little, "hard" classes to big. **Should be the primary development dataset.**
> - **ImageNet-100** (100-class subset, ~130K images, 224x224): Realistic resolution, meaningful multi-resolution experiments. *Question:* Which 100 classes? Random, mini-ImageNet, or curated difficulty range?
> - **Visual Wake Words** (115K, 2 classes, from COCO): Edge-specific benchmark from MLPerf Tiny. Binary classification is the simplest routing scenario. Use for edge deployment validation only.
>
> **OPEN QUESTION:** Do we need a detection dataset (COCO) to demonstrate generality, or is that out of scope? Extending to detection requires non-trivial framework changes.

### 4.2 Baselines

> **TO DETERMINE: Baseline set that addresses each aspect of our contribution.**
>
> Each baseline answers a specific question:
>
> | Baseline | Question It Answers |
> |----------|-------------------|
> | MobileNetV3-Small/Large (static) | Is dynamic inference worth the complexity? |
> | EfficientNet-B0 (static) | Can compound scaling match dynamic inference? |
> | MobileNetV3-Small→Large (independently trained cascade) | **Does co-training matter?** (most critical baseline) |
> | OFA sub-network pair (cascaded) | Does our search add value beyond what OFA provides? |
> | MSDNet (early-exit) | Is big/little routing better than early-exit? |
> | BranchyNet-style exits on MobileNetV3 | Does NAS-searched architecture beat manual early-exit? |
> | EDANAS/NACHOS (if reproducible) | Do we advance beyond NAS + dynamic inference SOTA? |
>
> **The independently-trained cascade is the most critical baseline.** The supervisor's project description explicitly states current approaches "cascade two separate models" trained independently. If co-training doesn't beat this, the thesis is undermined.
>
> **OPEN QUESTION:** How to ensure fair comparison with early-exit baselines? Early-exit models see every sample progressively; big/little routing makes a binary decision. Compare at equal *expected* cost per sample.

### 4.3 Evaluation Metrics

> **TO DETERMINE: Full metric set.**
>
> **Accuracy:** Top-1, Top-5, per-class accuracy (to analyze what the router learns).
>
> **Efficiency:**
> - Average MACs/FLOPs: `E[MACs] = p * MACs_little + (1-p) * MACs_big` (key metric)
> - ADA MACs (following EDANAS): weighted-average MACs by exit fraction
> - Peak MACs (worst-case = big model cost, relevant for real-time guarantees)
>
> **Edge performance:** Latency (ms), throughput (img/s), peak memory (MB), energy (mJ), model size (MB).
>
> **Dynamic inference metrics:**
> - Routing ratio (% to little vs. big; ideally 60-80% to little)
> - Routing accuracy (when routed to little, was little correct? when to big, was big actually needed?)
> - Oracle routing gap (run both models on every test sample, compare oracle decisions vs. router decisions)
> - Confidence calibration (ECE)
>
> **Search cost:** GPU-hours on specified hardware.
>
> **The Pareto front plot (accuracy vs. average MACs) is the most important figure in the paper.** Our Pareto front must dominate the baselines' fronts.

### 4.4 Edge Hardware

> **TO DETERMINE: Hardware targets and measurement protocols.**
>
> - **Primary: Jetson Nano** (128 CUDA cores, 4GB RAM, TensorRT). Most capable edge target. Latency via CUDA events, memory via `nvidia-smi`, power via built-in INA3221 monitor.
> - **Secondary: Raspberry Pi 4** (ARM Cortex-A72, CPU-only). Pure CPU -- different performance characteristics. Tests whether our architectures generalize or overfit to GPU cost models.
> - **Stretch: Coral Dev Board / Android mobile** -- only if time permits.
>
> **OPEN QUESTIONS:**
> - Run search with ONE device's cost model and evaluate on all devices (OFA approach)? Or separate searches per device?
> - How fine-grained should latency lookup tables be? Per-operator or per-block?
> - For search (thousands of evaluations), use simulated latency from lookup tables rather than actual hardware.

### 4.5 Implementation Details

> **TO DETERMINE: Training hyperparameters.**
>
> - Framework: PyTorch 2.0+, timm library, Weights & Biases / TensorBoard
> - Deployment: ONNX export, TensorRT optimization, INT8/FP16 quantization
>
> **Supernet training (needs tuning):**
> - Optimizer: SGD with momentum 0.9, weight decay 3e-5 (following OFA/BigNAS)
> - LR: cosine decay from 0.025 (CIFAR) or scaled linear (ImageNet-100)
> - Batch size: 64 (CIFAR), 256-512 (ImageNet-100)
> - Epochs: 300 (CIFAR), 150-250 (ImageNet-100)
> - Progressive shrinking schedule: kernel (75 epochs) -> depth (75 epochs) -> width (75 epochs)
>
> **Architecture search:**
> - NSGA-II: population 50, 30-50 generations (following EDANAS)
> - Evaluation via supernet weights (no retraining)
>
> **OPEN QUESTION:** After search, retrain discovered pair from scratch or fine-tune from supernet weights? OFA and BigNAS do not retrain; DARTS-style methods retrain from scratch.
>
> **Compute budget estimate:** ~200-400 GPU-hours for full plan. If over budget, prioritize CIFAR-100, reduce ImageNet-100 to single seed, drop stretch ablations.

### 4.6 Ablation Studies

> **TO DETERMINE: Ablation priority order.**
>
> Ranked by expected informativeness:
>
> 1. **Co-training vs. independent training** (MOST CRITICAL -- directly tests core thesis)
>    - Train big/little independently, cascade them. Compare against co-trained pair.
>    - If co-training does not help, the project needs to pivot.
>
> 2. **Router ablation**
>    - No router (always big): upper bound on accuracy
>    - No router (always little): upper bound on efficiency
>    - Random routing: is the router learning beyond a coin flip?
>    - Confidence vs. learned vs. difficulty-aware
>    - Oracle routing: run both models, pick correct one -- upper bound on any router
>
> 3. **Search space coupling** -- nested vs. sibling vs. partially overlapping architectures
>
> 4. **Distillation ablation** -- no distillation vs. logit-only vs. logit + feature distillation
>
> 5. **Search algorithm comparison** -- evolutionary vs. DrNAS vs. random search (sanity check)
>
> 6. **Progressive shrinking vs. single-stage training** -- compare little model quality
>
> 7. **Loss term weighting sensitivity** -- sweep alpha, beta, gamma, lambda

---

## 5. Results

> **This section will contain the experimental results. The key tables, figures, and analyses needed are outlined below. Each subsection identifies what "success" and "negative results" would look like.**

### 5.1 Main Accuracy-Efficiency Comparison

> **Table 1: Main comparison on CIFAR-10/CIFAR-100** -- all models, all metrics.
>
> **Success benchmark:** Our dynamic model achieves higher accuracy than EDANAS (>80.9%) at equal or lower ADA MACs (<=17.31M) on CIFAR-10, OR comparable accuracy at significantly lower ADA MACs. Co-trained pair outperforms independently-trained cascade by >=1-2% at matched compute.
>
> **Negative result indicator:** Co-training degrades big model accuracy compared to independent training, or dynamic system underperforms simple cascade at all operating points.

### 5.2 Accuracy-Efficiency Pareto Frontier

> **Figure 1 (most important figure):** Three subplots -- (a) Accuracy vs. FLOPs, (b) Accuracy vs. Latency (Jetson), (c) Accuracy vs. Energy.
>
> Show: our dynamic Pareto front (swept across tau), static baselines as points, cascaded baseline front, MSDNet front, EDANAS front. Reference anchors: BigNAS range (76.5-80.9% at 242-1040 MFLOPs).
>
> **Success:** Our Pareto front dominates cascade and early-exit baselines across the middle 50% of the compute range.

### 5.3 NAS Strategy Comparison

> **Table 2:** Strategy, Best Top-1, Search Cost (GPU-hrs), Architectures Evaluated.
>
> **Key question:** Does one strategy clearly dominate, or do they occupy distinct niches?
> **Surprising result:** If OFA-style matches evolutionary/differentiable despite less explicit optimization -- suggests co-training matters more than search algorithm.

### 5.4 Ablation Studies

> **5.4.1 Co-training vs. Independent Training** (Table 3 -- the single most important table)
> - Independent, co-trained without distillation, co-trained with distillation, co-trained + progressive shrinking
> - Reference: US-Nets show +2.2% from inplace distillation. Hypothesis: similar or larger benefit for big/little pair.
> - Key question: Does co-training improve little (via distillation), big (via regularization), or both?
>
> **5.4.2 Routing Strategy Comparison** (Table 4)
> - Confidence, learned MLP, difficulty-aware. Include router overhead (MACs) and calibration (ECE).
> - Reference: BlockDrop policy overhead 3-5%; SkipNet RNNGate 0.04%.
> - Key question: Does learned router outperform confidence thresholding enough to justify overhead?
>
> **5.4.3 Search Space Dimensions** (Table 5)
> - Width only, depth only, kernel only, full combination.
> - Reference: RegNet finding that depth stabilizes at ~20 blocks. Hypothesis: width is most impactful (following Slimmable/US-Nets).
>
> **5.4.4 Loss Function Components** (Table 6)
> - Full loss, without L_routing, without L_consistency, without distillation, sensitivity to weights.
>
> **5.4.5 Distillation Temperature** (Table 7)
> - T in {1, 2, 4, 8, 16, 20}. Reference: Hinton uses T=20; optimal for inplace supernet distillation may differ.

### 5.5 Edge Deployment Results

> **Table 8-9: Jetson Nano and Raspberry Pi 4 deployment.**
> - All models in FP32 and FP16/INT8 quantized variants.
> - Include TensorRT-optimized and ONNX runtime numbers.
>
> **Success:** Dynamic model achieves >=1.5x latency reduction over big-only at <1% accuracy loss.
> **Key concern:** Does routing overhead (conditional branching, memory transfers) erode theoretical FLOPs savings on real hardware? MCUNet showed peak activation memory is the binding constraint on MCUs -- similar mismatches may appear.
>
> **Figure 2: Latency vs. FLOPs scatter** -- shows correlation (or lack thereof) between theoretical FLOPs and measured latency per platform. Separate for Jetson (GPU) vs. RPi (CPU).

### 5.6 Dynamic Behavior Analysis

> **Figure 3: Exit distribution histogram** -- fraction of samples to little vs. big, stratified by class/difficulty. Reference: BiLD reports ~80% of small model predictions match large.
>
> **Figure 4: Confidence calibration** (reliability diagram) -- ECE for little model. Critical for confidence-based routing.
>
> **Figure 5: Threshold sensitivity curve** -- accuracy and FLOPs vs. tau. Is the curve smooth (robust) or sharp (fragile)?
>
> **Figure 6: Qualitative examples** -- sample images colored by routing decision, including failure cases.
>
> **Table 10: Per-class exit rate** -- which classes are "easy" (high exit) vs. "hard" (low exit)?

---

## 6. Discussion

### 6.1 Key Findings

> **Outline of expected findings to discuss (3-5 paragraphs, each tied to a table/figure):**
>
> 1. **Co-training enables representation sharing that independent training cannot.** Discuss magnitude from ablation. If little model improves more than big: the big model acts as a richer teacher. If both improve: L_consistency acts as mutual regularizer. Connect to US-Nets' +2.2% improvement.
>
> 2. **The Pareto frontier shifts with NAS-driven co-training.** Quantify how far the frontier shifts vs. cascaded baselines. Compare to EDANAS/NACHOS numbers. If improvement is modest (<1%): argue value is in unified framework. If large (>2%): validate joint search as fundamentally superior.
>
> 3. **NAS strategy trade-offs.** Expected: OFA-style most practical, evolutionary finds marginally better architectures at higher cost, differentiable is fast but may suffer collapse.
>
> 4. **Routing strategy matters less/more than expected.** If confidence routing is competitive: aligns with BranchyNet/MSDNet literature. If learned routing wins: discuss what MLP learns beyond confidence (feature-level difficulty signals, model diversity as in FrugalGPT).
>
> 5. **The FLOPs-to-latency gap.** Where FLOPs reduction doesn't translate to proportional latency reduction. Connect to RegNet (activations dominate GPU time) and MCUNet (peak memory dominates MCU constraints). Argue for hardware-in-the-loop NAS.

### 6.2 Surprising Results to Watch For

> - Little model accuracy exceeding its independently-trained equivalent (mirrors FrugalGPT: cascading can improve accuracy, not merely reduce cost)
> - Certain "hard" classes handled better by little (FrugalGPT: 6% of samples, cheapest model beats GPT-4)
> - NAS strategies converging on similar architectures (would suggest search space dominates over algorithm)
> - Quantization disproportionately affecting the dynamic system (changing optimal routing thresholds)
> - Router becoming a bottleneck on the most constrained hardware (RPi CPU-only)

### 6.3 Limitations

> **Honest assessment to address:**
> - Dataset scope: vision classification only. No evidence for detection, segmentation, or non-vision tasks.
> - Hardware scope: 2-4 edge platforms out of hundreds in practice.
> - Search cost: if pipeline requires multiple GPU-days, practical advantage over training two models separately must be weighed. Search cost amortizes only if discovered architecture is deployed at scale.
> - Two-model limit: binary big/little is a simplification. Optimal may be a continuum (US-Nets arbitrary widths, MSDNet multiple exits).
> - Static router at deployment: no on-device adaptation to distribution shift.
> - CIFAR-10 ceiling effects: accuracy saturates above ~97%. Need CIFAR-100/ImageNet-100 for meaningful differentiation.
> - Supernet training sensitivity: BigNAS required 5 specific techniques to work. Report variance across seeds.

### 6.4 Future Work

> - Multi-exit generalization: extend from 2 models to N models with NAS-searched exit placement
> - Task generalization: detection (YOLO/SSD), segmentation, NLP (following BiLD)
> - On-device NAS: combine with NASWOT-style zero-cost proxies for on-device architecture adaptation
> - Hardware co-design: jointly optimize neural architecture and accelerator dataflow (following NAAS/CIMNAS) for dynamic inference workloads
> - Compiler co-optimization: extend NACOS to dynamic architectures (different inputs traverse different paths)
> - Multi-device cascading: little model on MCU (MCUNet constraints), hard samples offloaded to GPU-equipped edge node (following EdgeFM)
> - Continual learning: adapt router to distribution shift over time

---

## 7. Conclusion

> **Structure: 3-4 paragraphs.**
>
> **Paragraph 1: Problem and approach.**
> Current dynamic inference approaches train components independently, preventing joint optimization. We proposed a NAS framework for co-training coupled big/little sub-networks within a unified supernet with an input-adaptive router.
>
> **Paragraph 2: Key results.** (fill with actual numbers)
> - On CIFAR-10/100: accuracy at ADA MACs vs. EDANAS (80.9% at 17.31M) and NACHOS (72.65% at 2.44M)
> - Co-training improvement over independent training vs. US-Nets reference (+2.2%)
> - Edge deployment: latency, speedup, accuracy loss on Jetson Nano and Raspberry Pi
> - NAS strategy winner and search cost
>
> **Paragraph 3: Broader significance.**
> First framework to jointly search big/little architectures (not just exit classifiers on fixed backbone). Open-source framework enables hardware-specialized dynamic model discovery.
>
> **Paragraph 4: Scope and future.**
> Limited to vision classification, small hardware set. Binary big/little is a first step; multi-exit extensions natural. Key insight -- co-training produces better pairs than independent training -- likely generalizes across modalities.

---

## References

<!-- TODO: Maintain as BibTeX, convert for final paper -->

### Neural Architecture Search

1. Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. *ICLR 2017*.
2. Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized Evolution for Image Classifier Architecture Search. *AAAI 2019*.
3. Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable Architecture Search. *ICLR 2019*.
4. Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020). Once-for-All: Train One Network and Specialize it for Efficient Deployment. *ICLR 2020*.
5. Yu, J., Jin, P., Liu, H., Bender, G., Kindermans, P.-J., Tan, M., ... & Le, Q. V. (2020). BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models. *ECCV 2020*.
6. Guo, Z., Zhang, X., Mu, H., Heng, W., Liu, Z., Wei, Y., & Sun, J. (2020). Single Path One-Shot Neural Architecture Search with Uniform Sampling. *ECCV 2020*.
7. Liang, J., Zhu, K., Li, Y., Li, Y., & Gong, Y. (2024). Multi-Objective Evolutionary Neural Architecture Search with Weight-Sharing Supernet. *Applied Sciences, 14*(14), 6143.
8. Anonymous. (2025). Neural Architecture Search by Learning a Hierarchical Search Space. Under review, *ICLR 2025*.
9. HPE-DARTS: Hybrid Pruning and Proxy Evaluation in Differentiable Architecture Search. (2025). *ICAART 2025*.
10. Finkler, U., Merler, M., Panda, R., et al. (2021). Large Scale Neural Architecture Search with Polyharmonic Splines. *AAAI 2021 Workshop*.
11. Chen, X., Wang, R., Cheng, M., Tang, X., & Hsieh, C.-J. (2021). DrNAS: Dirichlet Neural Architecture Search. *ICLR 2021*.
12. Mellor, J., Turner, J., Storkey, A., & Crowley, E. J. (2021). Neural Architecture Search without Training. *ICML 2021*.
13. Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollar, P. (2020). Designing Network Design Spaces. *CVPR 2020*.

### Knowledge Distillation and Width-Variable Training

14. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS 2014 Deep Learning Workshop*. arXiv:1503.02531.
15. Yu, J., & Huang, T. (2019). Slimmable Neural Networks. *ICLR 2019*. arXiv:1812.08928.
16. Yu, J., & Huang, T. (2019). Universally Slimmable Networks and Improved Training Techniques. *arXiv:1903.05134*.

### Dynamic Inference and Early Exit

17. Teerapittayanon, S., McDanel, B., & Kung, H. T. (2017). BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. *arXiv:1709.01686*.
18. Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., & Weinberger, K. Q. (2018). Multi-Scale Dense Networks for Resource Efficient Image Classification. *ICLR 2018*.
19. Wu, Z., Nagarajan, T., Kumar, A., Rennie, S., Davis, L. S., Grauman, K., & Feris, R. (2018). BlockDrop: Dynamic Inference Paths in Residual Networks. *CVPR 2018*.
20. Wang, X., Yu, F., Dou, Z.-Y., Darrell, T., & Gonzalez, J. E. (2018). SkipNet: Learning Dynamic Routing in Convolutional Networks. *ECCV 2018*.
21. Han, Y., Huang, G., Song, S., Yang, L., Wang, H., & Wang, Y. (2021). Dynamic Neural Networks: A Survey. *IEEE TPAMI, 44*(11), 7436--7456.
22. Laskaridis, S., Kouris, A., & Lane, N. D. (2021). Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions. *EMDL Workshop 2021*.
23. Zou, M., Li, X., Fang, J., Wen, H., & Fang, W. (2023). Dynamic Deep Neural Network Inference via Adaptive Channel Skipping. *Turkish J. of EE & CS, 31*(5).
24. Bajpai, D. J., & Hanawal, M. K. (2025). A Survey of Early Exit Deep Neural Networks in NLP. *arXiv:2501.07670*.
25. Yang, L., Han, Y., Chen, X., Song, S., Dai, J., & Huang, G. (2020). Resolution Adaptive Networks for Efficient Inference. *CVPR 2020*.

### Big/Little Model Collaboration

26. Kim, S., Mangalam, K., Moon, S., Malik, J., Mahoney, M. W., Gholami, A., & Keutzer, K. (2023). Speculative Decoding with Big Little Decoder. *NeurIPS 2023*.
27. Yang, B., He, L., Ling, N., Yan, Z., Xing, G., et al. (2023). EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge. *SenSys 2023*.
28. Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. *arXiv:2305.05176*.
29. Chen, Y., Zhao, J., & Han, H. (2025). A Survey on Collaborative Mechanisms Between Large and Small Language Models. *arXiv:2505.07460*.
30. Wang, F., Chen, J., Yang, S., et al. (2025). A Survey on Collaborating Small and Large Language Models. *arXiv:2510.13890*.

### Hardware-Aware NAS and Edge Deployment

31. Benmeziane, H., El Maghraoui, K., Ouarnoughi, H., Niar, S., Wistuba, M., & Wang, N. (2021). A Comprehensive Survey on Hardware-Aware Neural Architecture Search. *arXiv:2101.09336*.
32. Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for MobileNetV3. *ICCV 2019*.
33. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
34. Wu, B., Dai, X., Zhang, P., Wang, Y., Sun, F., Wu, Y., ... & Keutzer, K. (2019). FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search. *CVPR 2019*.
35. Cai, H., Zhu, L., & Han, S. (2019). ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. *ICLR 2019*.
36. Tu, X., Chen, D., Altintas, O., Han, K., & Wang, H. (2025). PlatformX: An End-to-End Transferable Platform for Energy-Efficient Neural Architecture Search. *SEC 2025*.
37. Mao, S., Qin, M., Dong, W., Liu, H., & Gao, Y. (2025). RAM-NAS: Resource-aware Multiobjective Neural Architecture Search. *IROS 2024*.
38. Lin, Y., Yang, M., & Han, S. (2021). NAAS: Neural Accelerator Architecture Search. *DAC 2021*.
39. Lin, J., Chen, W.-M., Lin, Y., Cohn, J., Gan, C., & Han, S. (2020). MCUNet: Tiny Deep Learning on IoT Devices. *NeurIPS 2020*.
40. Krestinskaya, O., Fouda, M. E., Eltawil, A., & Salama, K. N. (2025). CIMNAS: A Joint Framework for Compute-In-Memory-Aware Neural Architecture Search. *arXiv:2509.25862*.
41. Bachiri, I., Benmeziane, H., Ouarnoughi, H., Baghdadi, R., Niar, S., & Aries, A. (2024). Combining Neural Architecture Search and Automatic Code Optimization: A Survey. *arXiv:2408.04116*.
42. Wang, T., Wang, K., Cai, H., Lin, J., Liu, Z., Wang, H., Lin, Y., & Han, S. (2020). APQ: Joint Search for Network Architecture, Pruning and Quantization Policy. *CVPR 2020*.
43. Al Youssef, H., Awada, S., Raad, M., Valle, M., & Ibrahim, A. (2025). Combining NAS and Weight Reshaping for Optimized Embedded Classifiers in Multisensory Glove. *Sensors, 25*(20), 6142.
44. Gupta, D., Lee, R. D., & Wynter, L. (2024). On Efficient Object-Detection NAS for ADAS on Edge Devices. *IEEE CAI 2024*.
45. Edge Intelligence: A Review of Deep Neural Network Inference for Edge Devices. (2025). *Electronics, 14*(12), 2495.

### NAS for Dynamic Inference

46. Gambella, M., & Roveri, M. (2023). EDANAS: Adaptive Neural Architecture Search for Early Exit Neural Networks. *IJCNN 2023*.
47. Gambella, M., Pomponi, J., Scardapane, S., & Roveri, M. (2025). NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks. *arXiv:2401.13330*.
48. Sponner, M., Servadei, L., Waschneck, B., Wille, R., & Kumar, A. (2024). Efficient Post-Training Augmentation for Adaptive Inference in Heterogeneous and Distributed IoT Environments. *arXiv:2403.07957*.
49. Casarin, F. (2025). NAS Just Once: Neural Architecture Search for Joint Image-Video Recognition. *ICCVW 2025*.

---

## Appendix

### A. Hyperparameter Settings

> Full reproducibility tables needed:
> - A.1 Supernet training: optimizer, LR, schedule, weight decay, batch size, epochs, warmup, dropout, seeds. Separate entries for CIFAR-10, CIFAR-100, ImageNet-100. Progressive shrinking schedule.
> - A.2 NAS search: per-strategy parameters (DrNAS architecture LR, temperature, regularization; NSGA-II population, generations, tournament size, mutation/crossover rates; OFA predictor architecture and training).
> - A.3 Loss function: values of alpha, beta, gamma, lambda, T. How they were tuned (grid search ranges, final values).
> - A.4 Router: MLP architecture, difficulty-aware features, confidence threshold sweep values.
> - A.5 Edge deployment: ONNX export params, TensorRT optimization level, quantization calibration (samples, algorithm), inference protocol (warmup, timed iterations, batch size).

### B. Architecture Visualizations

> - B.1 Discovered big sub-network: DAG diagram with per-block choices, annotated with FLOPs per block.
> - B.2 Discovered little sub-network: same format, highlight where big and little diverge most.
> - B.3 Cross-strategy comparison: side-by-side architectures from differentiable, evolutionary, OFA-style search. Highlight structural commonalities and differences.
> - B.4 Weight sharing visualization: heatmap of weight magnitude differences between big and little at each layer.

### C. Additional Results

> - C.1 Per-class accuracy breakdown: little-only, big-only, dynamic columns for all classes.
> - C.2 Variance across seeds: full 3-seed results with mean +/- std.
> - C.3 Sandwich rule variants: {min, max, 2 random} vs. {min, max, 1 random} vs. {min, max only}.
> - C.4 Training curves: loss (total + components), accuracy (big + little), exit rate evolution.
> - C.5 Confidence distribution histograms: per-class, before and after temperature calibration.
> - C.6 NASWOT validation: rank correlation between zero-cost scores and trained accuracy for 100 sub-networks.

### D. Edge Deployment Details

> - D.1 Device specifications: software versions (JetPack, CUDA, TensorRT, ONNX Runtime, PyTorch), OS, thermal config, power measurement setup.
> - D.2 Quantization impact: FP32 vs. FP16 vs. INT8 accuracy and latency per model per device. Layer-by-layer quantization sensitivity.
> - D.3 Memory profiling: peak activation memory (per MCUNet insight), parameter memory, runtime overhead. Memory timeline plot.
> - D.4 Thermal throttling: sustained throughput under continuous inference, temperature curves, p50/p95/p99 latency.
> - D.5 Router overhead breakdown: microsecond-level profiling of feature extraction, router forward pass, branching logic. Percentage of total inference time.

### E. Search Space Analysis

> - E.1 Search space cardinality: formal computation of total (big, little) architecture pairs.
> - E.2 Sub-network distribution: sample 1000 random sub-networks, plot FLOPs and accuracy distributions.
> - E.3 Architecture correlation: correlation between big and little accuracy across sampled pairs (expected positive from weight sharing).
