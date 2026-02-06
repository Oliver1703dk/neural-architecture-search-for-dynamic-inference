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

DARTS (Liu et al., 2019) introduced differentiable NAS by relaxing the discrete search space into a continuous one, enabling gradient-based optimization. This reduced search cost to 1.5 GPU-days on CIFAR-10 while achieving 2.76% test error. However, DARTS suffers from performance collapse and high memory usage. HPE-DARTS (2025) addresses these issues through hybrid pruning with a proxy evaluation strategy (NetPerfProxy), completing search in 0.61 hours on NAS-Bench-201 with competitive accuracy. DrNAS, P-DARTS, and beta-DARTS further improve DARTS stability and search-evaluation gap issues.

#### 2.1.3 Supernet and One-Shot Methods

One-shot methods decouple supernet training from architecture search through weight sharing. **Once-for-All (OFA)** (Cai et al., 2020) trains a single supernet supporting many sub-networks through progressive shrinking of kernel size, depth, and width. After training, sub-networks can be extracted without retraining, enabling deployment across diverse hardware platforms. **BigNAS** (Yu et al., 2020) scaled single-stage NAS with five key techniques: the sandwich rule (sampling smallest, largest, and random sub-networks per batch), in-place distillation from the largest to smaller sub-networks, modified BatchNorm initialization, exponential learning rate decay, and regularization applied only to the largest model. BigNAS achieves 76.5--80.9% top-1 accuracy on ImageNet at 242--1040 MFLOPs without any post-search retraining. SPOS (Guo et al., 2020) used uniform path sampling to further decouple supernet training from architecture search. A hierarchical MCTS-based approach (under review, ICLR 2025) learns the tree structure via agglomerative clustering of architecture output vectors, finding globally optimal architectures in NAS-Bench-Macro.

#### 2.1.4 Sample-Efficient and Zero-Cost Methods

Recent work has focused on reducing the number of evaluations needed. Finkler et al. (2021) proposed polyharmonic spline interpolation for NAS, requiring only 2d+3 evaluations for a d-dimensional search space. Applied to ResNet18 on ImageNet-22K (14M images, 21,841 classes), it achieved a 3.13% absolute improvement over state-of-the-art with just 15 evaluations exploring approximately 3 trillion configurations. Zero-cost proxies like NASWOT estimate architecture quality without training, enabling rapid candidate screening.

### 2.2 Dynamic Neural Networks and Dynamic Inference

Dynamic neural networks adapt their structure or parameters to each input at inference time, offering a promising path to efficient edge deployment. Han et al. (2021) provide a comprehensive survey categorizing dynamic networks into three types: instance-wise (adapting per sample), spatial-wise (adapting per spatial location), and temporal-wise (adapting across time steps). We focus on the instance-wise category most relevant to our work.

#### 2.2.1 Early-Exit Networks

Early-exit networks augment a backbone with intermediate classifiers that allow test samples to exit before reaching the final layer when the network is sufficiently confident. **BranchyNet** (Teerapittayanon et al., 2017) is a seminal work establishing entropy-based early exiting: side branches consisting of convolutional and fully-connected layers produce softmax predictions, and if the entropy falls below a threshold, the sample exits. On MNIST, 94.3% of samples exit at the first branch with 5.4x CPU speedup. The paper explicitly identifies the need to "derive an algorithm to find the optimal placement locations of the branches automatically" -- precisely the gap NAS can fill. **MSDNet** (Huang et al., 2018) introduced multi-scale dense connectivity for anytime prediction with strong accuracy-efficiency trade-offs. Laskaridis et al. (2021) provide a comprehensive taxonomy of early-exit networks, decomposing the design into backbone selection, exit architecture, exit placement, training strategies (end-to-end vs. IC-only with frozen backbone), and exit policies (entropy, confidence, patience-based, learnable). They identify NAS-based exploration of the early-exit design space as an open and promising research direction.

A complementary survey on early exit methods in NLP (Bajpai & Hanawal, 2025) covers exit criteria including confidence-based (DeeBERT), patience-based (PABEE), distribution-based (PALBERT), and ensemble methods, as well as threshold selection strategies ranging from static validation-set-based to dynamic Multi-Armed Bandit approaches. The cross-domain applicability of early-exit mechanisms confirms their generality for dynamic inference.

#### 2.2.2 Channel-Level and Layer-Level Adaptive Computation

Beyond early exits, dynamic inference can operate at finer granularities. **Adaptive Channel Skipping (ACS)** (Zou et al., 2023) introduces channel-level dynamic inference, where a gating network produces binary decisions per channel based on input features. ACS achieves 62% FLOPs reduction on CIFAR-10 with DenseNet-41 while improving accuracy from 86.70% to 88.64%. An enhanced variant, ACS-DG, uses dynamic grouping convolutions to reduce the gating network's own cost by up to 50.91%. SkipNet and BlockDrop learn to selectively skip entire layers or blocks conditioned on the input, while Resolution Adaptive Networks adjust input resolution dynamically.

#### 2.2.3 Big/Little Model Cascading

Cascaded approaches route inputs through models of different sizes. **BiLD** (Kim et al., 2023) coordinates a small and large decoder model for text generation: the small model generates tokens until its confidence drops below a fallback threshold, then the large model corrects predictions in parallel. This achieves up to 2.12x speedup with approximately 1 point quality degradation. The observation that ~80% of small model predictions match the large model validates the efficiency of conditional computation. For vision, EdgeFM (Yang et al., 2023) deploys lightweight CNNs on edge devices with a cloud-hosted foundation model backup, achieving 3.2x latency reduction and 34.3% accuracy improvement through dynamic model switching based on input uncertainty and network conditions.

Broader surveys on small-large model collaboration (Chen et al., 2025; Wang et al., 2025) catalog collaboration modes including pipeline processing, hybrid routing, auxiliary enhancement, and knowledge distillation, with cascade routing frameworks like FrugalGPT reducing cost while maintaining quality. These patterns are directly applicable to dynamic inference systems where computation is adaptively allocated between models of different sizes.

### 2.3 Hardware-Aware NAS and Edge Deployment

Deploying neural networks on edge devices requires explicitly accounting for hardware constraints during architecture design. Benmeziane et al. (2021) provide the first comprehensive survey of hardware-aware NAS (HW-NAS), categorizing methods by search space (architecture and hardware), search strategy (RL, EA, gradient-based), acceleration techniques (weight sharing, accuracy predictors), and hardware cost estimation (real-time measurement, lookup tables, analytical models, learned predictors). They identify the key insight that FLOPs are poor proxies for actual latency on diverse hardware.

#### 2.3.1 Latency and Energy-Aware Search

**FBNet** (Wu et al., 2019) performs differentiable NAS with hardware-aware loss functions. **ProxylessNAS** (Cai et al., 2019) directly searches on the target task and hardware, removing the proxy-dataset gap. **PlatformX** (Tu et al., 2025) is a fully automated HW-NAS framework with transferable kernel-level energy predictors that generalize across edge devices with only 50--100 calibration samples. On CIFAR-10, it discovers models achieving up to 72% lower energy consumption than NAS-Bench-201 baselines. **RAM-NAS** (Mao et al., 2025) targets robotic edge hardware (NVIDIA Jetson AGX Orin, Xavier, Xavier NX) with a mutual distillation strategy where all subnets distill from each other using Decoupled Knowledge Distillation loss, achieving 76.7--81.4% ImageNet top-1 accuracy with latency-optimal models for each hardware platform.

#### 2.3.2 Neural Architecture and Hardware Co-Design

**NAAS** (Lin et al., 2021) jointly optimizes the neural network architecture, accelerator architecture (PE connectivity and dataflow), and compiler mapping strategy using CMA-ES, achieving 2.6--4.4x speedup over Eyeriss/NVDLA. **CIMNAS** (Krestinskaya et al., 2025) extends co-design to Compute-in-Memory hardware, jointly searching neural architecture, quantization policy, and CIM hardware parameters over a 9.9 x 10^85 search space, achieving 90--104.5x EDAP reduction without accuracy loss. The NACOS survey (Bachiri et al., 2024) systematizes the joint optimization of NAS and automatic code optimization, demonstrating that independent optimization of architecture and compiler schedule is sub-optimal and proposing taxonomies for two-stage and one-stage co-search methods.

#### 2.3.3 Edge Deployment Techniques

A comprehensive review of DL inference for edge intelligence (2025) covers model compression (pruning, quantization, knowledge distillation), embedded AI hardware platforms, and memory optimization. The review identifies NAS for edge inference as a key open challenge, emphasizing the need for hardware-aware architectures tailored to diverse edge platforms. Practical deployment pipelines include: APQ (Wang et al., 2020) jointly searching architecture, pruning policy, and quantization at the sub-network level; MCUNet achieving >70% ImageNet top-1 on off-the-shelf microcontrollers; and Al Youssef et al. (2025) combining HW-NAS with weight reshaping and quantization for deployment on ultra-low-power MCUs (512KB Flash, 96KB SRAM), achieving 87% inference time reduction and 89% energy reduction. Gupta et al. (2024) apply supernet-based NAS with in-place Pearson Correlation distillation to produce palettes of object detection models for ADAS edge deployment with limited training data.

### 2.4 NAS for Dynamic Inference

The most directly relevant prior works combine NAS with dynamic inference mechanisms, though this intersection remains underexplored.

**EDANAS** (Gambella & Roveri, 2023) is the first NAS framework that jointly designs Early Exit Neural Network (EENN) architectures and exit selection parameters. Built on the OFA supernet with NSGA-II search, it introduces ADA MACS -- a weighted-average MACs metric capturing the actual computational cost of early-exit inference, where weights reflect the fraction of samples exiting at each classifier. On CIFAR-10, EDANAS achieves 80.9% accuracy with 17.31M ADA MACS (vs. 21.67M for a single-exit NAS backbone). However, EDANAS is limited to coarse threshold values ({0.1, 0.2, 1.0}), short candidate training (5 epochs), and significant accuracy degradation with more than 3 exit points.

**NACHOS** (Gambella et al., 2025) extends EDANAS as the first NAS framework jointly co-designing backbone and early exit classifiers under user-defined MAC constraints. It introduces two novel regularization terms: L_cost (enforcing computational constraints during training) and L_peak (aligning exit confidence scores with operating points via a Support Matrix of per-class accuracies). NACHOS achieves 72.65% accuracy at 2.44M MACs on CIFAR-10 (vs. 67.78% for EDANAS), using fewer exit classifiers (2.80 vs. 4.60 on average) while producing EENNs whose accuracy exceeds the backbone-only accuracy. Search cost is 2 GPU-days on a single NVIDIA A40.

**Sponner et al. (2024)** propose a post-training augmentation framework that converts pretrained models into EENNs, formulating threshold configuration as a shortest-path problem solved via Bellman-Ford. The framework runs on a laptop CPU and maps EENNs to heterogeneous multi-processor IoT platforms (e.g., Cortex-M0 + Cortex-M4F), achieving 59--78% MACs reduction with minimal accuracy loss.

### 2.5 Gap in Existing Work

Despite significant progress in each individual area, a critical gap remains at their intersection. Table 1 summarizes the coverage of key works across the three dimensions central to our project.

| Work | NAS-Driven Architecture | Dynamic Inference | Hardware-Aware Edge |
|------|:-----------------------:|:-----------------:|:-------------------:|
| DARTS / BigNAS / OFA | Yes | No | Partial |
| BranchyNet / MSDNet | No | Yes (early exit) | No |
| ACS (Zou et al.) | No | Yes (channel skip) | No |
| FBNet / ProxylessNAS | Yes | No | Yes |
| PlatformX / RAM-NAS | Yes | No | Yes |
| NAAS / CIMNAS | Yes | No | Yes (co-design) |
| EDANAS / NACHOS | Yes | Yes (early exit) | Partial (MACs only) |
| Sponner et al. | Partial (post-hoc) | Yes (early exit) | Yes (MCU) |
| **Ours (proposed)** | **Yes** | **Yes (big/little + routing)** | **Yes (edge hardware)** |

Several specific gaps motivate our work:

1. **No joint big/little NAS.** EDANAS and NACHOS search for early exit classifiers on a fixed OFA backbone but do not jointly design the big and little sub-networks themselves. No existing NAS framework searches for coupled big/little architectures that are co-trained within a unified supernet.

2. **Limited routing mechanisms.** Current NAS-for-dynamic-inference methods rely solely on confidence thresholds for exit decisions. No work uses NAS to jointly optimize learned routing networks, confidence gating, or difficulty-aware routing alongside the architecture.

3. **Incomplete hardware awareness.** EDANAS and NACHOS use MACs as a hardware proxy but do not incorporate real latency or energy measurements from target edge devices. PlatformX and RAM-NAS provide hardware-aware NAS but do not support dynamic inference.

4. **No NAS + dynamic inference + compiler co-optimization.** The NACOS survey (Bachiri et al., 2024) demonstrates the importance of co-optimizing architecture and compiler schedules, but all existing NACOS methods target static architectures. Extending this to dynamic inference architectures where different inputs traverse different computational paths remains an open challenge.

Our work addresses these gaps by proposing a NAS framework that jointly discovers and co-trains coupled big/little dynamic inference models with an input-adaptive router, optimized for real edge hardware constraints.

---

## 3. Methodology

### 3.1 Overview

Our framework consists of three components: (1) a supernet search space encoding coupled big/little architectures, (2) a co-training strategy with joint loss optimization, and (3) an input-adaptive router. We compare three NAS strategies for architecture discovery within this framework.

<!-- TODO: Add overview figure showing the full pipeline -->

### 3.2 Search Space Design

The supernet is structured as follows:

```
Supernet
├── Shared Stem (fixed)
├── Searchable Blocks (N blocks)
│   ├── Width:     {0.25x, 0.5x, 0.75x, 1.0x}
│   ├── Depth:     {1, 2, 3, 4} layers per block
│   ├── Kernel:    {3x3, 5x5, 7x7}
│   └── Expansion: {1, 2, 4, 6}
├── Router Module (input-adaptive)
│   └── Confidence threshold → big/little selection
└── Dual Heads
    ├── Little classifier (early)
    └── Big classifier (final)
```

The "little" sub-network corresponds to narrow, shallow configurations (small width multipliers, fewer layers). The "big" sub-network uses wider, deeper configurations. Both share weights within the supernet, enabling joint optimization. The search space size is:

<!-- TODO: Calculate and report total search space cardinality -->

### 3.3 Co-Training Strategy

We employ four complementary training techniques:

**Sandwich Training.** Each batch trains the minimum, maximum, and two random sub-networks. This ensures the supernet performs well across the full spectrum of configurations.

**In-place Distillation.** The big sub-network's soft predictions serve as targets for the little sub-network, transferring knowledge without a separate distillation stage.

**Progressive Shrinking.** Training begins with the full network. Smaller configurations are progressively enabled, following the curriculum: kernel size → depth → width.

**Joint Loss.** The total loss combines four terms:

```
L = α · L_big + β · L_little + γ · L_routing + λ · L_consistency
```

Where:
- `L_big = CrossEntropy(y_big, y_true)` -- big model classification loss
- `L_little = CrossEntropy(y_little, y_true)` -- little model classification loss
- `L_routing = BCE(router_decision, optimal_decision) + η · efficiency_penalty` -- routing quality
- `L_consistency = KL(softmax(z_little / T), softmax(z_big / T))` -- distillation consistency

<!-- TODO: Report final hyperparameter values for α, β, γ, λ, T after tuning -->

### 3.4 Routing Mechanisms

We implement and compare three routing strategies:

**Confidence-based.** The little model produces a prediction with softmax confidence. If `max(softmax(z_little)) > τ`, the little model's prediction is used. Otherwise, the input is forwarded to the big model. The threshold `τ` controls the accuracy-efficiency trade-off.

**Learned router.** A small MLP takes intermediate features as input and outputs a binary routing decision. It is trained jointly with the main networks using the routing loss `L_routing`.

**Difficulty-aware.** The router is trained to classify inputs as "easy" or "hard" based on feature statistics (entropy, variance). Easy inputs are routed to the little model; hard inputs to the big model.

<!-- TODO: Add architecture details of the MLP router (layers, hidden dims) -->

### 3.5 NAS Strategies

#### 3.5.1 Differentiable NAS (DARTS-style)

Architecture choices are parameterized by continuous weights `α`, relaxed via softmax. We use bi-level optimization:
- Outer loop: update architecture parameters `α` on validation loss
- Inner loop: update network weights `w` on training loss

<!-- TODO: Report search hyperparameters (architecture LR, weight LR, epochs) -->

#### 3.5.2 Evolutionary NAS

A population of architectures evolves through tournament selection, mutation, and crossover. Fitness is evaluated as a weighted combination of accuracy and efficiency (FLOPs or latency).

<!-- TODO: Report population size, generations, mutation rate, crossover strategy -->

#### 3.5.3 Once-for-All Style

A single supernet is trained with progressive shrinking. After training, sub-networks are extracted and evaluated. An accuracy predictor selects Pareto-optimal architectures without additional training.

<!-- TODO: Report progressive shrinking schedule and predictor architecture -->

#### 3.5.4 Strategy Comparison

| Aspect | DARTS | Evolutionary | OFA-Style |
|--------|-------|--------------|-----------|
| Search time | Hours | Days | Single training run |
| Memory cost | High | Low | Medium |
| Flexibility | Medium | High | Medium |
| Big/little coupling | Good | Excellent | Good |

---

## 4. Experimental Setup

### 4.1 Datasets

#### Primary

**CIFAR-10.** 60,000 32x32 RGB images across 10 classes. 50k training, 10k test. Used for rapid prototyping and baseline validation.

**CIFAR-100.** 60,000 32x32 RGB images across 100 fine-grained classes. Tests model capacity and routing on a harder task.

#### Extended

**ImageNet-100.** 100-class subset of ImageNet at 224x224 resolution. Provides a realistic, higher-resolution evaluation.

**Visual Wake Words.** Binary classification (person / no person) derived from COCO. Directly relevant to edge deployment.

<!-- TODO: Report exact ImageNet-100 class selection and VWW split details -->

### 4.2 Data Preprocessing

- Standard augmentation: RandomCrop (with padding), RandomHorizontalFlip, ColorJitter
- Advanced augmentation: RandAugment, CutOut, MixUp / CutMix
- Normalization: per-dataset mean and standard deviation
- Multi-resolution: {32, 64, 128, 224} for resolution-adaptive experiments

### 4.3 Baselines

| Category | Model | Description |
|----------|-------|-------------|
| Static | MobileNetV3-Small | Lightweight baseline (~2.5M params) |
| Static | MobileNetV3-Large | Higher capacity baseline (~5.5M params) |
| Static | EfficientNet-B0 | Compound-scaled baseline (~5.3M params) |
| Cascaded | MobileNetV3-Small → Large | Independently trained, fixed threshold |
| Early-exit | MSDNet | Multi-scale dense network with anytime prediction |
| Early-exit | BranchyNet-style | Intermediate classifiers on backbone |
| NAS | Once-for-All subnets | Extracted sub-networks from OFA supernet |
| NAS | FBNet | Hardware-aware differentiable NAS |

### 4.4 Edge Hardware

| Device | Processor | RAM | Accelerator | Power Budget |
|--------|-----------|-----|-------------|--------------|
| NVIDIA Jetson Nano | ARM Cortex-A57 (quad) | 4 GB | 128 CUDA cores (Maxwell) | 5-10W |
| Raspberry Pi 4 | ARM Cortex-A72 (quad) | 4 GB | None (CPU-only) | 3-5W |
| Coral Dev Board | ARM Cortex-A53 (quad) | 1 GB | Edge TPU | 2-3W |
| Android mobile | Snapdragon 8xx | 6-8 GB | GPU + DSP | Variable |

### 4.5 Evaluation Metrics

**Accuracy:** Top-1 and Top-5 accuracy (%), reported as mean ± std over 3 runs with different seeds.

**Efficiency:** FLOPs (M), parameter count (M), model size on disk (MB).

**Edge performance:** Latency in ms (median over 1000 inferences after 100 warmup), throughput (images/sec), peak memory (MB), energy per inference (mJ, measured with hardware power meter).

**Dynamic behavior:** Exit rate (% of samples handled by little model), routing accuracy, confidence calibration (ECE).

**Search cost:** GPU-hours on specified hardware.

### 4.6 Implementation Details

- Framework: PyTorch 2.0+
- Pretrained models: timm library
- Experiment tracking: Weights & Biases / TensorBoard
- Deployment: ONNX export, TensorRT optimization, INT8/FP16 quantization

<!-- TODO: Report training hyperparameters:
- Optimizer, learning rate, schedule
- Batch size, epochs
- Weight decay, dropout
- GPU hardware used for training
- Random seeds
-->

---

## 5. Results

### 5.1 Main Results

<!-- TODO: Fill in after experiments -->

| Model | Top-1 (%) | Top-5 (%) | FLOPs (M) | Params (M) | Latency (ms) | Memory (MB) | Energy (mJ) |
|-------|-----------|-----------|-----------|------------|--------------|-------------|-------------|
| Ours (little) | | | | | | | |
| Ours (big) | | | | | | | |
| Ours (dynamic, τ=0.5) | | | | | | | |
| Ours (dynamic, τ=0.7) | | | | | | | |
| Ours (dynamic, τ=0.9) | | | | | | | |
| MobileNetV3-Small | | | | | | | |
| MobileNetV3-Large | | | | | | | |
| EfficientNet-B0 | | | | | | | |
| Cascaded (fixed) | | | | | | | |
| MSDNet | | | | | | | |

### 5.2 Accuracy-Efficiency Pareto Frontier

<!-- TODO: Insert Pareto plots: Accuracy vs Latency, Accuracy vs FLOPs, Accuracy vs Energy -->

### 5.3 NAS Strategy Comparison

<!-- TODO: Fill in after search experiments -->

| Strategy | Best Top-1 (%) | Search Cost (GPU-hrs) | Architectures Evaluated | Notes |
|----------|----------------|----------------------|------------------------|-------|
| DARTS | | | | |
| Evolutionary | | | | |
| OFA-style | | | | |

### 5.4 Ablation Studies

#### 5.4.1 Routing Strategy

<!-- TODO: Compare confidence-based, learned, difficulty-aware routing -->

| Router | Top-1 (%) | Avg Latency (ms) | Exit Rate (%) | Routing Accuracy (%) |
|--------|-----------|-------------------|---------------|---------------------|
| Confidence (τ=0.7) | | | | |
| Learned MLP | | | | |
| Difficulty-aware | | | | |

#### 5.4.2 Co-training vs. Independent Training

<!-- TODO: Quantify the benefit of joint optimization -->

| Training | Top-1 Little (%) | Top-1 Big (%) | Top-1 Dynamic (%) |
|----------|-------------------|---------------|-------------------|
| Independent | | | |
| Co-trained | | | |
| Co-trained + distillation | | | |

#### 5.4.3 Search Space Dimensions

<!-- TODO: Ablate width-only, depth-only, kernel-only, full -->

#### 5.4.4 Loss Function Components

<!-- TODO: Ablate each loss term -->

| Configuration | Top-1 (%) | Exit Rate (%) |
|---------------|-----------|---------------|
| Full loss | | |
| Without L_routing | | |
| Without L_consistency | | |
| Without distillation | | |

### 5.5 Edge Deployment Results

<!-- TODO: Fill in per-device benchmarks -->

#### Jetson Nano

| Model | Top-1 (%) | Latency (ms) | Throughput (img/s) | Memory (MB) | Energy (mJ) |
|-------|-----------|--------------|-------------------|-------------|-------------|
| Ours (dynamic) | | | | | |
| MobileNetV3-Small | | | | | |
| MobileNetV3-Large | | | | | |
| Cascaded | | | | | |

#### Raspberry Pi 4

<!-- TODO: Same table format as Jetson -->

#### Coral Dev Board

<!-- TODO: Same table format as Jetson -->

### 5.6 Dynamic Behavior Analysis

<!-- TODO: Include:
- Exit distribution histogram (easy/medium/hard samples)
- Confidence calibration plot (reliability diagram)
- Threshold sensitivity curve (accuracy vs exit rate as τ varies)
- Example images: correctly/incorrectly routed samples
-->

---

## 6. Discussion

### 6.1 Key Findings

<!-- TODO: Summarize 3-5 main findings from results -->

### 6.2 Co-training Benefits

<!-- TODO: Discuss why/when co-training outperforms independent training -->

### 6.3 NAS Strategy Trade-offs

<!-- TODO: Discuss when each NAS strategy is preferred -->

### 6.4 Routing Analysis

<!-- TODO: Discuss routing behavior, failure modes, calibration -->

### 6.5 Limitations

<!-- TODO: Honest assessment of:
- Datasets evaluated (only vision)
- Hardware evaluated (limited set)
- Search cost
- Generalization to other tasks (detection, segmentation)
-->

### 6.6 Future Work

<!-- TODO: Natural extensions:
- Apply to detection/segmentation tasks
- Multi-exit (>2 models) dynamic inference
- On-device NAS / continual adaptation
- More hardware platforms
-->

---

## 7. Conclusion

<!-- TODO: Write after all results are finalized. ~0.5-1 page covering:
- Restate the problem and approach
- Summarize key results with numbers
- State the significance/impact
-->

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

### Dynamic Inference and Early Exit

11. Teerapittayanon, S., McDanel, B., & Kung, H. T. (2017). BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. *arXiv:1709.01686*.
12. Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., & Weinberger, K. Q. (2018). Multi-Scale Dense Networks for Resource Efficient Image Classification. *ICLR 2018*.
13. Han, Y., Huang, G., Song, S., Yang, L., Wang, H., & Wang, Y. (2021). Dynamic Neural Networks: A Survey. *IEEE TPAMI, 44*(11), 7436--7456.
14. Laskaridis, S., Kouris, A., & Lane, N. D. (2021). Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions. *EMDL Workshop 2021*.
15. Zou, M., Li, X., Fang, J., Wen, H., & Fang, W. (2023). Dynamic Deep Neural Network Inference via Adaptive Channel Skipping. *Turkish J. of EE & CS, 31*(5).
16. Bajpai, D. J., & Hanawal, M. K. (2025). A Survey of Early Exit Deep Neural Networks in NLP. *arXiv:2501.07670*.
17. Yang, L., Han, Y., Chen, X., Song, S., Dai, J., & Huang, G. (2020). Resolution Adaptive Networks for Efficient Inference. *CVPR 2020*.

### Big/Little Model Collaboration

18. Kim, S., Mangalam, K., Moon, S., Malik, J., Mahoney, M. W., Gholami, A., & Keutzer, K. (2023). Speculative Decoding with Big Little Decoder. *NeurIPS 2023*.
19. Yang, B., He, L., Ling, N., Yan, Z., Xing, G., et al. (2023). EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge. *SenSys 2023*.
20. Chen, Y., Zhao, J., & Han, H. (2025). A Survey on Collaborative Mechanisms Between Large and Small Language Models. *arXiv:2505.07460*.
21. Wang, F., Chen, J., Yang, S., et al. (2025). A Survey on Collaborating Small and Large Language Models. *arXiv:2510.13890*.

### Hardware-Aware NAS and Edge Deployment

22. Benmeziane, H., El Maghraoui, K., Ouarnoughi, H., Niar, S., Wistuba, M., & Wang, N. (2021). A Comprehensive Survey on Hardware-Aware Neural Architecture Search. *arXiv:2101.09336*.
23. Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for MobileNetV3. *ICCV 2019*.
24. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
25. Wu, B., Dai, X., Zhang, P., Wang, Y., Sun, F., Wu, Y., ... & Keutzer, K. (2019). FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search. *CVPR 2019*.
26. Cai, H., Zhu, L., & Han, S. (2019). ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. *ICLR 2019*.
27. Tu, X., Chen, D., Altintas, O., Han, K., & Wang, H. (2025). PlatformX: An End-to-End Transferable Platform for Energy-Efficient Neural Architecture Search. *SEC 2025*.
28. Mao, S., Qin, M., Dong, W., Liu, H., & Gao, Y. (2025). RAM-NAS: Resource-aware Multiobjective Neural Architecture Search. *IROS 2024*.
29. Lin, Y., Yang, M., & Han, S. (2021). NAAS: Neural Accelerator Architecture Search. *DAC 2021*.
30. Krestinskaya, O., Fouda, M. E., Eltawil, A., & Salama, K. N. (2025). CIMNAS: A Joint Framework for Compute-In-Memory-Aware Neural Architecture Search. *arXiv:2509.25862*.
31. Bachiri, I., Benmeziane, H., Ouarnoughi, H., Baghdadi, R., Niar, S., & Aries, A. (2024). Combining Neural Architecture Search and Automatic Code Optimization: A Survey. *arXiv:2408.04116*.
32. Wang, T., Wang, K., Cai, H., Lin, J., Liu, Z., Wang, H., Lin, Y., & Han, S. (2020). APQ: Joint Search for Network Architecture, Pruning and Quantization Policy. *CVPR 2020*.
33. Al Youssef, H., Awada, S., Raad, M., Valle, M., & Ibrahim, A. (2025). Combining NAS and Weight Reshaping for Optimized Embedded Classifiers in Multisensory Glove. *Sensors, 25*(20), 6142.
34. Gupta, D., Lee, R. D., & Wynter, L. (2024). On Efficient Object-Detection NAS for ADAS on Edge Devices. *IEEE CAI 2024*.
35. Edge Intelligence: A Review of Deep Neural Network Inference for Edge Devices. (2025). *Electronics, 14*(12), 2495.

### NAS for Dynamic Inference

36. Gambella, M., & Roveri, M. (2023). EDANAS: Adaptive Neural Architecture Search for Early Exit Neural Networks. *IJCNN 2023*.
37. Gambella, M., Pomponi, J., Scardapane, S., & Roveri, M. (2025). NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks. *arXiv:2401.13330*.
38. Sponner, M., Servadei, L., Waschneck, B., Wille, R., & Kumar, A. (2024). Efficient Post-Training Augmentation for Adaptive Inference in Heterogeneous and Distributed IoT Environments. *arXiv:2403.07957*.
39. Casarin, F. (2025). NAS Just Once: Neural Architecture Search for Joint Image-Video Recognition. *ICCVW 2025*.

---

## Appendix

### A. Hyperparameter Settings

<!-- TODO: Full table of all hyperparameters for reproducibility -->

### B. Architecture Visualizations

<!-- TODO: DAG diagrams of discovered architectures -->

### C. Additional Results

<!-- TODO: Extended tables, per-class accuracy, additional ablations -->

### D. Deployment Configuration

<!-- TODO: ONNX/TensorRT settings, quantization details, device-specific optimizations -->
