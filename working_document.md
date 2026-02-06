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

### 2.1 Neural Architecture Search

Neural Architecture Search automates the design of network architectures. Early methods used reinforcement learning (Zoph & Le, 2017) and evolutionary algorithms (Real et al., 2019), but required thousands of GPU-days. Subsequent work focused on reducing search cost.

**DARTS** (Liu et al., 2019) introduced differentiable NAS by relaxing the discrete search space into a continuous one, enabling gradient-based optimization. This reduced search cost to 1.5 GPU-days on CIFAR-10 while achieving 2.76% test error.

**Once-for-All** (Cai et al., 2020) proposed training a single supernet that supports many sub-networks through progressive shrinking. After a single training run, sub-networks can be extracted without retraining, enabling efficient deployment across diverse hardware.

**BigNAS** (Yu et al., 2020) scaled single-stage NAS by training large supernets with sandwich sampling and in-place distillation, directly producing deployable models without post-search retraining.

**SPOS** (Guo et al., 2020) used uniform path sampling to decouple supernet training from architecture search, improving search efficiency.

<!-- TODO: Add 3-5 more references as literature review progresses -->

### 2.2 Dynamic Inference

Dynamic inference adapts computation to input difficulty at test time.

**Early-exit networks** attach intermediate classifiers to a backbone. BranchyNet (Teerapittayanon et al., 2016) pioneered confidence-based early exiting. MSDNet (Huang et al., 2018) introduced multi-scale dense connectivity for anytime prediction, achieving strong accuracy-efficiency trade-offs.

**Cascaded models** route inputs through a sequence of models. Big-little networks pair a small model with a large model, using the small model's confidence to decide if the large model is needed. This approach is simple but limited by independent training.

**Input-adaptive methods** such as SkipNet and BlockDrop learn to selectively execute layers or blocks conditioned on the input, reducing average computation while maintaining accuracy.

<!-- TODO: Expand with additional references on adaptive computation -->

### 2.3 Hardware-Aware and Edge ML

Efficient architectures such as MobileNet (Howard et al., 2019) and EfficientNet (Tan & Le, 2019) are designed for resource-constrained deployment. Hardware-aware NAS methods incorporate latency or energy predictors into the search objective.

FBNet (Wu et al., 2019) performs differentiable NAS with hardware-aware loss functions. ProxylessNAS (Cai et al., 2019) directly searches on the target task and hardware. These methods demonstrate that NAS can produce architectures that outperform manually designed efficient networks on specific hardware.

<!-- TODO: Add edge deployment and quantization references -->

### 2.4 Gap in Existing Work

No prior work applies NAS to jointly discover and co-train coupled big/little dynamic inference models. Existing NAS methods optimize a single architecture. Existing dynamic inference methods use fixed or independently trained components. Our work bridges this gap by embedding the big/little coupling directly into the NAS search space.

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

1. Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable Architecture Search. *ICLR 2019*.
2. Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020). Once-for-All: Train One Network and Specialize it for Efficient Deployment. *ICLR 2020*.
3. Yu, J., Jin, P., Liu, H., Bender, G., Kindermans, P.-J., Tan, M., ... & Adam, H. (2020). BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models. *ECCV 2020*.
4. Guo, Z., Zhang, X., Mu, H., Heng, W., Liu, Z., Wei, Y., & Sun, J. (2020). Single Path One-Shot Neural Architecture Search with Uniform Sampling. *ECCV 2020*.
5. Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. *ICPR 2016*.
6. Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., & Weinberger, K. Q. (2018). Multi-Scale Dense Networks for Resource Efficient Image Classification. *ICLR 2018*.
7. Yang, L., Han, Y., Chen, X., Song, S., Dai, J., & Huang, G. (2020). Resolution Adaptive Networks for Efficient Inference. *CVPR 2020*.
8. Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for MobileNetV3. *ICCV 2019*.
9. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
10. Wu, B., Dai, X., Zhang, P., Wang, Y., Sun, F., Wu, Y., ... & Keutzer, K. (2019). FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search. *CVPR 2019*.

<!-- TODO: Add references as literature review expands (target 20+) -->

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
