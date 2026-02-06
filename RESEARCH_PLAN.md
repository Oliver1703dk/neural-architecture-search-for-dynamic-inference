# Research Plan: Neural Architecture Search for Dynamic Inference

**Project Lead:** Oliver Larsen  
**Supervisor:** Francesco Daghero (fdag@mmmi.sdu.dk, DECO Lab)  
**Started:** February 2026

---

## 1. Project Overview

Develop a NAS framework for co-training big/little dynamic inference models, optimized for edge deployment. Unlike traditional cascaded or early-exit approaches, this explores joint architecture evolution where both models adapt dimensions and structure during training.

**Core Research Question:** Can NAS-driven co-training of coupled big/little models outperform static cascaded baselines in accuracy-efficiency trade-offs on edge hardware?

---

## 2. Timeline & Milestones

### Phase 1: Foundation (Weeks 1-3)
| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Literature review: NAS methods, dynamic inference, early-exit networks | Annotated bibliography (20+ papers) |
| 2 | Survey existing frameworks (Once-for-All, BigNAS, SPOS, DARTS) | Framework comparison matrix |
| 3 | Dataset selection & preprocessing pipeline | Data loader + augmentation code |

### Phase 2: NAS Framework Development (Weeks 4-8)
| Week | Task | Deliverable |
|------|------|-------------|
| 4-5 | Design supernet search space (big/little coupling) | Search space specification doc |
| 6 | Implement differentiable NAS baseline | Working DARTS-style training loop |
| 7 | Implement evolutionary NAS alternative | EA-based search implementation |
| 8 | Add input-adaptive routing mechanism | Router module + confidence gating |

### Phase 3: Co-Training & Experiments (Weeks 9-14)
| Week | Task | Deliverable |
|------|------|-------------|
| 9-10 | Joint training pipeline (width/depth scaling) | Co-training script |
| 11 | CIFAR-10 experiments + hyperparameter tuning | Baseline results table |
| 12-13 | ImageNet-subset or edge-relevant dataset experiments | Extended results |
| 14 | Ablation studies (routing strategies, scaling dims) | Ablation analysis |

### Phase 4: Edge Deployment (Weeks 15-18)
| Week | Task | Deliverable |
|------|------|-------------|
| 15 | Model export (ONNX/TensorRT) | Optimized model artifacts |
| 16 | Deploy on Jetson Nano | Latency/memory benchmarks |
| 17 | Deploy on Raspberry Pi / mobile | Cross-platform benchmarks |
| 18 | Energy measurement setup + profiling | Energy efficiency report |

### Phase 5: Analysis & Documentation (Weeks 19-22)
| Week | Task | Deliverable |
|------|------|-------------|
| 19-20 | Compare against baselines (cascaded, early-exit) | Comparison tables + plots |
| 21 | Write-up: methodology, results, insights | Draft paper/report |
| 22 | Final presentation preparation | Slides + demo |

---

## 3. Literature Review Focus

### Core Topics
1. **Neural Architecture Search**
   - DARTS (Liu et al., 2019)
   - Once-for-All (Cai et al., 2020)
   - BigNAS (Yu et al., 2020)
   - SPOS (Guo et al., 2020)

2. **Dynamic Inference**
   - Early-exit networks (BranchyNet, MSDNet)
   - Cascaded models (big-little networks)
   - Input-adaptive inference (SkipNet, BlockDrop)

3. **Edge ML**
   - MobileNet, EfficientNet families
   - Quantization-aware training
   - TensorRT/ONNX optimization

4. **Supernet Training**
   - Weight sharing strategies
   - Sandwich rule training
   - Progressive shrinking

### Key Papers to Start
- [ ] "DARTS: Differentiable Architecture Search" (Liu et al., ICLR 2019)
- [ ] "Once-for-All: Train One Network and Specialize it for Efficient Deployment" (Cai et al., ICLR 2020)
- [ ] "BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models" (Yu et al., ECCV 2020)
- [ ] "BranchyNet: Fast Inference via Early Exiting" (Teerapittayanon et al., ICPR 2016)
- [ ] "MSDNet: Multi-Scale Dense Networks for Resource Efficient Image Classification" (Huang et al., ICLR 2018)
- [ ] "Resolution Adaptive Networks for Efficient Inference" (Yang et al., CVPR 2020)

---

## 4. Technical Approach

### 4.1 Search Space Design

```
Supernet Structure:
├── Shared Stem (fixed)
├── Searchable Blocks (N blocks)
│   ├── Width: {0.25x, 0.5x, 0.75x, 1.0x}
│   ├── Depth: {1, 2, 3, 4} layers per block
│   ├── Kernel: {3x3, 5x5, 7x7}
│   └── Expansion: {1, 2, 4, 6}
├── Router Module (input-adaptive)
│   └── Confidence threshold → big/little selection
└── Dual Heads
    ├── Little classifier (early)
    └── Big classifier (final)
```

### 4.2 Co-Training Strategy

1. **Sandwich Training:** Sample {min, max, random} subnets per batch
2. **In-place Distillation:** Big subnet teaches little subnet
3. **Progressive Shrinking:** Start full, gradually enable smaller configs
4. **Joint Loss:**
   ```
   L = α·L_big + β·L_little + γ·L_routing + λ·L_consistency
   ```

### 4.3 Routing Mechanism

- **Confidence-based:** Little model exits if confidence > threshold
- **Learned router:** Small MLP predicts big/little based on features
- **Difficulty-aware:** Train router to recognize "hard" samples

### 4.4 NAS Strategies to Compare

| Strategy | Pros | Cons |
|----------|------|------|
| Differentiable (DARTS) | Fast, gradient-based | Memory intensive |
| Evolutionary | Flexible, no supernet bias | Slow, many evaluations |
| Once-for-All style | Single training, many subnets | Complex training |

---

## 5. Datasets

### Primary
- **CIFAR-10:** Quick iteration, baseline validation
- **CIFAR-100:** More classes, harder task

### Extended
- **ImageNet-100:** Subset for realistic evaluation
- **Visual Wake Words:** Edge-specific benchmark
- **COCO (detection):** If time permits, extend to detection

### Preprocessing
- Standard augmentation: RandomCrop, HorizontalFlip, ColorJitter
- AutoAugment / RandAugment for stronger baselines
- Resolution scaling: {32, 64, 128, 224} for multi-resolution experiments

---

## 6. Edge Hardware Targets

| Device | Specs | Use Case |
|--------|-------|----------|
| Jetson Nano | 128 CUDA cores, 4GB RAM | Primary edge target |
| Raspberry Pi 4 | ARM Cortex-A72, 4GB RAM | CPU-only baseline |
| Coral Dev Board | Edge TPU | Accelerator comparison |
| Mobile (Android) | Snapdragon 8xx | Real-world deployment |

### Metrics to Measure
- **Latency:** End-to-end inference time (ms)
- **Throughput:** Images/second
- **Memory:** Peak RAM usage
- **Energy:** Joules per inference (with power meter)
- **Accuracy:** Top-1, Top-5

---

## 7. Baselines for Comparison

1. **Static Models**
   - MobileNetV3-Small / Large
   - EfficientNet-B0

2. **Cascaded Models**
   - MobileNetV3-Small → MobileNetV3-Large
   - Fixed threshold routing

3. **Early-Exit**
   - MSDNet
   - BranchyNet-style exits

4. **Other NAS**
   - Once-for-All subnets
   - FBNet

---

## 8. Evaluation Protocol

### Accuracy-Efficiency Frontier
Plot Pareto curves: Accuracy vs. {Latency, FLOPs, Energy}

### Metrics Table
| Model | Top-1 Acc | FLOPs (M) | Latency (ms) | Memory (MB) | Energy (mJ) |
|-------|-----------|-----------|--------------|-------------|-------------|
| Ours (little) | ? | ? | ? | ? | ? |
| Ours (big) | ? | ? | ? | ? | ? |
| Ours (dynamic) | ? | ? | ? | ? | ? |
| MobileNetV3-S | baseline | baseline | baseline | baseline | baseline |

### Ablations
1. Routing strategy comparison
2. Co-training vs. independent training
3. Search space dimensions impact
4. NAS strategy comparison (DARTS vs. EA vs. OFA-style)

---

## 9. Repository Structure

```
neural-architecture-search-for-dynamic-inference/
├── README.md
├── RESEARCH_PLAN.md
├── project-description/
├── docs/
│   ├── literature/          # Paper notes & summaries
│   └── meeting-notes/       # Supervisor meetings
├── src/
│   ├── data/                # Data loaders, augmentation
│   ├── models/
│   │   ├── supernet/        # Supernet definition
│   │   ├── router/          # Routing modules
│   │   └── search_space/    # Search space configs
│   ├── nas/
│   │   ├── darts/           # Differentiable NAS
│   │   ├── evolutionary/    # EA-based search
│   │   └── ofa/             # Once-for-All style
│   ├── training/            # Training loops, co-training
│   ├── evaluation/          # Benchmarking scripts
│   └── deployment/          # ONNX export, TensorRT
├── configs/                 # Experiment configs (YAML)
├── scripts/                 # Training/eval shell scripts
├── notebooks/               # Exploration & visualization
├── experiments/             # Results, logs, checkpoints
└── edge/                    # Edge deployment code
    ├── jetson/
    ├── rpi/
    └── mobile/
```

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NAS training instability | High | Use proven baselines (OFA), extensive logging |
| Edge deployment issues | Medium | Start deployment early (week 15), have backup devices |
| ImageNet compute cost | Medium | Use ImageNet-100 subset, cloud credits if needed |
| Time overrun | High | Prioritize CIFAR-10 + Jetson, treat others as stretch |

---

## 11. Open Source Plan

- **License:** Apache 2.0 or MIT
- **Documentation:** Comprehensive README, docstrings, tutorials
- **Reproducibility:** Config files, random seeds, Docker/conda envs
- **Contributions:** Follow existing framework conventions where applicable

---

## 12. Next Steps (This Week)

1. [ ] Set up development environment (PyTorch, CUDA)
2. [ ] Clone reference repos: Once-for-All, DARTS, timm
3. [ ] Start literature review with 5 core papers
4. [ ] Write CIFAR-10 data loader with augmentation
5. [ ] Schedule kickoff meeting with Francesco

---

## 13. Resources

### Compute
- Local: MacBook (dev), GPU workstation if available
- Cloud: SDU compute cluster / Google Colab Pro
- Edge: Jetson Nano (acquire if not available)

### Frameworks
- PyTorch + torchvision
- timm (pretrained models)
- NNI or AutoML frameworks for NAS
- ONNX + TensorRT for deployment

---

*Last updated: 2026-02-06*
