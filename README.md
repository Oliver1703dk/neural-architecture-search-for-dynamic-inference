# Neural Architecture Search for Dynamic Inference

**Course:** Engineering Research, SDU
**Supervisor:** Francesco Daghero (fdag@mmmi.sdu.dk, DECO Lab)
**Team:** Oliver Larsen, Oliver Svendsen, Aleksander Korsholm, Kristoffer Petersen

## About

This project explores Neural Architecture Search (NAS) for big/little dynamic inference networks. Current dynamic inference solutions typically cascade two separate models or use early-exit strategies, limiting adaptability since the models are trained independently.

We propose a framework where big and little models are co-trained in a NAS-supernet-like manner, allowing their dimensions and architecture to evolve dynamically during training. The goal is to discover dynamic models optimized for edge deployment that reduce inference cost while maintaining accuracy.

## Research Question

Can NAS-driven co-training of coupled big/little models outperform static cascaded baselines in accuracy-efficiency trade-offs on edge hardware?

## Objectives

1. **Data and Dataset Preparation** -- Benchmark on standard vision datasets (CIFAR-10, ImageNet, edge-relevant datasets) and preprocess data for dynamic inference evaluation.
2. **NAS Exploration** -- Investigate and implement NAS strategies including supernet training, evolutionary NAS, and differentiable NAS for discovering coupled big/little models.
3. **Dynamic Model Training** -- Co-train big and little components in a unified framework, exploring dimension adaptation (width, depth, block-level scaling) during training.
4. **Edge Deployment** -- Deploy discovered architectures on edge hardware (Jetson Nano, Raspberry Pi, mobile devices) and evaluate latency, memory, and energy consumption.
5. **Evaluation** -- Benchmark accuracy, inference cost, and dynamic behavior against standard cascaded and early-exit baselines.

## Methodology

- **Dataset Preparation:** Select vision datasets for edge deployment, apply preprocessing and augmentation, partition for NAS validation and testing.
- **NAS Framework:** Build a supernet representing the search space of coupled big/little models. Implement differentiable and evolutionary NAS techniques with data-aware routing components.
- **Dynamic Co-training:** Train big and little models jointly, allowing parameters and dimensions to evolve during training.
- **Edge Deployment & Benchmarking:** Deploy on edge hardware, measure latency/memory/energy, compare against static and cascaded baselines.
- **Analysis & Documentation:** Analyze accuracy-efficiency-adaptability trade-offs and document methodology and results.

## Expected Outcomes

1. A framework for NAS-driven dynamic big/little models with joint training and input-adaptive scaling.
2. Benchmarked models for vision tasks showing accuracy vs. inference cost trade-offs.
3. Edge-deployable models with measured latency, energy, and memory efficiency.
4. Evaluation of different NAS strategies in the context of dynamic models.

## Team Roles

| Role | Responsibility |
|------|----------------|
| Data Engineer | Prepares datasets, handles preprocessing and augmentation |
| Machine Learning Engineer | Designs NAS framework, co-training strategy, and dynamic inference policies |
| Edge Software Developer | Implements model deployment on edge devices, measures latency and energy metrics |
| Evaluation Lead | Designs benchmarking protocols, evaluates accuracy vs. efficiency trade-offs, and documents results |

## Project Timeline

| Step | Description |
|------|-------------|
| 1 | Literature review and selection of benchmark datasets |
| 2 | Implement NAS frameworks and define the big/little supernet search space |
| 3 | Train and evaluate dynamic coupled architectures on vision datasets |
| 4 | Deploy models on edge devices, benchmark efficiency, and compare baselines |
| 5 | Documentation and final presentation of results |

## Repository Structure

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

## Open Source

All software developed as part of this project adheres to open-source principles with maintainable, well-structured, and publicly accessible code. The project encourages contributions to existing open-source frameworks and aims to provide reusable components for future research.

## Contact

Francesco Daghero -- fdag@mmmi.sdu.dk -- DECO Lab
