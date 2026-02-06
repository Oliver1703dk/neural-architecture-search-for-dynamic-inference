# Research Papers for Neural Architecture Search for Dynamic Inference

A curated collection of relevant research papers organized by topic.

---

## 1. Neural Architecture Search (NAS) - Foundational

### DARTS: Differentiable Architecture Search
- **Authors:** Hanxiao Liu, Karen Simonyan, Yiming Yang
- **Venue:** ICLR 2019
- **arXiv:** https://arxiv.org/abs/1806.09055
- **Code:** https://github.com/quark0/darts
- **Abstract:** This paper addresses the scalability challenge of architecture search by formulating the task in a differentiable manner. Unlike conventional approaches of applying evolution or reinforcement learning over a discrete and non-differentiable search space, our method is based on the continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent.
- **Relevance:** ⭐⭐⭐⭐⭐ Core differentiable NAS method - primary technique to implement

---

### Once-for-All: Train One Network and Specialize it for Efficient Deployment
- **Authors:** Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
- **Venue:** ICLR 2020
- **arXiv:** https://arxiv.org/abs/1908.09791
- **Code:** https://github.com/mit-han-lab/once-for-all
- **Abstract:** We propose to train a once-for-all (OFA) network that supports diverse architectural settings by decoupling training and search. We can quickly get a specialized sub-network by selecting from the OFA network without additional training. To efficiently train OFA networks, we also propose a novel progressive shrinking algorithm, a generalized pruning method that reduces the model size across many more dimensions than pruning (depth, width, kernel size, and resolution). It can obtain a surprisingly large number of sub-networks (>10^19) that can fit different hardware platforms and latency constraints.
- **Relevance:** ⭐⭐⭐⭐⭐ Core supernet training method - directly applicable to big/little co-training

---

### BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models
- **Authors:** Jiahui Yu, Pengchong Jin, Hanxiao Liu, Gabriel Bender, Pieter-Jan Kindermans, Mingxing Tan, Thomas Huang, Xiaodan Song, Quoc Le
- **Venue:** ECCV 2020
- **arXiv:** https://arxiv.org/abs/2003.11142
- **Abstract:** We propose BigNAS, an approach that challenges the conventional wisdom that post-processing of the weights is necessary to get good prediction accuracies. Without extra retraining or post-processing steps, we are able to train a single set of shared weights on ImageNet and use these weights to obtain child models whose sizes range from 200 to 1000 MFLOPs.
- **Relevance:** ⭐⭐⭐⭐⭐ Single-stage supernet training - key technique for efficient NAS

---

### ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
- **Authors:** Han Cai, Ligeng Zhu, Song Han
- **Venue:** ICLR 2019
- **arXiv:** https://arxiv.org/abs/1812.00332
- **Abstract:** We present ProxylessNAS that can directly learn the architectures for large-scale target tasks and target hardware platforms. We address the high memory consumption issue of differentiable NAS and reduce the computational cost to the same level of regular training while still allowing a large candidate set.
- **Relevance:** ⭐⭐⭐⭐⭐ Hardware-aware NAS - essential for edge deployment optimization

---

### FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
- **Authors:** Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, Kurt Keutzer
- **Venue:** CVPR 2019
- **arXiv:** https://arxiv.org/abs/1812.03443
- **Abstract:** We propose a differentiable neural architecture search (DNAS) framework that uses gradient-based methods to optimize ConvNet architectures. FBNets surpass state-of-the-art models both designed manually and generated automatically. FBNet-B achieves 74.1% top-1 accuracy on ImageNet with 295M FLOPs and 23.1 ms latency on a Samsung S8 phone.
- **Relevance:** ⭐⭐⭐⭐ Hardware-aware differentiable NAS with latency optimization

---

## 2. Dynamic Inference & Early Exit Networks

### BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks
- **Authors:** Surat Teerapittayanon, Bradley McDanel, H.T. Kung
- **Venue:** ICPR 2016
- **arXiv:** https://arxiv.org/abs/1709.01686
- **Abstract:** We present BranchyNet, a novel deep network architecture that is augmented with additional side branch classifiers. The architecture allows prediction results for a large portion of test samples to exit the network early via these branches when samples can already be inferred with high confidence. BranchyNet exploits the observation that features learned at an early layer of a network may often be sufficient for the classification of many data points.
- **Relevance:** ⭐⭐⭐⭐⭐ Foundational early-exit work - key baseline and design inspiration

---

### Multi-Scale Dense Networks for Resource Efficient Image Classification (MSDNet)
- **Authors:** Gao Huang, Danlu Chen, Tianhong Li, Felix Wu, Laurens van der Maaten, Kilian Weinberger
- **Venue:** ICLR 2018
- **arXiv:** https://arxiv.org/abs/1703.09844
- **Abstract:** We investigate image classification with computational resource limits at test time for anytime classification and budgeted batch classification. We train multiple classifiers with varying resource demands, which we adaptively apply during test time. To maximally re-use computation between the classifiers, we incorporate them as early-exits into a single deep convolutional neural network and inter-connect them with dense connectivity.
- **Relevance:** ⭐⭐⭐⭐⭐ State-of-the-art early-exit architecture - primary baseline

---

### SkipNet: Learning Dynamic Routing in Convolutional Networks
- **Authors:** Xin Wang, Fisher Yu, Zi-Yi Dou, Trevor Darrell, Joseph E. Gonzalez
- **Venue:** ECCV 2018
- **arXiv:** https://arxiv.org/abs/1711.09485
- **Code:** https://github.com/ucbdrive/skipnet
- **Abstract:** We introduce SkipNet, a modified residual network, that uses a gating network to selectively skip convolutional blocks based on the activations of the previous layer. We formulate the dynamic skipping problem in the context of sequential decision making and propose a hybrid learning algorithm that combines supervised learning and reinforcement learning.
- **Relevance:** ⭐⭐⭐⭐⭐ Input-adaptive layer skipping - key technique for dynamic routing

---

### BlockDrop: Dynamic Inference Paths in Residual Networks
- **Authors:** Zuxuan Wu, Tushar Nagarajan, Abhishek Kumar, Steven Rennie, Larry S. Davis, Kristen Grauman, Rogerio Feris
- **Venue:** CVPR 2018
- **arXiv:** https://arxiv.org/abs/1711.08393
- **Abstract:** We introduce BlockDrop, an approach that learns to dynamically choose which layers of a deep network to execute during inference so as to best reduce total computation without degrading prediction accuracy. Given a pretrained ResNet, we train a policy network in an associative reinforcement learning setting for the dual reward of utilizing a minimal number of blocks while preserving recognition accuracy.
- **Relevance:** ⭐⭐⭐⭐ RL-based dynamic block selection - alternative routing approach

---

### Hardware-aware Neural Architecture Search of Early Exiting Networks on Edge Accelerators
- **Authors:** Alaa Zniber, Arne Symons, Ouassim Karrakchou, Marian Verhelst, Mounir Ghogho
- **Venue:** arXiv 2025
- **arXiv:** https://arxiv.org/search/?searchtype=author&query=Zniber%2C+A (December 2025)
- **Abstract:** Addresses the growing demand for embedded intelligence at the edge with stringent computational and energy constraints through NAS for early-exit networks.
- **Relevance:** ⭐⭐⭐⭐⭐ Very recent work directly combining NAS + early-exit + edge deployment

---

### Confidence-gated training for efficient early-exit neural networks
- **Authors:** Saad Mokssit, Ouassim Karrakchou, Alejandro Mousist, Mounir Ghogho
- **Venue:** arXiv 2025
- **arXiv:** (September 2025, updated January 2026)
- **Abstract:** Training method for early-exit networks using confidence-based gating.
- **Relevance:** ⭐⭐⭐⭐ Recent training strategy for early-exit networks

---

## 3. Efficient Architectures for Edge Deployment

### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
- **Authors:** Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
- **Venue:** arXiv 2017
- **arXiv:** https://arxiv.org/abs/1704.04861
- **Abstract:** We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks.
- **Relevance:** ⭐⭐⭐⭐ Foundational efficient architecture - baseline for "little" model

---

### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- **Authors:** Mingxing Tan, Quoc V. Le
- **Venue:** ICML 2019
- **arXiv:** https://arxiv.org/abs/1905.11946
- **Code:** https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- **Abstract:** We systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient.
- **Relevance:** ⭐⭐⭐⭐ Compound scaling - informs search space design for width/depth/resolution

---

### Designing Network Design Spaces (RegNet)
- **Authors:** Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár
- **Venue:** CVPR 2020
- **arXiv:** https://arxiv.org/abs/2003.13678
- **Abstract:** We present a new network design paradigm. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function.
- **Relevance:** ⭐⭐⭐⭐ Design space methodology - informs search space construction

---

### CondConv: Conditionally Parameterized Convolutions for Efficient Inference
- **Authors:** Brandon Yang, Gabriel Bender, Quoc V. Le, Jiquan Ngiam
- **Venue:** NeurIPS 2019
- **arXiv:** https://arxiv.org/abs/1904.04971
- **Code:** https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv
- **Abstract:** We propose conditionally parameterized convolutions (CondConv), which learn specialized convolutional kernels for each example. Replacing normal convolutions with CondConv enables us to increase the size and capacity of a network, while maintaining efficient inference.
- **Relevance:** ⭐⭐⭐⭐ Input-adaptive convolutions - potential component for dynamic models

---

## 4. Recent Work (2025-2026)

### SQUAD: Scalable Quorum Adaptive Decisions via ensemble of early exit neural networks
- **Authors:** Matteo Gambella, Fabrizio Pittorino, Giuliano Casale, Manuel Roveri
- **Venue:** arXiv January 2026
- **Abstract:** Ensemble approach to early-exit networks for scalable adaptive decisions.
- **Relevance:** ⭐⭐⭐ Recent ensemble early-exit approach

---

### Bridging Efficiency and Safety: Formal Verification of Neural Networks with Early Exits
- **Authors:** Yizhak Yisrael Elboher, Avraham Raviv, et al.
- **Venue:** arXiv December 2025
- **Abstract:** Formal verification provides guarantees of neural network safety and efficiency for early-exit networks.
- **Relevance:** ⭐⭐⭐ Verification methods - relevant for safety-critical edge deployment

---

### NeuCODEX: Edge-Cloud Co-Inference with Spike-Driven Compression and Dynamic Early-Exit
- **Authors:** Maurf Hassan, Steven Davy, et al.
- **Venue:** arXiv September 2025
- **Abstract:** Spiking neural networks with dynamic early-exit for edge-cloud co-inference.
- **Relevance:** ⭐⭐⭐ Edge-cloud collaboration with dynamic inference

---

## 5. Surveys & Overview Papers

### Neural Architecture Search: A Survey
- **Authors:** Thomas Elsken, Jan Hendrik Metzen, Frank Hutter
- **Venue:** JMLR 2019
- **arXiv:** https://arxiv.org/abs/1808.05377
- **Abstract:** Comprehensive survey of NAS methods covering search space, search strategy, and performance estimation.
- **Relevance:** ⭐⭐⭐⭐ Background reading for NAS fundamentals

---

### Dynamic Neural Networks: A Survey
- **Authors:** Yizeng Han, Gao Huang, et al.
- **Venue:** IEEE TPAMI 2021
- **arXiv:** https://arxiv.org/abs/2102.04906
- **Abstract:** Survey covering dynamic depth, width, routing, and other adaptive inference methods.
- **Relevance:** ⭐⭐⭐⭐⭐ Essential survey for dynamic inference techniques

---

## Reading Priority

### Week 1-2 (Foundations)
1. DARTS (differentiable NAS basics)
2. Once-for-All (supernet training)
3. BranchyNet (early-exit fundamentals)
4. MSDNet (advanced early-exit)
5. Dynamic Neural Networks Survey

### Week 3-4 (Advanced NAS)
1. BigNAS (single-stage training)
2. ProxylessNAS (hardware-aware)
3. FBNet (differentiable hardware-aware)
4. SkipNet (dynamic routing)
5. BlockDrop (RL-based routing)

### Week 5-6 (Edge & Recent)
1. Hardware-aware NAS for Early Exiting (2025)
2. MobileNets
3. EfficientNet
4. RegNet (design spaces)
5. Confidence-gated training (2025)

---

## Code Repositories

| Paper | Repository |
|-------|------------|
| DARTS | https://github.com/quark0/darts |
| Once-for-All | https://github.com/mit-han-lab/once-for-all |
| SkipNet | https://github.com/ucbdrive/skipnet |
| EfficientNet | https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet |
| MSDNet | https://github.com/kalviny/MSDNet-PyTorch |
| timm (pretrained models) | https://github.com/huggingface/pytorch-image-models |

---

## BibTeX References

```bibtex
@inproceedings{liu2019darts,
  title={DARTS: Differentiable Architecture Search},
  author={Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  booktitle={ICLR},
  year={2019}
}

@inproceedings{cai2020once,
  title={Once-for-All: Train One Network and Specialize it for Efficient Deployment},
  author={Cai, Han and Gan, Chuang and Wang, Tianzhe and Zhang, Zhekai and Han, Song},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{yu2020bignas,
  title={BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models},
  author={Yu, Jiahui and Jin, Pengchong and Liu, Hanxiao and Bender, Gabriel and Kindermans, Pieter-Jan and Tan, Mingxing and Huang, Thomas and Song, Xiaodan and Le, Quoc},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{teerapittayanon2016branchynet,
  title={BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks},
  author={Teerapittayanon, Surat and McDanel, Bradley and Kung, HT},
  booktitle={ICPR},
  year={2016}
}

@inproceedings{huang2018msdnet,
  title={Multi-Scale Dense Networks for Resource Efficient Image Classification},
  author={Huang, Gao and Chen, Danlu and Li, Tianhong and Wu, Felix and van der Maaten, Laurens and Weinberger, Kilian},
  booktitle={ICLR},
  year={2018}
}

@inproceedings{wang2018skipnet,
  title={SkipNet: Learning Dynamic Routing in Convolutional Networks},
  author={Wang, Xin and Yu, Fisher and Dou, Zi-Yi and Darrell, Trevor and Gonzalez, Joseph E},
  booktitle={ECCV},
  year={2018}
}

@inproceedings{wu2018blockdrop,
  title={BlockDrop: Dynamic Inference Paths in Residual Networks},
  author={Wu, Zuxuan and Nagarajan, Tushar and Kumar, Abhishek and Rennie, Steven and Davis, Larry S and Grauman, Kristen and Feris, Rogerio},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={ICML},
  year={2019}
}

@article{howard2017mobilenets,
  title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal={arXiv preprint arXiv:1704.04861},
  year={2017}
}

@inproceedings{cai2019proxylessnas,
  title={ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware},
  author={Cai, Han and Zhu, Ligeng and Han, Song},
  booktitle={ICLR},
  year={2019}
}
```

---

*Last updated: 2026-02-06*
*Compiled by: Lars Claw 🦞*
