# Progress Log

**Project:** Neural Architecture Search for Dynamic Inference
**Course:** Engineering Research, SDU
**Team:** Oliver Larsen, Oliver Svendsen, Aleksander Korsholm, Kristoffer Petersen
**Supervisor:** Francesco Daghero

---

## Week 1 (2026-02-06)

### Done
- Initialized project repository
- Added project description PDF
- Created research plan (`RESEARCH_PLAN.md`)
- Set up README with project overview, objectives, methodology, and team roles
- Created progress log (`PROGRESS.md`)
- Created `CLAUDE.md` with project conventions
- Created working document (`working_document.md`) -- research paper skeleton with all chapters
- **Literature review completed**: Analyzed all 33 initial papers in `research-papers/` folder
- **Related Works chapter written** in `working_document.md` (Section 2), covering:
  - NAS methods (DARTS, evolutionary, supernet/OFA, sample-efficient)
  - Dynamic inference (early exit, channel skipping, big/little cascading)
  - Hardware-aware NAS and edge deployment
  - NAS for dynamic inference (EDANAS, NACHOS, post-training augmentation)
  - Gap analysis with comparison table across all key works
- **References expanded** from 10 to 39 citations organized by topic
- **Second batch of 13 papers analyzed** and integrated into Related Works:
  - Knowledge Distillation (Hinton et al.), Slimmable Networks, US-Nets (sandwich rule, inplace distillation)
  - MSDNet, BlockDrop, SkipNet, RANet (dynamic inference foundations with detailed results)
  - DrNAS (Dirichlet NAS), NASWOT (zero-cost proxy), RegNet (design space design)
  - MobileNetV3, MCUNet (edge deployment), FrugalGPT (LLM cascading)
- **Related Works refined**: Expanded all sections with concrete results/numbers from newly read papers
- **Gap analysis updated**: Added Slimmable/US-Nets, BlockDrop/SkipNet, MCUNet entries; added 5th gap (width-variable training without NAS optimization)
- **References expanded** from 39 to 49 citations

### Decisions
- *None yet*

### Open Questions
- Which NAS strategy to prioritize first: OFA-style supernet (most practical, closest to EDANAS/NACHOS baseline) vs. evolutionary (more flexible for multi-objective optimization)?
- Should we start with CIFAR-10 for rapid iteration, or directly target a more realistic dataset?
- How to handle the router design: fixed confidence threshold vs. learned routing network?

---

<!-- Template for new weeks:

## Week N (YYYY-MM-DD)

### Done
-

### Decisions
-

### Open Questions
-

---

-->
