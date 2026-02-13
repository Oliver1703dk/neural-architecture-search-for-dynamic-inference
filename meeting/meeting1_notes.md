# Meeting 1 - Kickoff Meeting

**Date:** Early semester (Week 1)
**Attendees:**
- **Supervisors:** Emil Jorgensen (Assistant Professor, NAS expert, based in Vejle), Francesco Daghero (Postdoc, DECO Lab SDU Odense, dynamic inference/TinyML expert)
- **Students:** Oliver Larsen, Oliver Svendsen, Aleksander Korsholm, Kristoffer Petersen

**Format:** Online video call

---

## Agenda & Questions Brought by Students

0. Can we get the slides shared?
1. What is our current state and what should we focus on?
2. Could this project lead to a paper submission?
3. What compute resources are available for training and evaluation?

---

## 1. Supervisor Introductions

### Emil Jorgensen
- Recently started as assistant professor at SDU (HCI area), based at the new Vejle campus.
- Research background: **Neural Architecture Search**, dataset properties for HCI, and predictive maintenance.
- Physical meetings possible in Odense with advance notice.

### Francesco Daghero
- Postdoc at DECO Lab, SDU Odense, started November.
- Research: Efficient ML / TinyML - pruning, quantization, dynamic inference, NAS, AI compilers for heterogeneous hardware, RISC-V compilers.
- Age 31; Francesco handles dynamic inference expertise, Emil handles NAS expertise - described as "the perfect match of supervision."

---

## 2. Student Introductions & Background Assessment

### Oliver Larsen
- 2nd semester, MSc Software Engineering, SDU Odense.
- Works as instructor in Data Management and Computer Systems; also research assistant for Professor Mayar on edge AI/inference.
- Recently had a paper accepted at IXA on edge AI on vehicles.
- Developing a startup with Kristoffer.
- Chose project because it's hard and interesting; experience with AI/ML.
- Also runs and did triathlon.

### Aleksander Korsholm
- Chose project due to interest from an AI course.
- AI experience: A* search, basic algorithms (routing/GPS), some linear regression/decision trees exposure.
- No deep learning experience.

### Kristoffer Petersen
- Some experience with deep learning (built a handwritten digit recognizer - basic MNIST-type project).
- Interested in gaining more expertise in the area.

### Oliver Svendsen
- 2nd semester, MSc Software Engineering, SDU Odense.
- Part-time full-stack developer at an app company in Odense.
- No machine learning experience; only basic AI from bachelor courses.

**Summary:** Oliver Larsen has significant ML/edge AI experience. The other three have limited-to-no deep learning background, requiring a learning ramp-up period.

---

## 3. Project Introduction by Supervisors

### 3.1 Edge AI / TinyML Context
- **Target hardware:** Microcontrollers with ~512 KB RAM (bare metal, no OS).
  - No dynamic memory allocation (no malloc, no resizable lists/vectors).
  - Even a Raspberry Pi is considered the "high end" of edge devices.
- **Why edge?** Cheap, low power (battery-operated possible), deployed everywhere (coffee machines, security cameras, offshore windmills).
- **Challenge:** Very limited compute/memory means models must be extremely small and efficient.

### 3.2 Model Optimization Techniques

**Quantization:**
- Convert model weights from high-precision (FP32) to lower-precision (INT8).
- Reduces model size ~4x and speeds up inference (integer ops faster than floating point at hardware level).
- Down to INT8 generally preserves accuracy; below that, still active research.
- Potential fairness issues: minority class accuracy may suffer more.
- **For this project:** Will always quantize models - doesn't make sense to leave them FP32.

**Pruning:**
- Remove weights/neurons that don't contribute significantly to output.
- Saves storage and computation.
- Must be done in a **structured way** so hardware can exploit the sparsity (unstructured pruning can actually be slower on parallel hardware).
- **For this project:** Will encounter pruning through NAS (channel-level pruning). The relationship between pruning and NAS is part of the literature review.

### 3.3 Dynamic Inference (Big/Little Architecture)
- Standard models use the same compute regardless of input difficulty. Dynamic inference changes computation based on input complexity.
- **How it works:**
  1. Image first processed by the **little model** (cheap, less accurate CNN).
  2. Output is a probability distribution over classes.
  3. A **lightweight policy** (max probability or score margin) checks confidence.
  4. If confidence exceeds a user-defined threshold: accept little model's output (stop here).
  5. If not confident enough: run same input through the **big model** and use its output.
- **Policy must be lightweight** (e.g., max probability, score margin) - if policy itself is expensive, it negates the savings.
- **Overheads:**
  - **Memory:** Two models deployed instead of one (but little model is orders of magnitude smaller, so overhead is small).
  - **Compute on hard inputs:** Actually increases (run both models), but this should be rare since most inputs are easy.
- **Applications:** Image classification, time series, etc. For this project: supervised classification (likely image classification on CIFAR-10).

### 3.4 Neural Architecture Search (NAS)
- Neural networks require many design decisions: number of layers, neurons per layer, connections, skip connections, etc.
- Hard to predict which architecture will perform best without training and testing.
- **Manual approach:** Expert iterates on architectures - slow and expensive.
- **NAS approach:** Computer searches through architecture space to find optimal designs.
- **Search methods:**
  1. Reinforcement learning
  2. Evolutionary/genetic algorithms (slow but realistic)
  3. Differentiable NAS (gradient descent on architecture itself - more complex, newer)
- **Advice:** For students without deep learning background, start with traditional approaches (RL, evolutionary), not differentiable NAS.

### 3.5 The Core Research Problem
**Current approach (suboptimal):**
- Little model is designed/searched separately.
- Big model is designed/searched separately.
- They are merged together for dynamic inference.
- Advantage: Can swap/optimize independently.
- Disadvantage: Separate optimization is suboptimal for overall system performance.

**Proposed research direction (joint optimization):**
1. **Step 1:** Extend NAS to output **two models** (little + big) jointly, not separately. Consider memory budget constraints for both.
2. **Step 2:** Further extend NAS to understand the models will run **in cascade** (little first, then big). Optimize the little model specifically for good probability estimation (since that's used for the confidence policy).

**Expectations:**
- Step-by-step approach - getting as far as possible.
- Even just Step 1 (joint NAS for two models) is already an interesting result.
- No failure scenario - understanding background + attempting the direction = success for the course.
- Further progress = more interesting for potential publication.

---

## 4. Compute Resources Discussion

- **Training tiny models:** Laptop is sufficient. Tiny models can actually be *slower* on GPU than CPU because data transfer overhead exceeds the GPU compute speedup.
- **UCloud:** Students have limited default access (~10,000 core hours, U1 standard). Supervisors will investigate getting more resources. Students can be invited to supervisor workspaces.
- **Supervisor servers:** Francesco mentioned potential access to group servers (needs to figure out access).
- **Hardware for deployment:** Francesco can provide microcontrollers if needed for deployment/demonstration.
- **Action:** Supervisors will look into compute resources; students can start on laptops.

---

## 5. Technical Starting Points & Homework (2-week deadline)

### Learning Goals
- What is a neural network / convolutional neural network / convolution
- What is MobileNetV2 (designed via NAS)
- How a PyTorch training loop works
- What is a batch size
- Basic PyTorch declarations and workflow
- How to do automated NAS with PyTorch
- **Reproducibility:** Learn early how to set random seeds everywhere (torch, numpy, random) - critical for paper-quality results

### Deliverables for Next Meeting (~2 weeks)
1. **2 custom (human-designed) CNN architectures** trained on CIFAR-10 with reported results.
2. **1-2 NAS-generated architectures** on CIFAR-10 using existing libraries (do NOT implement NAS from scratch).
3. Show training results for all models.
4. Begin exploring whether existing NAS libraries can be modified to output 2 models.

### Technical Constraints
- Use **PyTorch** (not TensorFlow) - most NAS libraries and modifiable codebases use PyTorch.
- Focus on **CIFAR-10** (small dataset, runs everywhere, plenty of literature).
- Focus on **2D CNNs** for image classification.
- Results must be **reproducible** (same random seeds = same test set numbers).

---

## 6. Advice from Supervisors

1. **Don't be afraid to try things.** Reading about complex topics can be intimidating; just start experimenting.
2. **LLM usage is fine**, but understand the output, verify information, and be able to explain every decision. Everything must be justifiable for a paper.
3. **Be bold but know when to stop.** Creative/risky ideas are welcome, but only after understanding the basics. Don't go fancy during the background phase.
4. **Create a time plan.** Format doesn't matter, but track progress against milestones to know if you're on track.
5. **Start writing early.** Common pitfall is starting the report too late. Document decisions, hyperparameter choices, and reasoning as you go.
6. **Everyone must understand all parts.** Workload splitting is fine, but every team member should be able to explain any section of the project.

---

## 7. Collaboration & Logistics

### Meeting Frequency
- **Decided: Bi-weekly meetings** with the option to request ad-hoc meetings anytime via email.
- Weekly meetings considered too frequent at this stage; may increase frequency closer to deadlines.
- If a week has no progress, skip the meeting rather than waste everyone's time.

### Communication
- **One designated contact person** sends emails to supervisors (always CC both Emil and Francesco).
- Students can visit Francesco's office in Odense (but keep Emil in the loop via meetings).
- Physical meetings possible with advance notice.

### Version Control
- **Create a GitHub repository** (or organization with multiple repos).
- Invite both Emil and Francesco.
- Handle version control however the team prefers, but use it consistently.

### Course Requirements
- Students need to inform supervisors of course deadlines (presentations, reports) with at least 3 days advance notice.
- A mid-course presentation expected ~4-5 weeks in.
- Send presentation drafts to supervisors for feedback before submitting.
- Exam: Oral exam with course supervisor (Abhishek) and project supervisor(s).

### Publication Possibility
- If the project produces novel results, it could lead to a publication.
- The course grade and research output are separate - course expectations are more contained.
- Possible to extend the project into a master's thesis (10 ECTS next semester).

---

## Action Items

| Action | Owner | Deadline |
|--------|-------|----------|
| Share meeting slides with students | Emil/Francesco | ASAP |
| Set up GitHub repository, invite supervisors | Students | This week |
| Learn PyTorch basics (tutorials on pytorch.org) | Students | 2 weeks |
| Train 2 custom CNNs on CIFAR-10 | Students | 2 weeks |
| Run NAS library on CIFAR-10, report results | Students | 2 weeks |
| Learn reproducibility in PyTorch (random seeds) | Students | 2 weeks |
| Start literature review | Students | Ongoing |
| Investigate compute resources (UCloud/servers) | Emil/Francesco | Next meeting |
| Create a time plan for the project | Students | Next meeting |
| Inform supervisors of course deadlines/milestones | Students | ASAP |
| Decide on meeting frequency and communicate | Students | This week |
