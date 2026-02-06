# CLAUDE.md

## Project Context

This is an Engineering Research project at SDU exploring Neural Architecture Search (NAS) for big/little dynamic inference networks optimized for edge deployment. The codebase uses Python and PyTorch.

## Progress Tracking

**Update `PROGRESS.md` when:**
- A task, feature, or milestone is completed
- A key decision is made (e.g., choosing a NAS strategy, dataset, or hyperparameter)
- A new open question arises that affects the project direction
- Experiment results are obtained

Add entries under the current week. If a new week has started, create a new week section using the template at the bottom of the file.

## Code Standards

### Python
- Python 3.10+
- Use type hints for function signatures
- Follow PEP 8 naming conventions
- Use `pathlib.Path` over `os.path`

### PyTorch
- Use `torch.nn.Module` for all model components
- Keep model definitions in `src/models/`, training logic in `src/training/`
- Set random seeds for reproducibility (`torch.manual_seed`, `numpy`, `random`)
- Log all hyperparameters and configs alongside experiment results

### Project Structure
- Follow the repository structure defined in `README.md`
- Config files go in `configs/` as YAML
- Experiment results, logs, and checkpoints go in `experiments/`
- Notebooks are for exploration only -- production code belongs in `src/`

## Experiments

- Every experiment must have a config file that fully reproduces it
- Log metrics to a structured format (CSV or JSON) in `experiments/`
- Name experiment runs descriptively: `{dataset}_{method}_{date}` (e.g., `cifar10_darts_20260215`)
- Never overwrite previous experiment results -- keep all runs

## Git Practices

- Use feature branches, never commit directly to main
- Write descriptive commit messages
- Stage specific files, not `git add -A`
- Keep `.gitignore` updated -- never commit datasets, checkpoints, or `.DS_Store`

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and objectives |
| `RESEARCH_PLAN.md` | Detailed research plan with timeline and technical approach |
| `PROGRESS.md` | Living document tracking completed work, decisions, and open questions |
| `working_document.md` | Research paper draft -- fill in sections as work progresses |
| `project-description/` | Original project description from supervisor |
