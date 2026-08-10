# Heterogeneous Multi-Robot Collaboration Using Language Models

This repository contains the implementation and report for a master's thesis investigating whether locally deployed language models can support grounded, high-level decision-making for heterogeneous robot collaboration in a simulated warehouse.

## Thesis report

[Read or download the latest thesis PDF](Report/main.pdf)

The LaTeX source, bibliography, figures, appendices, review responses, and build files are available in [`Report/`](Report/).

## Study overview

The experiments use a modified Battery-TA-RWARE environment containing AGVs and Picker robots with complementary roles. A supervisory controller converts the current simulator state into a prompt, queries a locally hosted language model, validates the proposed discrete actions, and passes accepted actions to the simulator.

The study evaluates:

- six locally deployable models from approximately 1B to 14B parameters;
- centralized joint planning and shared-context per-agent planning;
- natural-language and JSON-based prompt configurations;
- six warehouse scenarios with different layouts and AGV/Picker balances; and
- prompted objectives concerning shelf delivery, LLM-call usage, and battery management.

The current matched analysis finds that natural-language prompting generally produces stronger delivery results, with a small JSON advantage for Phi4. Shared-context planning produces more deliveries across all six scenario groups and for five of the six models, while centralized planning uses fewer model interactions and performs better for Qwen. Performance does not increase consistently with nominal model size. These results are descriptive observations within the tested simulator and do not establish superiority over non-language-model planning methods.

## Repository structure

| Path | Contents |
| --- | --- |
| [`Battery-TA-RWARE/`](Battery-TA-RWARE/) | Modified warehouse simulator, grounded LLM controllers, experiment runners, tests, and analysis utilities |
| [`Battery-TA-RWARE/scripts/`](Battery-TA-RWARE/scripts/) | Centralized and shared-context runners for the three objective families |
| [`Battery-TA-RWARE/results_summaries/`](Battery-TA-RWARE/results_summaries/) | Git-sized `summary.json` and `summary.txt` files for runs represented in the analysis dataset |
| [`Battery-TA-RWARE/tarware/`](Battery-TA-RWARE/tarware/) | Simulator environment, action definitions, navigation, rendering, and wrappers |
| [`Report/`](Report/) | Thesis source and compiled PDF |
| [`Report/figures/charts/latex/`](Report/figures/charts/latex/) | Native LaTeX/PGFPlots figures used in the report |

Large raw experiment logs and model files are not stored in the repository. The stripped results archive contains only summaries for the Battery, Battery + Shelf, and Calls objectives that are referenced by the `Data Combined - all-runs` analysis sheet; legacy Shelf Deliveries runs and per-step records are excluded. The report documents how repeated executions are averaged into configuration-level observations and how matched configurations are selected for each comparison.

## Python environment

The simulator package requires Python 3.9 or later. LLM experiments additionally require a running [Ollama](https://ollama.com/) service and the model tags selected by the runner.

```bash
cd Battery-TA-RWARE
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[llm,test]"
```

Run the automated tests with:

```bash
pytest
```

## Experiment runners

The current runners are organised by objective and planning architecture:

| Objective | Centralized | Shared context |
| --- | --- | --- |
| Battery + Shelf | `run_obj1_centralized_llm.py` | `run_obj1_shared_context_llm.py` |
| LLM Calls | `run_obj2_centralized_llm.py` | `run_obj2_shared_context_llm.py` |
| Battery | `run_obj3_centralized_llm.py` | `run_obj3_shared_context_llm.py` |

Each runner supports model, scenario, prompt-format, seed, termination, and output-directory arguments. Inspect the complete interface before starting a run:

```bash
cd Battery-TA-RWARE
python scripts/run_obj1_centralized_llm.py --help
```

Experiment runs can take a long time, particularly for larger models and JSON prompting. Choose the required model and scenario explicitly when testing a configuration.

## Building the thesis

The report build requires a LaTeX installation with BibTeX and `makeglossaries`.

```bash
cd Report
make pdf
```

The generated document is written to [`Report/main.pdf`](Report/main.pdf).

## Upstream environment

The simulator is based on TA-RWARE/Battery-TA-RWARE and has been extended with the language-model interface, prompt configurations, validation, action persistence, experiment logging, and battery-aware evaluation used in this thesis. See [`Battery-TA-RWARE/LICENSE`](Battery-TA-RWARE/LICENSE) and [`Battery-TA-RWARE/CITATION.cff`](Battery-TA-RWARE/CITATION.cff) for the upstream licensing and citation information.
