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

The experiments were long-running, and the complete outputs are too large to include in the Git repository because each run contains detailed per-step records and logs. A stripped version is available in [`Battery-TA-RWARE/results_summaries/`](Battery-TA-RWARE/results_summaries/), containing only `summary.json` and `summary.txt` for the Battery, Battery + Shelf, and Calls objectives. Legacy Shelf Deliveries runs and per-step records are excluded.

The combined run data and analysis are available in the [experiment results spreadsheet](Report/Analysis/Updated-all-runs.xlsx), particularly the `Data Combined - all-runs` sheet. The report documents how repeated executions are averaged into configuration-level observations and how matched configurations are selected for each comparison.

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

## Docker with GPU and GUI support

The repository-level [`Dockerfile`](Dockerfile) provides the Ollama, Python, GPU, and rendering dependencies used for the experiments. The host must have Docker, NVIDIA Container Toolkit, a working NVIDIA driver, X11, and `xauth`. Build the image from the repository root:

```bash
sudo docker build -t ollama-battertarware-gui:v1 .
```

Set the source directory to the absolute location of `Battery-TA-RWARE` on the host, prepare the X11 authorization file, and create the container:

```bash
export TARWARE_SOURCE=/absolute/path/to/polyphony/Battery-TA-RWARE
export TARWARE_XSOCK=/tmp/.X11-unix
export TARWARE_XAUTH="${HOME}/.docker.xauth"

touch "$TARWARE_XAUTH"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$TARWARE_XAUTH" nmerge -

sudo docker run --network=host \
  --name tarware-remote-gui \
  --gpus all \
  -it \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=/root/.docker.xauth \
  -v "$TARWARE_XSOCK:$TARWARE_XSOCK:rw" \
  -v "$TARWARE_XAUTH:/root/.docker.xauth:rw" \
  -v "$TARWARE_SOURCE:/docker-mount/Battery-TA-RWARE" \
  ollama-battertarware-gui:v1
```

Inside the container, install the mounted project and optional X11 test utilities:

```bash
cd /docker-mount/Battery-TA-RWARE
apt-get update
apt-get install -y xauth x11-apps
python3 -m pip install --break-system-packages -e .
python3 -m pip install --break-system-packages -e ".[llm]"
```

Start Ollama inside the container and leave it running:

```bash
ollama serve
```

From another host terminal, open a second shell in the same container, obtain the required model, and run a small current-runner example:

```bash
sudo docker exec -it tarware-remote-gui bash
cd /docker-mount/Battery-TA-RWARE
ollama pull llama3.2:3b
python3 scripts/run_obj1_centralized_llm.py \
  --prompt_format language \
  --selected_models llama3.2:3b \
  --only_scenario tiny_balanced_1v1 \
  --seed 0 \
  --render
```

After the container has been stopped, restart and attach to it with:

```bash
sudo docker start -ai tarware-remote-gui
```

To open an additional shell while it is running:

```bash
sudo docker exec -it tarware-remote-gui bash
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
