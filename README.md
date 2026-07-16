# LLMGA Simulator

Research-oriented discrete-event simulator for virtualized cloud scheduling.

This repository provides a modular simulation stack for evaluating scheduling policies under timing, energy, SLA, and fairness objectives. It is designed so researchers can reproduce baseline runs quickly and then plug in custom schedulers (for example GA/LLM-guided approaches).

## What This Simulator Covers

- Event-driven cloud simulation (no periodic sampling required for core timing/utilization metrics).
- Datacenter model with physical machines, virtual machines, and tasks.
- Scheduler abstraction with a built-in random scheduler baseline.
- Post-simulation metrics pipeline:
	- makespan, throughput, waiting time, response time
	- VM/PM energy and total energy
	- SLA penalties
	- fairness metrics
	- composite fitness
- Deterministic verification scenarios with analytical reference values.

## Project Structure

- `src/desim/`: simulator core and analytics modules.
- `tests/`: unit and integration tests for all modules.
- `tests/verification/`: deterministic equation-conformance verification tests.
- `examples/sample_run.py`: run a complete sample simulation from config + dataset.
- `examples/sample_config.yaml`: sample experiment configuration.
- `examples/sample_dataset.json`: sample workload and infrastructure dataset.
- `examples/manual_example_for_verification.py`: direct in-code scenario example.
- `VERIFICATION.md`: explanation of deterministic validation methodology.

## Requirements

- Python 3.10+
- OS: Windows/Linux/macOS

## Setup

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
python -m pip install pyyaml
```

If your shell session does not automatically discover `src`, set:

```powershell
$env:PYTHONPATH="src"
```

## Quick Start

Run the sample experiment:

```powershell
$env:PYTHONPATH="src"
python examples/sample_run.py
```

Expected output includes:

- `Makespan`
- `Throughput`
- `Energy`
- `Waiting time`
- `Response time`
- `SLA penalty`
- `Fairness`
- `Utilization`
- `Fitness`

## Reproducibility Notes

- Use `examples/sample_config.yaml` to control random seeds and experiment parameters.
- Important seed fields:
	- `random_seeds.global_seed`
	- `random_seeds.scheduler_seed`
- For repeatable baselines, keep seeds fixed and document Python version.

## Dataset Format (JSON)

The simulator validates dataset schema before execution.

Top-level fields:

- `epoch_length`
- `datacenter`
- `tasks`

Required task fields (each task row):

- `task_id`
- `workload_mi`
- `arrival_time`
- `deadline`
- `cpu_demand_mips`
- `memory_demand_mb`
- `io_size_mb`

If required fields are missing, execution stops with a validation error.

## Core Configuration (YAML)

The sample config `examples/sample_config.yaml` includes:

- `simulation`: stopping conditions (`until`, `max_events`)
- `scheduler`: scheduler type and options
- `energy`: calibration coefficients
- `metrics`: SLA/fairness/fitness parameters
- `dataset`: source path and format
- `random_seeds`: reproducibility control
- `plugins`: future extension hooks

## Verification and Testing

Run deterministic verification suite:

```powershell
$env:PYTHONPATH="src"
python -m unittest tests.verification.test_deterministic_verification -v
```

Run full tests:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

See `VERIFICATION.md` for analytical details of all deterministic scenarios.

## Extending With New Schedulers

Implement the scheduler interface in `src/desim/scheduling.py` and return a task-to-VM mapping (`SchedulingResult`).

Recommended workflow:

1. Start from deterministic scenarios in `tests/verification/`.
2. Add unit tests for scheduler-specific constraints.
3. Compare against random baseline and report seed values.
4. Export and archive configuration + dataset used for each result table.

## Citation and Usage in Papers

When reporting results produced by this simulator, include at minimum:

- commit hash
- config file used
- dataset file used
- seed values
- Python version

This is enough for another researcher to reproduce your run.
