# Verification Guide

This document explains the deterministic verification scenarios in [tests/verification/test_deterministic_verification.py](tests/verification/test_deterministic_verification.py).

The purpose of these tests is to verify that the simulator matches the mathematical model using manually computed reference values. In every case, the expected value is derived analytically inside the test and compared against simulator output with a tolerance of `1e-6`.

## Verification Principles

1. The simulator output is never reused as the expected result.
2. Every expected quantity is computed from the governing equations.
3. Each scenario isolates one behavior or metric.
4. The scenarios are deterministic and small enough to audit by hand.

## Scenarios

### 1. Single Task on One VM

Purpose:
Validate the simplest execution path.

What is checked:
1. execution time
2. waiting time
3. response time
4. makespan
5. VM energy
6. total energy
7. utilization interval

Manual logic:
- Execution time is:
  $\Delta = L / C^{cpu}$
- Waiting time is zero because the VM is immediately available.
- Response time equals execution time.
- Makespan equals the finish time of the single task.
- VM energy equals idle energy plus active static energy plus dynamic energy.

### 2. Two Sequential Tasks on One VM

Purpose:
Verify FIFO queueing on a single VM.

What is checked:
1. average waiting time
2. average response time
3. makespan

Manual logic:
- The first task starts immediately.
- The second task starts only after the first task finishes.
- Waiting and response values are computed task by task, then averaged.

### 3. Two VMs with No Queueing

Purpose:
Verify parallel execution with independent VMs.

What is checked:
1. zero waiting time
2. response time under no queueing
3. makespan
4. throughput

Manual logic:
- Each task is assigned to a different VM.
- Both start immediately.
- Makespan is the maximum of the two completion times.
- Throughput is:
  $N / \text{makespan}$

### 4. VM Carry-Over with `availability_time > 0`

Purpose:
Verify cross-epoch carry-over semantics.

What is checked:
1. waiting time due to carry-over
2. response time
3. delayed start behavior

Manual logic:
- The VM is unavailable until `availability_time`.
- The task cannot start before that time even if it arrives earlier.
- Waiting time is the difference between actual start and arrival.

### 5. PM Base Energy Validation

Purpose:
Verify the fixed PM energy term.

What is checked:
1. PM base energy only

Manual logic:
- PM base energy is:
  $E_s^{base} = P_s^{base} \cdot H$
- This term does not depend on the task schedule.

### 6. VM Dynamic Energy Validation

Purpose:
Verify the dynamic part of the VM energy model.

What is checked:
1. dynamic energy only

Manual logic:
- With the current calibration used in verification, dynamic energy is integrated over the utilization interval.
- The formula is:
  $E_{dyn} = \int (\alpha u + \beta u^2) \, dt$
- In the test, utilization is constant over the interval, so this reduces to interval length times the polynomial value.

### 7. Waiting Time Validation

Purpose:
Verify task waiting behavior when tasks arrive before the VM queue drains.

What is checked:
1. average waiting time

Manual logic:
- Waiting time per task is:
  $W_i = s_i - a_i$
- The test computes both values manually and averages them.

### 8. Response Time Validation

Purpose:
Verify end-to-end task delay.

What is checked:
1. average response time

Manual logic:
- Response time per task is:
  $R_i = f_i - a_i$
- The test computes response per task and averages it.

### 9. Makespan Validation

Purpose:
Verify the total schedule completion time.

What is checked:
1. makespan

Manual logic:
- Makespan is the latest task finish time across all VMs.

### 10. SLA Penalty Validation

Purpose:
Verify deadline overshoot and the exponential SLA penalty.

What is checked:
1. deadline violation
2. normalized violation
3. aggregate SLA penalty

Manual logic:
- Deadline violation is:
  $\delta_i = \max(0, f_i - d_i)$
- Normalized violation is:
  $\eta_i = \delta_i / d_i$
- Penalty follows the piecewise exponential formula from the system model.

### 11. Jain Fairness Validation

Purpose:
Verify fairness computation from per-task SLA penalties.

What is checked:
1. Jain fairness index

Manual logic:
- With penalties $\phi_i$, Jain index is:
  $\mathcal{J}(\phi) = \frac{(\sum_i \phi_i)^2}{N \sum_i \phi_i^2}$
- The expected value is computed manually from the task penalties.

## How to Run Only Verification Tests

From the repository root:

```powershell
$env:PYTHONPATH="src"
python -m unittest tests.verification.test_deterministic_verification -v
```

## How to Run the Full Test Suite

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

## Why This Matters

These tests are not just regression tests. They are equation-conformance tests. They give you evidence that the simulator behavior is numerically aligned with the mathematical model before you start adding more advanced schedulers such as GA or LLM-guided GA.
