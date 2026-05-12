# RL-based Memristor Programming: Beyond Neural Pulse Predictors

This repository contains a PyTorch-based framework for high-precision iterative programming of analog resistive memory (ReRAM/Memristors). 

The project reproduces the state-of-the-art Neural Pulse Predictor (NPP) baseline from the *Ouroboros* methodology (Yu et al., 2024) and extends it by introducing a Reinforcement Learning (RL) agent. The goal is to overcome the fundamental limitations of greedy supervised learning algorithms in highly stochastic and non-linear physical environments.

## 1. Physical Foundation (Memristor Model)

To ensure the research is physically grounded and not merely an algorithmic exercise on synthetic data, the environment utilizes the `LinearStepDevice` from IBM's `aihwkit`. The simulation explicitly models the non-ideal behaviors of real filamentary ReRAM devices (e.g., JART v1b):

* **Asymmetry:** SET (conductance increase) and RESET (conductance decrease) operations exhibit different physical dynamics ($up\_down = 0.5$).
* **Non-linearity (Saturation):** The device response exponentially decays as it approaches the boundary conductances ($G_{min} = -1.5$, $G_{max} = 1.5$).
* **Multiplicative Cycle-to-Cycle Noise:** The variance of the physical response scales with the applied pulse duration ($mult\_noise = True$), making large state jumps highly unpredictable.

## 2. Methodology: Neural Pulse Predictor (Baseline)

The baseline is a Multilayer Perceptron (MLP: 2 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 1) trained to solve the inverse physical function:
$t_{pulse} = f^{-1}(G_{start}, \Delta G_{target})$

### Data Generation & Curriculum Learning
To prevent the model from failing to converge due to extreme physical noise, the NPP is trained using a Curriculum Learning approach on 1,000,000 generated samples:
1. Data is binned by $G_{start}$ and direction (SET/RESET).
2. Moving average smoothing is applied to extract the underlying physical trend.
3. The network trains on progressively noisier data (windows: 501 $\rightarrow$ 101 $\rightarrow$ 21 $\rightarrow$ 1).

**Loss Function:** Mean Absolute Percentage Error (MAPE) is used instead of MSE to properly penalize relative errors on extremely short pulses.

### Baseline Performance (Iterative Programming / Write-Verify)
In a closed-loop iterative programming benchmark (max 10 steps, absolute tolerance $\pm 0.05$):

* **Median Relative Percentage Deviation (RPD):** ~18.45%
* **Success Rate:** 44.90%
* **Average Steps:** 7.21
* **Mean Final Error:** 0.1356

**Limitation of the Baseline:** NPP is a greedy algorithm. It attempts to reach the target in a single large pulse. Due to multiplicative noise, large pulses cause massive stochastic deviations, leading to overshoots and infinite oscillations around the target.

## 3. Reinforcement Learning Solution: Active Inference Policy

To surpass the limitations of greedy one-shot regression, the programming process is redefined as a Markov Decision Process (MDP) where the objective is not weight-matching, but direct optimization of the global task loss.

### Policy Architecture
The agent implements a Stochastic Gaussian Policy $\pi(a|s) = \mathcal{N}(\mu, \sigma)$ with the following constraints:

* **Mean ($\mu$):** A parsimonious linear function of the observations: $\mu(s) = Ws + b$. This ensures stability and prevents over-parameterization of the pulse logic.
* **Variance ($\sigma$):** A standalone trainable constant parameter ($\log \sigma$). It is independent of the observation to maintain a consistent exploration pressure throughout the state space.
* **Action Space:** Continuous $a \in [-1, 1]$, where $|a|$ maps to $t_{pulse}$ and $sign(a)$ determines the polarity (SET/RESET).

### Observation Space (State)
The state vector $s$ purposefully excludes the explicit target conductance ($G_{target}$) to force the agent to learn the underlying gradient of the task. 

**State components:**
1.  **Local State:** Current conductance $G_{current}$ of the memristive cell.
2.  **Structural Context:** Activations from the preceding layer ($h_{l-1}$) and the subsequent layer ($h_{l+1}$). This provides the "synaptic context" necessary to understand the importance of the weight in the current forward pass.

### Reward Mechanism: Cross-Entropy Driven
The reward function is strictly tied to the application-level performance.

* **Primary Reward:** Negative Cross-Entropy Loss ($- \mathcal{L}_{CE}$) of the neural network on the current batch.
* **Entropy Regularization:** Policy entropy is utilized during the update phase to prevent premature convergence to a sub-optimal deterministic pulse strategy.
* **No Explicit Targeting:** By rewarding the reduction of Cross-Entropy, the agent implicitly discovers the optimal conductance values. This eliminates the need for a predefined $G_{target}$, bypassing the error-prone step of calculating "ideal" weights before programming.

### Advantage over NPP
Unlike the NPP baseline, which treats every pulse as an isolated regression task, this RL approach:
1.  **Learns the Cost of Action:** Accounts for device asymmetry—understanding that an overshoot in SET is physically harder to correct than a small undershoot.
2.  **Context-Aware Tuning:** Adjusts pulse precision based on the magnitude of layer activations, effectively implementing a physical version of adaptive learning rate (like Adam) at the hardware level.

### Results Comparison

| Metric | NPP (Greedy Baseline) | RL Agent (Proposed) |
| :--- | :--- | :--- |
| **Success Rate (Tol $\pm$ 0.05)** | 44.90% | *TBD* |
| **Average Steps** | 7.21 | *TBD* |
| **Median Single-step RPD** | 18.45% | *N/A* |
| **Mean Final Error** | 0.1356 | *TBD* |

## 4. Installation & Requirements

The project requires a specific environment setup to compile the C++ core of `aihwkit`.

```bash
# Clone the repository
git clone [https://github.com/AndrewArzumaniyan/rl-init-mem.git](https://github.com/AndrewArzumaniyan/rl-init-mem.git)
cd rl-init-mem

# Install dependencies (ensure matching CUDA versions)
pip install torch torchvision numpy
# System libs
sudo apt-get install libopenblas-dev
# Prevents version conflicts
pip install -r requirements.txt --no-build-isolation