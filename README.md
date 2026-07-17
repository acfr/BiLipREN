# Bi-Lipschitz Recurrent Equilibrium Network (BiLipREN)

> 📄 [arXiv:2607.10026](https://arxiv.org/abs/2607.10026): **Robustly Invertible Nonlinear Dynamics and the BiLipREN: From Inversion-Based Control to Generative Trajectory Modelling** 

## TL;DR

BiLipREN is a neural dynamical system that defines a robustly invertible signal-to-signal mapping.

![invertible mapping](figures/invertible_mapping.png)

The REN architecture <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/G_dark.png"><img alt="G" src="figures/eq/inline/G.png" height="18"></picture> is a feedback interconnection between a learnable LTI system <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/bold_G_dark.png"><img alt="bold G" src="figures/eq/inline/bold_G.png" height="18"></picture> and a fixed nonlinear activation <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/sigma_dark.png"><img alt="sigma" src="figures/eq/inline/sigma.png" height="18"></picture>.

<p align="center"><img src="figures/REN.png" alt="REN architecture" width="200"></p>

The following properties are guaranteed *by construction* (plug-and-play with AutoDiff and SGD): 

1. The forward model <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/forward_dark.png"><img alt="y = G(u)" src="figures/eq/inline/forward.png" height="18"></picture> is an invertible, stable and bi-Lipschitz REN.

2. Its analytical inverse <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/inverse_dark.png"><img alt="u = inverse G(y)" src="figures/eq/inline/inverse.png" height="18"></picture> is a causal, stable and bi-Lipschitz REN.


3. Both models enable robust signal reconstruction under disturbances and initial-state mismatch:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_reconstruction_dark.png">
    <img alt="robust reconstruction bounds" src="figures/eq/eq_reconstruction.png" width="250">
  </picture>
</p>

<p align="center"><img src="figures/robust-inverse.png" alt="Robust inverse" width="320"></p>

## Applications

### 1. Optimization-Aware Dynamic Surrogate Loss 

**TD;LR:** *Learn an optimization-friendly surrogate loss for black-box trajectory optimization*

**Black-box Trajectory Optimization.** Suppose that <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/unknowns_dark.png"><img alt="f, a, c_t, c_f" src="figures/eq/inline/unknowns.png" height="18"></picture> are unknown, and only a dataset <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/dataset_dark.png"><img alt="sampled input-loss pairs" src="figures/eq/inline/dataset.png" height="20"></picture> is available:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_problem_dark.png">
    <img alt="black-box trajectory optimization problem" src="figures/eq/eq_problem.png" width="350">
  </picture>
</p>

**Can we find a new input sequence <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/u_T_dark.png"><img alt="u_[T]" src="figures/eq/inline/u_T.png" height="18"></picture> that is likely to achieve a lower cost than any sample in the dataset?**

- **Surrogate optimization framework:**

1. Fit a differentiable surrogate loss to the dataset:
   
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_surrogate_dark.png">
    <img alt="surrogate loss" src="figures/eq/eq_surrogate.png" width="200">
  </picture>
</p>

where <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/G_dark.png"><img alt="G" src="figures/eq/inline/G.png" height="18"></picture> is a neural dynamical model that captures temporal structure and <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/c_real_dark.png"><img alt="c in R" src="figures/eq/inline/c_real.png" height="18"></picture> is a learnable parameter. 

2. Optimize the surrogate loss:
   
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_argmin_dark.png">
    <img alt="surrogate loss minimization" src="figures/eq/eq_argmin.png" width="200">
  </picture>
</p>


- **Our approach**: parameterize <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/G_dark.png"><img alt="G" src="figures/eq/inline/G.png" height="18"></picture> as a BiLipREN, giving the surrogate <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/J_hat_dark.png"><img alt="J hat" src="figures/eq/inline/J_hat.png" height="18"></picture> two nice properties:

1. It satisfies the Polyak–Łojasiewicz (PL) condition. Consequently, despite being nonconvex, it has no spurious local minima, and gradient-based methods converge linearly under standard step-size conditions.
2. The minimizer can be computed efficiently through dynamic inversion:
   
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_inversion_dark.png">
    <img alt="dynamic inversion" src="figures/eq/eq_inversion.png" width="120">
  </picture>
</p>

- **Results:**

<div align="center">
   
| Model | Fitting loss $L$ | Best cost $J$ | Worst cost $J$ |
| -------- | -------- | -------- | -------- |
| Dataset | - | 1863 | 5055 |
| LSTM | 1718 | 1868 | 4758 |
| C-REN | 6014 | 1918 | 2996 |
| BiLipREN | 22805 | 1672 | - |
| IPOPT | - | 1618 | 5837 |

</div>

1.  The *LSTM* fits the dataset well but is less suitable for the subsequent optimization step because its loss landscape may contain spurious local minima, flat regions, poorly conditioned gradients, and strong sensitivity to initialization.

2. The *C-REN*, which is stable but not necessarily invertible, encounters similar difficulties.

3. The *BiLipREN* has a higher fitting loss but yields a lower optimized cost and a more tractable optimization landscape.

4. Even when *IPOPT* is applied to the true optimization problem, poor initial guesses can produce poor solutions because the problem is highly nonconvex.

<p align="center"><img src="figures/mbd_pl.png" alt="Surrogate-cost trajectory optimization" width="640"></p>

### 2. Signal-to-Signal Nomralizing Flow

**TL;DR:** *Learn a signal-2-signal normalizing flow that generates trajectory distributions from Gaussian white noise*

**Generative trajectory modelling.** We seek a robustly invertible dynamical model <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/G_dark.png"><img alt="G" src="figures/eq/inline/G.png" height="18"></picture> that generates samples matching the data distribution:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_generation_dark.png">
    <img alt="generative model" src="figures/eq/eq_generation.png" width="200">
  </picture>
</p>

The model is trained by minimizing the negative log-likelihood (NLL) under the normalizing-flow change-of-variables formula:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_nll_dark.png">
    <img alt="negative log-likelihood loss" src="figures/eq/eq_nll.png" width="280">
  </picture>
</p>

- **Results:**

1. The generated trajectories capture the multimodal, obstacle-avoiding distribution of the training data.

<p align="center"><img src="figures/mbd_generation.png" alt="Dataset vs. generated trajectories" width="500"></p>

2. Mapping the data through <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/G_inverse_dark.png"><img alt="inverse G" src="figures/eq/inline/G_inverse.png" height="18"></picture> produces approximately white Gaussian latent variables: their autocorrelations remain within the 95% confidence band, and their Q–Q plot closely follows that of a standard Gaussian distribution.

<p align="center"><img src="figures/mbd_acf_qq.png" alt="Latent ACF and Q-Q plots" width="400"></p>

### 3. Inversion-based Control Design

**TL;DR:** *Design a tracking controller for a stable, nonminimum-phase plant.*

**Internal model control (IMC).** We learn an inner–outer factorization of the plant and invert only its minimum-phase outer factor, thereby obtaining a stable controller.

<p align="center"><img src="figures/imc_blk_diag.png" alt="Internal model control block diagram" width="400"></p>

  1. Learn an inner–outer factorization from input–output data generated by the true system <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/P_dark.png"><img alt="P" src="figures/eq/inline/P.png" height="18"></picture>:
     
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_iofact_dark.png">
    <img alt="inner-outer factorization" src="figures/eq/eq_iofact.png" width="100">
  </picture>
</p>
  
  where the inner factor <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/O_z_dark.png"><img alt="O(z)" src="figures/eq/inline/O_z.png" height="18"></picture> is an all-pass filter (stable but non-minimum-phase system) and the outer factor <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/G_dark.png"><img alt="G" src="figures/eq/inline/G.png" height="18"></picture> is a BiLipREN (stable minimum-phase system). 
  
  2. Construct the IMC controller in the Youla form with <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/Q_dark.png"><img alt="Q" src="figures/eq/inline/Q.png" height="18"></picture>-parameter
     
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/eq/eq_Q_dark.png">
    <img alt="IMC controller" src="figures/eq/eq_Q.png" width="100">
  </picture>
</p>
  
  where <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/R_dark.png"><img alt="R" src="figures/eq/inline/R.png" height="18"></picture> is a low-pass filter and <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/P_sharp_dark.png"><img alt="approximate inverse P sharp" src="figures/eq/inline/P_sharp.png" height="20"></picture> is an approximate inverse of <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/P_hat_dark.png"><img alt="P hat" src="figures/eq/inline/P_hat.png" height="18"></picture>. For piecewise-constant inputs, <picture><source media="(prefers-color-scheme: dark)" srcset="figures/eq/inline/e_u_dark.png"><img alt="input reconstruction error" src="figures/eq/inline/e_u.png" height="20"></picture> converges exponentially to zero.


- **Results.** The controller achieves reference tracking for a four-tank system with delayed input flow. 

<p align="center"><img src="figures/imc_closed_loop.png" alt="Closed-loop tracking performance" width="400"></p>

## Get started

```bash
git clone https://github.com/acfr/BiLipREN.git
cd BiLipREN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Repository layout

| Folder | Description |
| --- | --- |
| `BiLipRENs/` | Core: BiLipREN models and orthogonal layers |
| `surrogate_cost/` | Application 1 — dynamic surrogate loss learning. |
| `flow/` | Application 2 — signal-to-signal normalizing flow. |
| `imc/` | Application 3 — inversion-based control design. |
| `io_fact/` | Example: nonlinear I/O factorization. |
| `robust_inv/` | Example: robust inversion |

## Contacts

Yurui Zhang (*yurui.zhang@sydney.edu.au*) 

Ruigang Wang (*ruigang.wang@sydney.edu.au*)
