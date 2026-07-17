# Bi-Lipschitz Recurrent Equilibrium Network (BiLipREN)

> 📄 [arXiv:2607.10026](https://arxiv.org/abs/2607.10026): **Robustly Invertible Nonlinear Dynamics and the BiLipREN: From Inversion-Based Control to Generative Trajectory Modelling**

## TL;DR

BiLipREN is a neural dynamical system that defines a robustly invertible signal-to-signal mapping.

![Invertible mapping](figures/invertible_mapping.png)

The REN architecture $\mathcal{G}$ is a feedback interconnection between a learnable LTI system $\boldsymbol{G}$ and a fixed nonlinear activation $\sigma$.

<p align="center">
  <img src="figures/REN.png" alt="REN architecture" width="200">
</p>

The following properties are guaranteed *by construction* and are compatible with automatic differentiation and stochastic gradient descent:

1. The forward model $y=\mathcal{G}(u)$ is an invertible, stable, and bi-Lipschitz REN.

2. Its analytical inverse $u=\mathcal{G}^{-1}(y)$ is a causal, stable, and bi-Lipschitz REN.

3. Both models enable robust signal reconstruction under disturbances and initial-state mismatch:

$$
\begin{aligned}
\lVert e_u\rVert_T
&\leq
\lambda_{xu}\lvert a-b\rvert
+\lambda_{yu}\lVert\Delta y\rVert_T,\
\lVert e_y\rVert_T
&\leq
\lambda_{xy}\lvert a-b\rvert
+\lambda_{uy}\lVert\Delta u\rVert_T.
\end{aligned}
$$

<p align="center">
  <img src="figures/robust-inverse.png" alt="Robust inverse" width="320">
</p>

## Applications

### 1. Optimization-Aware Dynamic Surrogate Loss

**TL;DR:** *Learn an optimization-friendly surrogate loss for black-box trajectory optimization.*

#### Black-box trajectory optimization

Suppose that $f$, $a$, $x_t$, $c_t$, and $c_f$ are unknown, and only a dataset

$$
\mathcal{D}
===========

\left{
\left(u_{[0:T]}^i,J^i\right)
:
1\leq i\leq n
\right}
$$

is available. The underlying trajectory-optimization problem is

$$
\begin{aligned}
\min_{u_{[0:T]}\in\ell^m}\quad
&
J\left(u_{[0:T]}\right)
:=
c_f(x_{T+1})
+\sum_{t=0}^{T}c_t(x_t,u_t)\
\text{subject to}\quad
&
x_{t+1}=f(x_t,u_t),
\qquad
x_0=a.
\end{aligned}
$$

**Can we find a new input sequence $u_{[0:T]}$ that is likely to achieve a lower cost than every sample in the dataset?**

#### Surrogate-optimization framework

1. Fit a differentiable surrogate loss to the dataset:

   $$
   \widehat{J}\left(u_{[0:T]}\right)
   =================================

   \frac{1}{2}
   \left\lVert
   \mathcal{G}\left(u_{[0:T]}\right)
   \right\rVert^2
   +c,
   $$

   where $\mathcal{G}$ is a neural dynamical model that captures temporal structure and $c\in\mathbb{R}$ is a learnable parameter.

2. Optimize the surrogate loss:

   $$
   \widehat{u}*{[0:T]}^\star
   :=
   \operatorname*{arg,min}*{u_{[0:T]}\in\ell^m}
   \widehat{J}\left(u_{[0:T]}\right).
   $$

#### Our approach

We parameterize $\mathcal{G}$ as a BiLipREN, giving the surrogate loss $\widehat{J}$ two desirable properties:

1. It satisfies the Polyak–Łojasiewicz (PL) condition. Consequently, despite being nonconvex, it has no spurious local minima, and gradient-based methods converge linearly under standard step-size conditions.

2. Its minimizer can be computed efficiently through dynamic inversion:

   $$
   \widehat{u}_{[0:T]}^\star
   =========================

   \mathcal{G}^{-1}(0).
   $$

#### Results

| Model    | Fitting loss $L$ | Best cost $J$ | Worst cost $J$ |
| -------- | ---------------: | ------------: | -------------: |
| Dataset  |                — |          1863 |           5055 |
| LSTM     |             1718 |          1868 |           4758 |
| C-REN    |             6014 |          1918 |           2996 |
| BiLipREN |            22805 |          1672 |              — |
| IPOPT    |                — |          1618 |           5837 |

1. The *LSTM* fits the dataset well but is less suitable for the subsequent optimization step because its loss landscape may contain spurious local minima, flat regions, poorly conditioned gradients, and strong sensitivity to initialization.

2. The *C-REN*, which is stable but not necessarily invertible, encounters similar difficulties.

3. The *BiLipREN* has a higher fitting loss but yields a lower optimized cost and a more tractable optimization landscape.

4. Even when *IPOPT* is applied to the true optimization problem, poor initial guesses can produce poor solutions because the problem is highly nonconvex.

<p align="center">
  <img src="figures/mbd_pl.png" alt="Surrogate-cost trajectory optimization" width="640">
</p>

### 2. Signal-to-Signal Normalizing Flow

**TL;DR:** *Learn a signal-to-signal normalizing flow that generates trajectory distributions from Gaussian white noise.*

#### Generative trajectory modelling

We seek a robustly invertible dynamical model $\mathcal{G}$ that generates samples matching the data distribution:

$$
y_{[0:T]}
=========

\mathcal{G}\left(u_{[0:T]}\right),
\qquad
u_t\sim\mathcal{N}(0,I).
$$

The model is trained by minimizing the negative log-likelihood under the normalizing-flow change-of-variables formula:

$$
\mathcal{L}_{\mathrm{NLL}}
==========================

-\sum_{t=0}^{T}
\left(
\log p_u(u_t)
+
\log
\left\lvert
\det
\left(
\frac{\partial u_t}{\partial y_t}
\right)
\right\rvert
\right).
$$

#### Results

1. The generated trajectories capture the multimodal, obstacle-avoiding distribution of the training data.

   <p align="center">
     <img src="figures/mbd_generation.png" alt="Dataset versus generated trajectories" width="500">
   </p>

2. Mapping the data through $\mathcal{G}^{-1}$ produces approximately white Gaussian latent variables: their autocorrelations remain within the 95% confidence band, and their Q–Q plot closely follows that of a standard Gaussian distribution.

   <p align="center">
     <img src="figures/mbd_acf_qq.png" alt="Latent ACF and Q-Q plots" width="400">
   </p>

### 3. Inversion-Based Control Design

**TL;DR:** *Design a tracking controller for a stable, nonminimum-phase plant.*

#### Internal model control

We learn an inner–outer factorization of the plant and invert only its minimum-phase outer factor, thereby obtaining a stable controller.

<p align="center">
  <img src="figures/imc_blk_diag.png" alt="Internal model control block diagram" width="400">
</p>

1. Learn an inner–outer factorization from input–output data generated by the true system $\mathcal{P}$:

   $$
   \widehat{\mathcal{P}}
   =====================

   \boldsymbol{O}(z)\circ\mathcal{G},
   $$

   where the inner factor $\boldsymbol{O}(z)$ is an all-pass filter—a stable but nonminimum-phase system—and the outer factor $\mathcal{G}$ is a BiLipREN representing a stable minimum-phase system.

2. Construct the IMC controller in Youla form with the $\mathcal{Q}$ parameter

   $$
   \mathcal{Q}
   ===========

   \mathcal{R}\circ\widehat{\mathcal{P}}^{\sharp},
   $$

   where $\mathcal{R}$ is a low-pass filter and

   $$
   \widehat{\mathcal{P}}^{\sharp}
   ==============================

   \mathcal{G}^{-1}
   \circ
   \boldsymbol{O}^{\top}(1)
   $$

   is an approximate inverse of $\widehat{\mathcal{P}}$. For piecewise-constant input signals, the reconstruction error

   $$
   e_u
   ===

   \widehat{\mathcal{P}}
   \circ
   \widehat{\mathcal{P}}^{\sharp}(u)
   -u
   $$

   converges exponentially to zero.

#### Results

The controller achieves reference tracking for a four-tank system with delayed input flow.

<p align="center">
  <img src="figures/imc_closed_loop.png" alt="Closed-loop tracking performance" width="400">
</p>

## Get Started

```bash
git clone https://github.com/acfr/BiLipREN.git
cd BiLipREN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Layout

| Folder            | Description                                       |
| ----------------- | ------------------------------------------------- |
| `BiLipRENs/`      | Core BiLipREN models and orthogonal layers        |
| `surrogate_cost/` | Application 1 — dynamic surrogate-loss learning   |
| `flow/`           | Application 2 — signal-to-signal normalizing flow |
| `imc/`            | Application 3 — inversion-based control design    |
| `io_fact/`        | Example: nonlinear input–output factorization     |
| `robust_inv/`     | Example: robust inversion                         |

## Contacts

Yurui Zhang (*[yurui.zhang@sydney.edu.au](mailto:yurui.zhang@sydney.edu.au)*)
Ruigang Wang (*[ruigang.wang@sydney.edu.au](mailto:ruigang.wang@sydney.edu.au)*)
