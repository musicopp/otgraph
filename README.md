# otgraph
# A Variational Deep Learning Framework for Dynamic Optimal Transport on Graphs


We provide deep learning framework that solves the **dynamic formulation of optimal transport** on graphs by integrating calculus of variations into neural network training. It learns geodesics or transport paths between probability distributions by minimizing variational residuals—without relying on ground-truth trajectories.

This repo contains our implementation for applying deep neural networks to graph-based dynamic optimal transport, where the geometry is defined by a **Markov chain**, and the goal is to dynamics of optimal transport on graph, or geodesics on Wasserstein spaces.

---

## 🌉 Key Features

- ✅ Learns continuous transport paths on discrete graph structures
- ✅ Variational losses derived from the dynamic OT formulation
- ✅ Automatically satisfies boundary constraints
- ✅ Applicable to discrete Wasserstein geodesics

---

## 📘 Method Overview

We consider the dynamic formulation of optimal transport over a graph with stationary distribution \( \pi \) and generator \( Q \), and learn a transport path \( \rho(t) \in \mathcal{M}_\pi \) that minimizes the transport energy:

\[
\mathcal{E}[\rho] = \int_0^1 \frac{1}{2} \sum_{i,j} Q_{ij} \left( \frac{(\rho_i - \rho_j)^2}{\pi_i} \right) \, dt
\]

Rather than solving this PDE directly, we propose parameterizes \( \rho(t) \) via a neural network and minimizes the residuals of continuity equation and geodesics equation.
