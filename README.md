# SW-KAN: Kolmogorov–Arnold Networks with Stieltjes–Wigert $q$-Orthogonal Polynomials

SW-KAN is a Kolmogorov–Arnold Network (KAN) variant that replaces B-spline edge activations with **Stieltjes–Wigert $q$-orthogonal polynomials** — a polynomial family orthogonal with respect to a log-normal weight on the semi-infinite domain $(0,\infty)$. A smooth, invertible **exponential-of-tanh** mapping bridges unbounded real-valued layer inputs to this domain, and a numerically stable **orthonormal three-term recurrence** evaluates the polynomial basis in $\mathcal{O}(N)$ operations per edge without any special-function calls.

This repository contains the reference PyTorch implementation used to produce the results reported in the accompanying paper, *"SW-KAN: Kolmogorov–Arnold Networks with Stieltjes–Wigert q-Orthogonal Polynomials."*

---

## Key Ideas

- **SW-KAN edge activations** — every edge in a KAN layer is parameterized by a learnable Stieltjes–Wigert polynomial expansion of degree $N$, with a per-edge (or per-layer) learnable shape parameter $q \in (0,1)$.
- **Exponential-of-tanh domain mapping** — $\phi(x) = \exp(\gamma \tanh(x/\iota))$ bijectively maps $\mathbb{R}$ onto a bounded, strictly positive interval $(e^{-\gamma}, e^{\gamma}) \subset (0,\infty)$, avoiding the saturation/gradient-vanishing issues of naively clipping unbounded inputs into $[-1,1]$-supported polynomial families.
- **$\mathcal{O}(N)$ recurrence-based evaluation** — the three-term recurrence (Koekoek–Lesky–Swarttouw §3.27) is evaluated in its numerically stable **orthonormal** form to avoid the `inf`/`NaN` blow-up of the raw monic recurrence.
- **Parameter efficiency** — SW-KAN matches or beats 18 polynomial-KAN baselines (Fermat, Chebyshev-family, Gottlieb, Vieta–Pell, Askey–Wilson, etc.) on MNIST at a comparable or smaller parameter budget, and generalizes to Fashion-MNIST (under PCA + limited data) and continuous 2D function approximation.

---

## Repository Contents

| File | Description |
|---|---|
| `swkanmnist32.ipynb` | Clean, self-contained training script for the **SW-KAN (medium)** MNIST classifier — the model reported in the paper's main results table. Defines `stieltjes_wigert_polynomials`, `StieltjesWigertKANLayer`, and `StieltjesWigertKAN_MNIST`, and trains/evaluates on standard MNIST with `AdamW` + cosine annealing. |
| `swkan.ipynb` | Exploratory research notebook (Kaggle-style) containing: alternative/experimental polynomial-basis formulations, Stieltjes–Wigert polynomial visualizations for different $q$ values, the continuous 2D function-approximation experiments, and the Fashion-MNIST (PCA-reduced, resource-constrained) benchmark used elsewhere in the paper. |

> Note: `swkan.ipynb` is a working/experimentation notebook and contains several superseded iterations of the polynomial-evaluation function as the implementation was refined toward the final, numerically stable version used in `swkanmnist32.ipynb` and described in the paper.

---

## Method Summary

For a KAN layer mapping $\mathbf{x}^{(\ell)} \in \mathbb{R}^{n_\ell} \to \mathbf{x}^{(\ell+1)} \in \mathbb{R}^{n_{\ell+1}}$, each edge activation is

$$
\varphi_{j,i}^{(\ell)}(x) = \sum_{n=0}^{N} c_{j,i,n}^{(\ell)}\, S_n\!\big(\phi(x);\, q_{j,i}^{(\ell)}\big)
\varphi_{j,i}^{(\ell)}(x) = \sum_{n=0}^{N} c_{j,i,n}^{(\ell)}\, S_n\!\big(\phi(x);\, q_{j,i}^{(\ell)}\big)
$$

where:
- $S_n(\cdot; q)$ are Stieltjes–Wigert polynomials (three-term recurrence, evaluated in stable orthonormal form),
- $\phi(x) = \exp(\gamma \tanh(x/\iota))$ maps $\mathbb{R} \to (0,\infty)$ (paper uses $\gamma=2$, $\iota=1$),
- $c_{j,i,n}^{(\ell)}$ are learnable expansion coefficients,
- $q_{j,i}^{(\ell)} = \sigma(\hat q_{j,i}^{(\ell)}) \in (0,1)$ is a learnable shape parameter via sigmoid reparameterization.

For a layer with $n_\ell$ inputs, $n_{\ell+1}$ outputs, and max degree $N$, the parameter count is $n_\ell n_{\ell+1}(N+2)$ (coefficients + per-edge $q$). In the code, `q_param` is implemented per-layer (a single scalar) rather than fully per-edge, for a lighter/faster variant of the layer.

---

## Installation

```bash
pip install torch torchvision
```

The MNIST notebook downloads the dataset automatically via `torchvision.datasets.MNIST` on first run; Fashion-MNIST cells in `swkan.ipynb` download directly from the official Fashion-MNIST S3 bucket.

---

## Usage

### Train SW-KAN (medium) on MNIST

```bash
jupyter nbconvert --to script swkanmnist32.ipynb --stdout > train_sw_kan_mnist.py
python3 train_sw_kan_mnist.py
```

or run `swkanmnist32.ipynb` directly. Key hyperparameters (edit in the notebook):

```python
model = StieltjesWigertKAN_MNIST(
    input_dim=784, hidden_dim=32, num_classes=10, degree=3
)
EPOCHS = 50
BATCH_SIZE = 64
```

The trained model is saved to `sw_kan_mnist_exact_fastio.pth`.

### Explore polynomial behavior / function approximation / Fashion-MNIST

Open `swkan.ipynb` and run the relevant section — cells are organized (in order of appearance) as: polynomial-basis prototyping → Stieltjes–Wigert visualization for various $q$ → 1D/2D function-approximation experiments → Fashion-MNIST benchmark.

---

## Results

### MNIST (784 → hidden → hidden → 10)

| Model | Accuracy | # Params | Time (s) |
|---|---|---|---|
| Fermat-KAN | 96.19% | 105,856 | 757.2 |
| Boubaker-KAN | 97.31% | 105,856 | 807.4 |
| Gottlieb-KAN | 97.59% | 219,907 | 924.0 |
| Vieta–Pell-KAN | 97.49% | 105,856 | 792.5 |
| Askey–Wilson-KAN | 96.93% | 105,871 | 2,418.0 |
| **SW-KAN (medium)** | **97.53%** | **105,859** | **357.0** (28 epochs) |
| **SW-KAN (large)** | **98.24%** | **219,907** | **221.5** (8 epochs) |

Both SW-KAN configurations outperform all 18 established polynomial-KAN baselines at their respective parameter budgets (full comparison in the paper).

### Fashion-MNIST (PCA-reduced to 40 dims, 10k training samples)

| Model | Accuracy | # Params |
|---|---|---|
| B-spline KAN | 82.65% | 21,658 |
| **SW-KAN** | **83.10%** | **14,634** |

### 2D Function Approximation

A 16-neuron, degree-4 SW-KAN (389 parameters) fit three qualitatively different 2D surfaces with test MSE of $2.5\times10^{-5}$–$2.9\times10^{-5}$ (smooth targets) and $3.5\times10^{-3}$ (oscillatory target).
