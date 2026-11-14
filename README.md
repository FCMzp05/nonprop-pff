# Load-order-dependent phase-field model for fatigue crack paths

> **Status: submitted.** This repository is a partial release that contains the core components of the method. The complete code, including the phase-field fatigue solver, the case drivers, the meshes and the scripts that produce the figures, will be made publicly available here upon acceptance of the manuscript.

Manuscript:

> P. Zhang, K. Tang. *A load-order-dependent phase-field model for fatigue crack paths under non-proportional loading*. Submitted to Aerospace Science and Technology.

The model evaluates the load cycle on material planes. On each plane the shear stress increment is accumulated while the plane is open, which gives the signed area enclosed by the traction loop. The plane-wise values form a traceless tensor that changes sign with the traversal direction of the loop and vanishes for proportional and retraced loading. A matrix exponential of this tensor gives the structural tensor of an anisotropic crack-surface energy in the damage equation.

## Contents of this partial release

| File | Purpose |
|---|---|
| `oddtensor.py` | Plane traction, opening-gated shear accumulation, traceless odd tensor, structural tensor, damage stiffness with the anisotropic surface term |

## Requirements

Python 3.10 or newer, with `torch` and `numpy`.

```
pip install -r requirements.txt
```

## Usage

The inputs are the undegraded elastic strains `[eps_xx, eps_yy, gamma_xy]` of every element at the load states of one cycle, the Lame constants and the number of material planes.

```python
import oddtensor as ot

O, E = ot.accumulate(eps_cycle, lam, mu, 36)
T, Emax = ot.odd_tensor(O, E)
A, S = ot.structural_tensor(T, Emax, kappa, lmax)
```

`A` enters `damage_stiffness` in place of the identity tensor. Material and numerical parameters are listed in the paper.

## Data

The experimental crack paths and specimen geometries of Wolf et al. and Fremy et al. are taken from the data set of Hengeveld et al. (Zenodo, record 15807377).

## License

MIT, see `LICENSE`.

## Contact

Keke Tang (kktang@tongji.edu.cn)
