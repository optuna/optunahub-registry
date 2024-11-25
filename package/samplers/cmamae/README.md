---
author: Bryon Tjanaka
title: CMA-MAE Sampler
description: This sampler searches for solutions using CMA-MAE, a quality diversity algorihm implemented in pyribs.
tags: [sampler, quality diversity, pyribs]
optuna_versions: [4.0.0]
license: MIT License
---

## Abstract

This package provides a sampler using CMA-MAE as implemented in pyribs.
[CMA-MAE](https://dl.acm.org/doi/abs/10.1145/3583131.3590389) is a quality
diversity algorithm that has demonstrated state-of-the-art performance in a
variety of domains. [Pyribs](https://pyribs.org) is a bare-bones Python library
for quality diversity optimization algorithms. For a primer on CMA-MAE, quality
diversity, and pyribs, we recommend referring to the series of
[pyribs tutorials](https://docs.pyribs.org/en/stable/tutorials.html).

For simplicity, this implementation provides a default instantiation of CMA-MAE
with a
[GridArchive](https://docs.pyribs.org/en/stable/api/ribs.archives.GridArchive.html)
and
[EvolutionStrategyEmitter](https://docs.pyribs.org/en/stable/api/ribs.emitters.EvolutionStrategyEmitter.html)
with improvement ranking, all wrapped up in a
[Scheduler](https://docs.pyribs.org/en/stable/api/ribs.schedulers.Scheduler.html).
However, it is possible to implement many variations of CMA-MAE and other
quality diversity algorithms using pyribs.

## Class or Function Names

- CmaMaeSampler

Please take a look at:

- [GridArchive](https://docs.pyribs.org/en/stable/api/ribs.archives.GridArchive.html), and
- [EvolutionStrategyEmitter](https://docs.pyribs.org/en/stable/api/ribs.emitters.EvolutionStrategyEmitter.html)
  for the details of each argument.

## Installation

```shell
$ pip install ribs
```

## Example

```python
import optuna
import optunahub

module = optunahub.load_module("samplers/cmamae")
CmaMaeSampler = module.CmaMaeSampler


def objective(trial: optuna.trial.Trial) -> float:
    """Returns an objective followed by two measures."""
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    trial.set_user_attr("m0", 2 * x)
    trial.set_user_attr("m1", x + y)
    return x**2 + y**2


if __name__ == "__main__":
    sampler = CmaMaeSampler(
        param_names=["x", "y"],
        measure_names=["m0", "m1"],
        archive_dims=[20, 20],
        archive_ranges=[(-1, 1), (-1, 1)],
        archive_learning_rate=0.1,
        archive_threshold_min=-10,
        n_emitters=1,
        emitter_x0={
            "x": 0,
            "y": 0,
        },
        emitter_sigma0=0.1,
        emitter_batch_size=20,
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10000)
```

## Others

### Reference

#### CMA-MAE

Matthew Fontaine and Stefanos Nikolaidis. 2023. Covariance Matrix Adaptation
MAP-Annealing. In Proceedings of the Genetic and Evolutionary Computation
Conference (GECCO '23). Association for Computing Machinery, New York, NY, USA,
456–465. https://doi.org/10.1145/3583131.3590389

#### Pyribs

Bryon Tjanaka, Matthew C Fontaine, David H Lee, Yulun Zhang, Nivedit Reddy
Balam, Nathaniel Dennler, Sujay S Garlanka, Nikitas Dimitri Klapsis, and
Stefanos Nikolaidis. 2023. Pyribs: A Bare-Bones Python Library for Quality
Diversity Optimization. In Proceedings of the Genetic and Evolutionary
Computation Conference (GECCO '23). Association for Computing Machinery, New
York, NY, USA, 220–229. https://doi.org/10.1145/3583131.3590374

### Bibtex

#### CMA-MAE

```
@inproceedings{10.1145/3583131.3590389,
  author = {Fontaine, Matthew and Nikolaidis, Stefanos},
  title = {Covariance Matrix Adaptation MAP-Annealing},
  year = {2023},
  isbn = {9798400701191},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3583131.3590389},
  doi = {10.1145/3583131.3590389},
  abstract = {Single-objective optimization algorithms search for the single highest-quality solution with respect to an objective. Quality diversity (QD) optimization algorithms, such as Covariance Matrix Adaptation MAP-Elites (CMA-ME), search for a collection of solutions that are both high-quality with respect to an objective and diverse with respect to specified measure functions. However, CMA-ME suffers from three major limitations highlighted by the QD community: prematurely abandoning the objective in favor of exploration, struggling to explore flat objectives, and having poor performance for low-resolution archives. We propose a new quality diversity algorithm, Covariance Matrix Adaptation MAP-Annealing (CMA-MAE), that addresses all three limitations. We provide theoretical justifications for the new algorithm with respect to each limitation. Our theory informs our experiments, which support the theory and show that CMA-MAE achieves state-of-the-art performance and robustness.},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  pages = {456–465},
  numpages = {10},
  location = {Lisbon, Portugal},
  series = {GECCO '23}
}
```

#### Pyribs

```
@inproceedings{10.1145/3583131.3590374,
  author = {Tjanaka, Bryon and Fontaine, Matthew C and Lee, David H and Zhang, Yulun and Balam, Nivedit Reddy and Dennler, Nathaniel and Garlanka, Sujay S and Klapsis, Nikitas Dimitri and Nikolaidis, Stefanos},
  title = {pyribs: A Bare-Bones Python Library for Quality Diversity Optimization},
  year = {2023},
  isbn = {9798400701191},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3583131.3590374},
  doi = {10.1145/3583131.3590374},
  abstract = {Recent years have seen a rise in the popularity of quality diversity (QD) optimization, a branch of optimization that seeks to find a collection of diverse, high-performing solutions to a given problem. To grow further, we believe the QD community faces two challenges: developing a framework to represent the field's growing array of algorithms, and implementing that framework in software that supports a range of researchers and practitioners. To address these challenges, we have developed pyribs, a library built on a highly modular conceptual QD framework. By replacing components in the conceptual framework, and hence in pyribs, users can compose algorithms from across the QD literature; equally important, they can identify unexplored algorithm variations. Furthermore, pyribs makes this framework simple, flexible, and accessible, with a user-friendly API supported by extensive documentation and tutorials. This paper overviews the creation of pyribs, focusing on the conceptual framework that it implements and the design principles that have guided the library's development. Pyribs is available at https://pyribs.org},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  pages = {220–229},
  numpages = {10},
  keywords = {software library, framework, quality diversity},
  location = {Lisbon, Portugal},
  series = {GECCO '23}
}
```
