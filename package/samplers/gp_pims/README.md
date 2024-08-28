---
author: Shion TAKENO
title: Gaussian-Process Probability of Improvement from Maximum of Sample Path Sampler
description: This sampler searches for each trial based on Probability of Improvement from Maximum of Sample Path using Gaussian process.
tags: [sampler, Gaussian process]
optuna_versions: [3.6.1]
license: MIT License
---

<!--
This is an example of the frontmatters.
All columns must be string.
You can omit quotes when value types are not ambiguous.
For tags, a package placed in
- package/samplers/ must include the tag "sampler"
- package/visualilzation/ must include the tag "visualization"
- package/pruners/ must include the tag "pruner"
respectively.

---
author: Optuna team
title: My Sampler
description: A description for My Sampler.
tags: [sampler, 2nd tag for My Sampler, 3rd tag for My Sampler]
optuna_versions: [3.6.1]
license: "MIT License"
---
-->

## Class or Function Names

- PIMSSampler

## Installation

```shell
$ pip install -r https://hub.optuna.org/samplers/gp_pims/requirements.txt
```

## Example

Please see example.py.

## Others

### Reference

Shion Takeno, Yu Inatsu, Masayuki Karasuyama, Ichiro Takeuchi,
Posterior Sampling-Based Bayesian Optimization with Tighter Bayesian Regret Bounds,
Proceedings of the 41st International Conference on Machine Learning, PMLR 235:47510-47534, 2024.

### Bibtex

```

@InProceedings{pmlr-v235-takeno24a,
  title = 	 {Posterior Sampling-Based {B}ayesian Optimization with Tighter {B}ayesian Regret Bounds},
  author =       {Takeno, Shion and Inatsu, Yu and Karasuyama, Masayuki and Takeuchi, Ichiro},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {47510--47534},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/takeno24a/takeno24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/takeno24a.html}
}

```
