---
author: Shogo Iwazaki
title: Mean Variance Analysis Scalarization Sampler)
description: This sampler searches for each trial based on the UCB  criterion of the scalarized mean-variance objective using Gaussian process.
tags: [sampler, Gaussian process, mean-variance analysis]
optuna_versions: [4.0.0]
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

- MeanVarianceAnalysisScalarizationSimulatorSampler

## Installation

```shell
$ pip install scipy
```

## Example

Please see example.ipynb

## Others

For example, you can add sections to introduce a corresponding paper.

### Reference

Iwazaki, Shogo, Yu Inatsu, and Ichiro Takeuchi. "Mean-variance analysis in Bayesian optimization under uncertainty." International Conference on Artificial Intelligence and Statistics. PMLR, 2021.

### Bibtex

```
@inproceedings{iwazaki2021mean,
  title={Mean-variance analysis in Bayesian optimization under uncertainty},
  author={Iwazaki, Shogo and Inatsu, Yu and Takeuchi, Ichiro},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={973--981},
  year={2021},
  organization={PMLR}
}
```
