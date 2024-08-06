"""
.. _registration:

How to Register Your Algorithm with OptunaHub
===========================================================

After implementing your own algorithm, you can register the algorithm as a package with OptunaHub.
To add your package to the `optunahub-registry <https://github.com/optuna/optunahub-registry>`__ repository, you need to create a pull request from your fork.
Your pull request must be aligned with `the contribution guidelines <https://github.com/optuna/optunahub-registry/blob/main/CONTRIBUTING.md>`__.

The following is an example of the directory structure of a package.
See the `template directory <https://github.com/optuna/optunahub-registry/tree/main/template>`__ for an example of the directory structure.

| `package <https://github.com/optuna/optunahub-registry/tree/main/package>`__
| └── category (e.g. samplers, pruners, and visualization)
|     └── YOUR_PACKAGE_NAME (you need to create this directory and its contents)
|         ├── YOUR_ALGORITHM_NAME.py
|         ├── __init__.py
|         ├── README.md
|         ├── LICENSE
|         ├── (example.py, example.ipynb)
|         ├── (requirements.txt)
|         └── (images)
|             ├──  (figure1.png)
|             └──  (numerical_results.png)

An implemented algorithm should be put in the corresponding directory, e.g., a sampler should be put in the `samplers` directory.
In the `samplers` directory, you should create a directory with a unique identifier.
This unique identifier is the name of your algorithm package, is used to load the package, and is unable to change once it is registered.
The package name must be a valid Python module name, preferably one that is easily searchable.
Abbreviations are not prohibited in package names, but their abuse should be avoided.

The created directory should include the following files:

- `YOUR_ALGORITHM_NAME.py`: The implementation of your algorithm.
- `__init__.py`: An initialization file. This file must implement your algorithm or import its implementation from another file, e.g., `YOUR_ALGORITHM_NAME.py`.
- `README.md`: A description of your algorithm. This file is used to create an `web page of OptunaHub <https://hub.optuna.org/>`_. Let me explain the format of the `README.md` file later.
- `LICENSE`: A license file. This file must contain the license of your algorithm. It should be the MIT license in the alpha version of OptunaHub.
- `example.py`, `example.ipynb`: This is optional. This file should contain a simple example of how to use your algorithm (Example: `example.py for Simulated Annealing Sampler <https://github.com/optuna/optunahub-registry/blob/main/package/samplers/simulated_annealing/example.py>`_). You can provide examples in both formats.
- `requirements.txt`: This is optional. A file that contains the additional dependencies of your algorithm. If there are no additional dependencies other than Optuna and OptunaHub, you do not need to create this file.
- `images`: This is optional. A directory that contains images. Only relative references to images in this directory are allowed in README.md, e.g., ``![Numrical Results](images/numerical_results.png)``, and absolute paths to images are not allowed. The first image that appears in README.md will be used as the thumbnail.

All files must pass linter and formetter checks to be merged to the optunahub-registry repository.
You can check them by running the `pre-commit <https://pre-commit.com/>`__ tool as follows.

.. code-block:: bash

    pip install pre-commit
    pre-commit install
    pre-commit run  # This will run all checks against currently staged files.

Although we recommend you write proper type hints, if you find it difficult to comply with mypy, you can omit the check by writing the following directive as the first line of your code.

.. code-block:: python

  # mypy: ignore-errors


`README.md <https://github.com/optuna/optunahub-registry/blob/main/template/README.md>`__ must contain the following sections:

- A header section written in the following format:

  .. code-block:: markdown

      ---
      author: Optuna team
      title: Demo Sampler
      description: Demo Sampler of OptunaHub
      tags: [sampler]
      optuna_versions: [3.6.1]
      license: MIT License
      ---

  - `author` (string): The author of the package. It can be your name or your organization name.
  - `title` (string): The package title. It should not be a class/function name but a human-readable name. For example, `Demo Sampler` is a good title, but `DemoSampler` is not.
  - `description` (string): A brief description of the package. It should be a one-sentence summary of the package.
  - `tags` (list[string]): The package tags. It should be a list of strings. The tags must include `sampler`, `visualization`, or `pruner` depending on the type of the package. You can add other tags as needed. For example, "['sampler', 'LLM']".
  - `optuna_versions` (list[string]): A list of Optuna versions that the package supports. It should be a list of strings. You can find your Optuna version with `python -c 'import optuna; print(optuna.__version__)'`.
  - `license` (string): The license of the package. It should be a string. For example, `MIT License`. The license must be `MIT License` in the current version of OptunaHub.

- An `Installation` section that describes how to install the additional dependencies if required. For example:

  .. code-block:: markdown

      $ pip install -r requirements.txt


- `APIs` section that describes the documentation for classes/functions provided by the package. We highly recommend you provide enough information for users to use your package. For example:

  .. code-block:: markdown

      ### GPSampler(*, seed=None, independent_sampler=None, n_startup_trials=10, deterministic_objective=False)

      A sampler class of a Gaussian process-based surrogate Bayesian optimization algorithm.

      #### Parameters
        - `seed (int | None)` – Random seed to initialize internal random number generator. Defaults to None (a seed is picked randomly).
        - `independent_sampler (BaseSampler | None)` – Sampler used for initial sampling (for the first n_startup_trials trials) and for conditional parameters. Defaults to None (a random sampler with the same seed is used).
        - `n_startup_trials (int)` – Number of initial trials. Defaults to 10.
        - `deterministic_objective (bool)` – Whether the objective function is deterministic or not. If True, the sampler will fix the noise variance of the surrogate model to the minimum value (slightly above 0 to ensure numerical stability). Defaults to False.

      ### load_study(*, study_name, storage, sampler=None, pruner=None)

      #### Parameters
        - `study_name (str | None)` – Study’s name. Each study has a unique name as an identifier. If None, checks whether the storage contains a single study, and if so loads that study. study_name is required if there are multiple studies in the storage.
        - `storage (str | storages.BaseStorage)` – Database URL such as sqlite:///example.db.
        - `sampler` ('samplers.BaseSampler' | None) – A sampler object that implements background algorithm for value suggestion.
        - `pruner (pruners.BasePruner | None)` – A pruner object that decides early stopping of unpromising trials.

      #### Return type
        Study

- An `Example` section that describes how to use the package. It should be a python code block. It should be a few lines of code snippets that show how to use the package. If you want to provide a full example, please create a separete file like `example.py` and refer to it. For example:

  .. code-block:: markdown

      ```python
      sampler = DemoSampler()
      study = optuna.create_study(sampler=sampler)
      ```
      See `example.py <path/to/example.py>` for more details.

- An `Others` section that describes supplementary information about the package such as the paper reference or the original source code link. For example:

  .. code-block:: markdown

      - [Original Paper](Link/to/the/original/paper)
      - [Source Code](Link/to/the/source/code)

It is highly recommended that you confirm your package works properly (cf. :doc:`005_debugging`) before making a pull request.

Before making a pull request, please ensure the code examples in README.md and example.py do not contain your local directory and/or your fork of the registry.
Code such as ``load_local_module("your_package", registry_root=”your_local_directory”)`` or ``load_module("your_package_name", repo_owner=”your_github_id”, ref=”your_working_branch”)`` should be ``load_module("your_package_name")``.

After merging your pull request, your package will be available on the `OptunaHub <https://hub.optuna.org/>`__ in about 1 hour.
"""
