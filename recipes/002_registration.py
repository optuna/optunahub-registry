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
|         ├── (example.py)
|         ├── (requirements.txt)
|         └── (images)
|             ├──  (thumbnail.png)
|             └──  (screenshot.png)

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
- `example.py`: This is optional. This file should contain a simple example of how to use your algorithm (Example: `example.py for Simulated Annealing Sampler <https://github.com/optuna/optunahub-registry/blob/main/package/samplers/simulated_annealing/example.py>`_).
- `requirements.txt`: This is optional. A file that contains the additional dependencies of your algorithm. If there are no additional dependencies other than Optuna and OptunaHub, you do not need to create this file.
- `images`: This is optional. A directory that contains images. The images in this directory will be used the `web page of OptunaHub <https://hub.optuna.org/>`_. `thumbnail.png` will be used as a thumbnail in the web page. Note that `README.md` can also refer to image files, e.g. `images/screenshot.png`,  in this directory.

All files must pass linter and formetter checks to be merged to the optunahub-registry repository.
You can check them by running the `pre-commit <https://pre-commit.com/>`__ tool as follows.

.. code-block:: bash

    pip install pre-commit
    pre-commit install
    pre-commit run  # This will run all checks against currently staged files.

Although we recommend you write proper type hints, if you find it difficult to comply with mypy, you can omit the check by writing the following directive as the first line of your code.

.. code-block:: python

  # mypy: ignore-errors


`README.md` must contain the following sections:

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

- `Class or Function Names` section that describes the classes or functions provided by the package. If you provide multiple classes or functions, you should list them in this section. Note that the section must be a markdown list. If you provide only one class or function, you can simply write the class or function name. Note that the documentation of the classes or functions must be written in their docstrings. If you want to refer to the documentation, please leave the source code link, or write them in the following `Others` section. For example:

  .. code-block:: markdown

      - `DemoSampler1`
      - `DemoSampler2`
      - `demo_function1`

- An `Installation` section that describes how to install the additional dependencies if required. For example:

  .. code-block:: markdown

      $ pip install -r requirements.txt

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
After merging your pull request, your package will be available on the `OptunaHub <https://hub.optuna.org/>`__ in about 1 hour.
"""
