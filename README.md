OptunaHub Registry
==================

![OptunaHub](https://github.com/user-attachments/assets/ee24b6eb-a431-4e02-ae52-c2538ffe01ee)

:link: [**OptunaHub**](https://hub.optuna.org/)
| :page_with_curl: [**Docs**](https://optuna.github.io/optunahubhub/)
| :page_with_curl: [**Tutorials**](https://optuna.github.io/optunahubhub-registry/)
| [**Optuna.org**](https://optuna.org/)

OptunaHub Registry is a registry service for sharing and discovering user-defined Optuna packages. It provides a platform for users to share their Optuna packages with others and discover useful packages created by other users.

See the [OptunaHub Website](https://hub.optuna.org/) for registered packages.

See also the [OptunaHub API documentation](https://optuna.github.io/optunahub/) for the API to use the registry, and the [OptunaHub tutorial](https://optuna.github.io/optunahub-registry/) for how to register and discover packages.

## Contribution

Any contributions to OptunaHub are more than welcome!

OptunaHub is composed of the following three related repositories. Please contribute to the appropriate repository for your purposes.
- [optunahub](https://github.com/optuna/optunahub)
    - The python library to use OptunaHub. If you find issues and/or bugs in the optunahub library, please report it here via [Github issues](https://github.com/optuna/optunahub/issues).
- [optunahub-registry](https://github.com/optuna/optunahub-registry/) (*this repository*)
    - The registry of the OptunaHub packages. If you are interested in registering your package with OptunaHub, please contribute to this repository. For general guidelines on how to contribute to the repository, take a look at [CONTRIBUTING.md](https://github.com/optuna/optunahub-registry/blob/main/CONTRIBUTING.md).
- [optunahub-web](https://github.com/optuna/optunahub-web/)
    - The web frontend for OptunaHub. If you find issues and/or bugs on the website, please report it here via [GitHub issues](https://github.com/optuna/optunahub-web/issues).

## Quick TODO List towards Contribution

When creating your package, please check the following TODO list:

- Copy `./template/` to create your package
- Replace `<COPYRIGHT HOLDER>` in `LICENSE` of your package with your name
- Apply the formatter based on the tips below
- Check whether your module works as intended based on the  tips below
- Fill out `README.md`

For more details, please check [OptunaHub tutorial](https://optuna.github.io/optunahub-registry/).

> [!TIP]
> The following formatting is a requirement to merge your feature PR:
>
> ```shell
> $ pip install pre-commit
> $ pre-commit run --all-files
> ```
>
> Please also try the following to make sure that your module can be loaded from the registry:
>
> ```python
> import optunahub
>
> module = optunahub.load_module(
>     # category is one of [pruners, samplers, visualization].
>     package="<category>/<your_package_name>",
>     repo_owner="<your_github_id>",
>     ref="<your_branch_name>",
> )
> ```
>
> For more detail, please check [the tutorial](https://optuna.github.io/optunahub-registry/recipes/005_debugging.html).
