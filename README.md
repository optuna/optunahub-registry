# OptunaHub Registry

OptunaHub Registry is a registry service for sharing and discovering user-defined Optuna packages. It provides a platform for users to share their Optuna packages with others and discover useful packages created by other users.

See the [OptunaHub Website](https://hub.optuna.org/) for registered packages.

See also the [OptunaHub API documentation](https://optuna.github.io/optunahub/) for the API to use the registry, and the [OptunaHub tutorial](https://optuna.github.io/optunahub-registry/) for how to register and discover packages.

## TODO List towards Contribution

When creating your package, please check the following TODO list:

- Copy `./template/` to create your package
- Replace `<COPYRIGHT HOLDER>` in `LICENSE` of your package with your name
- Apply the formatter based on the tips below
- Check whether your module works as intended based on the  tips below
- Fill out `README.md`

> [!TIP]
> The following formatting is a requirement to merge this PR:
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
