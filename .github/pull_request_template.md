## Contributor Agreements

Please read the [contributor agreements](https://github.com/optuna/optunahub-registry/blob/main/CONTRIBUTING.md#contributor-agreements) and if you agree, please click the checkbox below.

- [ ] I agree to the contributor agreements.

> [!IMPORTANT]
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

## Motivation

<!-- Describe your motivation why you will submit this PR. This is useful for reviewers to understand the context of PR. -->

## Description of the changes

<!-- Describe the changes in this PR. -->
