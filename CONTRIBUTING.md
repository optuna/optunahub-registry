# Contribution Guidelines

If you are interested in contributing to Optuna Hub, please read the following guidelines.
If you are new to GitHub, please refer to [our blog](https://medium.com/optuna/optuna-wants-your-pull-request-ff619572302c) for more information.

## Tutorial

You can find tutorials to implement a package for the OptunaHub registry in [Tutorials for Contributors](https://optuna.github.io/optunahub/tutorials_for_contributors.html).

## Coding Standards and Guidelines

Please adhere to the following coding standards and guidelines:

- write your code comments and documentation in English
- give your package an appropriate name, the package name may be requested to be changed at the discretion of the maintainers

All files must pass linter and formetter checks to be merged to the optunahub-registry repository.
You can check them by running the [pre-commit](https://pre-commit.com/) tool as follows.

```bash
pip install pre-commit
pre-commit install
pre-commit run  # This will run all checks against currently staged files.
```

## Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

First, the **title** of your pull request should:

- briefly describe and reflect the changes
- wrap any code with backticks
- not end with a period

Second, the **description** of your pull request should:

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks

Finally, read [`contributor agreements`](#contributor-agreements) and if you agree, please click the checkbox

## Code Review Standards

All contributions must undergo code review and approval by at least one OptunaHub committer before being merged. Package development and long-term maintenance are primarily the responsibility of contributors. OptunaHub committers only ensure compliance with submission standards.

In general, contributions are expected to satisfy the following requirements:

1. **Minimal Working Example**\
   For new package registrations or updates, a minimal working example demonstrating basic functionality must be provided. This may be included in the pull request description, the package’s `README.md`, or a separate file such as `example.py`. Adding [unit tests](https://optuna.github.io/optunahub/generated/recipes/008_unit_tests.html) is strongly encouraged but not mandatory. Providing benchmark results is also welcome.

1. **Security and Safety**\
   Contributions must be free of malicious code. For security reasons, binary files (except images used in `README.md`) cannot be distributed directly through the `optunahub-registry`. If binary dependencies are required, they should be hosted separately (e.g., on PyPI) and declared appropriately.

1. **Engineering and Scientific Integrity**\
   Contributors are expected to adhere to sound engineering and scientific practices, including compliance with applicable software licenses. OptunaHub does not perform peer review of scientific validity; responsibility for the correctness and validity of methods remains with contributors. However, submissions that clearly violate research ethics, contain misleading claims, or demonstrate harmful or malicious intent may be declined or removed at the discretion of the OptunaHub committers. Community discussion via linked GitHub issues is encouraged to promote transparency and critical evaluation of contributed methods.

## Contributor Agreements

1. By making a contribution to this project, I certify that:
   (a) The contribution was created in whole or in part by me and I have the right to submit it under the MIT license indicated in the file:
   (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open-source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the MIT license, as indicated in the file; or
   (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

1. By making a contribution to this project, I agree that the MIT license applies to the contribution and the contribution is used by Preferred Networks (“PFN”) or third party under the MIT license.

1. I agree that PFN may remove my contribution from the Optuna Hub at any time, if the Contribution likely violates any of applicable law and regulation, or likely infringe or misappropriate any rights of any person or entity. I ALSO AGREE THAT PFN SHALL NOT BE RESPONSIBLE OR LIABLE FOR SUCH REMOVAL OF MY CONTRIBUTION.

1. I AGREE THAT PFN SHALL NOT BE RESPONSIBLE OR LIABLE FOR USE MODIFICATION OR REMOVAL OF MY CONTRIBUTION BY THIRD PARTY.
