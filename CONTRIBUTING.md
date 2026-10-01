# How to contribute

Welcome! Thanks for your interest in contributing to this project.

- We manage our day-to-day work on this internal [project management board](https://github.com/alan-turing-institute/ice-station-zebra-project-board).
- Our [list of open issues](https://github.com/alan-turing-institute/icenet-mp/issues) is on GitHub.
- Contact us by [email](mailto:SeaIce@turing.ac.uk).

## Submitting changes

- Please open a [GitHub Pull Request](https://github.com/alan-turing-institute/icenet-mp/pull/new/main) with a clear explanation of what you've done (read more about [pull requests](https://docs.github.com/en/pull-requests)).
- We use the [feature branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow), with each Pull Request squashed into a single commit on `main`.

## Coding conventions

- We use `pre-commit` to enforce code-style conventions.
- You can install `pre-commit` following the instructions [here](https://pre-commit.com/#install).
- Run `pre-commit install` inside your locally-checked-out repository to activate it.
- You can also run the style checks without installing `pre-commit` by running `uv run --group dev pre-commit run --all-files`

## Tests

We encourage the use of tests across the whole codebase.
Run the `pytest` tests with `uv run --group dev pytest`.
Run the `mypy` checks with `uv run --group dev mypy .`.
