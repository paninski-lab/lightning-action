# Contributing

We welcome community contributions to the Lightning Action repo! 
If you have found a bug or would like to request a minor change, please 
[open an issue](https://github.com/paninski-lab/lightning-action/issues).

In order to contribute code to the repo, please follow the steps below.

We strive to maintain a fun and inclusive environment for our users and contributors.
See our [code of conduct](CODE_OF_CONDUCT.md) for more information.

### Set up a development installation

In order to make changes to Lightning Action, you will need to [fork](https://guides.github.com/activities/forking/#fork) the
[repo](https://github.com/paninski-lab/lightning-action).

If you are not familiar with `git`, check out [this guide](https://guides.github.com/introduction/git-handbook/#basic-git).

Whenever you initially install the lightning-action repo, instead of
```bash
pip install -e .
```
run
```bash
pip install -e ".[dev]"
```
to install additional development tools.

### Create a pull request
After making changes in your fork, 
[open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) 
from your fork. Please read through the rest of this document before submitting the request.

#### Linting
We use [ruff](https://docs.astral.sh/ruff/) for formatting, import sorting, and linting.
Run both commands before submitting a pull request:

```bash
ruff check --fix lightning_action tests
```

To check without modifying files:

```bash
ruff check lightning_action tests
```

If you set up the pre-commit hook (see below), ruff runs automatically on every commit.

#### Pre-commit hook
Install the pre-commit hook so ruff runs automatically before each commit:

```bash
pre-commit install
```

To run it manually against all files:

```bash
pre-commit run --all-files
```

#### Testing
We run tests automatically via GitHub Actions on every push and pull request.
The CI suite runs on CPU-only machines; tests that require a GPU are omitted from CI
(see `pyproject.toml` coverage omit list).

Run the full test suite locally before submitting your pull request:
```bash
pytest
```

Run with coverage to check for regressions:
```bash
pytest --cov=lightning_action --cov-branch
```

To browse coverage results interactively, generate an HTML report:
```bash
pytest --cov=lightning_action --cov-branch --cov-report=html
open htmlcov/index.html
```
