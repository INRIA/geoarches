# Contribute to geoarches
## Setup

To contribute to `geoarches`, we recommend working from your own fork of the repository.

### 1. Fork and clone

First, [fork the official repo](https://github.com/INRIA/geoarches) to your own GitHub account.

Then clone your fork locally:

```sh
git clone git@github.com:<your_username>/geoarches.git
```

Next, add the original (upstream) repository as a remote, so you can pull updates from the main repo:

```sh
cd geoarches
git remote add upstream git@github.com:INRIA/geoarches.git
```

### 2. Install dependencies

Follow the [installation instructions](../getting_started/installation.md) to set up your environment and install required packages.

### 3. Work from a development branch

Before making changes, create a development branch off of `main`. For example:

```sh
git checkout main
git pull upstream main  # Just in case
git checkout -b dev_<your_name_or_feature>
```

You can now safely make changes on your fork. This avoids interfering with the main `geoarches` development, and keeps your work isolated until it's ready to contribute.

This setup also ensures you can:

- Contribute new code via pull requests (see [Code reviews](#code-reviews))
- Sync easily with updates from upstream (see [Pull code updates](#pull-code-updates))

## Local testing

All new code must be covered by tests under the `tests/` directory. To run tests:

```sh
pytest tests/
```

## Code formatting

We recommend reading [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for tips of writing readable code.

Before committing, run:

```sh
ruff check --fix
ruff format
codespell -w
```

### Optional: Pre-commit hooks

You can run these checks automatically before each commit using [`pre-commit`](https://pre-commit.com/). Run the following command in the root of your cloned repository to set it up:

```sh
pre-commit install
```

!!! success "Now, these tools will run automatically on `git commit`!"

## Code reviews

Once your code is ready and pushed to a branch, you can prepare it to be reviewed and merged into the main `geoarches` codebase.

You have two options:

- Use your current local branch directly (make sure it's up to date with `upstream/main`, see [Pull code updates](#pull-code-updates)),
- Or create a clean branch based on `upstream/main` and apply just your relevant commits on top of it.

To create a clean branch from `upstream/main`:

```sh
# Make sure your local copy of upstream is up to date
git fetch upstream main

# Create a new branch based on upstream/main
git checkout -b <feature_branch_name> -t upstream/main
```

Then use `git cherry-pick` to apply a specific range of commits (from commit A to B):

```sh
git cherry-pick A^..B
```

This allows you to transfer only the changes you want, without dragging along unrelated commit history.

---

To check that your code passes tests before submitting a PR, push the branch to **your fork** (not upstream yet):

```sh
git push origin <feature_branch_name>
```

Then open a **fake pull request on your own fork**. This triggers GitHub Actions and runs the test suite, just like it would on a real PR to geoarches.

---

Once you're confident everything works and tests pass, push the clean feature branch to the upstream repository:

```sh
git push upstream <feature_branch_name>
```

Then open a new [pull request](https://github.com/INRIA/geoarches/pulls) on GitHub which will target the `main` branch of geoarches. It will only be eligible to merge once:

- All CI tests pass
- You’ve received approval from a project maintainer

## Pull code updates

When the `main` branch of `geoarches` gets updated and you want to bring those changes into your branch, follow these steps. This is important for:

- Taking advantage of the latest code
- Avoiding merge conflicts before opening a PR

If you haven’t yet, set up a remote that tracks the upstream repo. You can check your remotes with:

```sh
git remote -v
```

If needed, add the upstream remote:

```sh
git remote add upstream git@github.com:INRIA/geoarches.git
```

### 1. Stash or commit your changes

If you have uncommitted work on your branch, stash it:

```sh
git stash push -m "WIP before syncing"
```

Or commit it normally.

### 2. Fetch upstream changes

Update your local references to the upstream repo:

```sh
git fetch upstream main
```

### 3. Checkout your development branch

Switch to the branch you want to sync (e.g. your dev branch):

```sh
git checkout dev_<your_name>
```

Then apply upstream changes. Use `rebase` if you're the only one working on this branch (recommended). Use `merge` if others are working on it too.

```sh
git rebase upstream/main
# or
git merge upstream/main
```

If there are conflicts, Git will pause and ask you to resolve them. Once resolved:

```sh
git rebase --continue
# or
git merge --continue
```

If you want to cancel everything and undo:

```sh
git rebase --abort
# or
git merge --abort
```

### 4. Apply stashed changes (if any)

If you stashed your work earlier, re-apply it now:

```sh
git stash pop
```

If there are conflicts during stash pop, resolve them. If you want to undo the stash application:

```sh
git reset --merge
```

This will cancel the stash apply, but keep the stash saved so you can re-apply it later if needed.
