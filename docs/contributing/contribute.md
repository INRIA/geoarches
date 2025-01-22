
# Contribute to geoarches
We welcome contributions to the codebase such as adding:

- Data
    - geospatial datasets beyond ERA5 and DCPP (both download and dataloaders)
    - formats beyond xarray (netcdf, zarr, etc) such as csv
    - other storage stypes such as cloud storage
- Modeling
    - model architecture backbones
    - training schemes beyond diffusion ddpm and flow-matching
- Visualization
    - better support for plotting metrics
    - tools for visualization of geospatial data

## Setup

We suggest you [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of the [geoarches repo](https://github.com/INRIA/geoarches).

Then clone your fork from GitHub:

```sh
git clone git@github.com:<your_username>/geoarches.git
```

Add the original remote repo to track changes:

```sh
git remote add upstream git@github.com:INRIA/geoarches.git
```

Follow [installation](../getting_started/installation.md) to install dependencies.

You can make changes on your forked repository. This way you are not blocked by development on geoarches. You will be able to contribute to the `main` branch (see [Code reviews](#code-reviews)) and incorporate updates from others (see [Pull code updates](#pull-code-updates)).

Create a `dev` branch from the `main` branch of your forked repo to start making changes.

```shell
cd geoarches
git checkout main
git checkout -b dev_<name>
```

## Local testing

Every piece of code will need a corresponding test file under `tests/`.

You can make sure tests still pass by running:

```sh
pytest tests/
```

## Code format

We recommend reading [Google Style Python Guide](https://google.github.io/styleguide/pyguide.html) for tips of writing readable code.

We also require you to run these commands before committing:
```sh
ruff check --fix
ruff format
codespell -w
```

### Optional: Automatically check format on commit

You can set up automatic checks on ready-to-commit code using `pre-commit`.

Run in the `geoarches/` repo:
```
pre-commit install
```

Now, pre-commit will run automatically on `git commit`!

## Code reviews

You've pushed your changes to a branch and your code is ready to be incorporated into geoarches.

You can either use your local branch as is (make sure it's synced with the upstream repo by following [Pull code updates](#pull-code-updates)), or make a clean branch from `upstream/main` and add just the necessary commits there.

```sh
# Update remote branches.
git fetch upstream main
# Make clean branch.
git checkout -b <feature> -t upstream/main
# Move commits between older commit A (included) and B (included).
git cherry-pick git cherry-pick A^..B
```

You can make sure all tests pass by pushing to your forked repo and making a fake pull request on your own forked repo. This should trigger test checks (implemented as Github Actions).
```sh
git push origin <feature>
```

When your code is ready, push the clean branch to the `upstream` repo.
```sh
git push upstream <feature>
```

Make a [pull request](https://github.com/INRIA/geoarches/pulls) on Github. You will only be able to merge with the `main` branch, once all tests pass and you receive approval from a maintainer.

## Pull code updates

When the `main` branch of geoarches gets updated, and you want to incorporate changes.
This is important for both:
- Allowing you to take advantage of new code.
- Preemptively resolving any merge conflicts before merge requests.

If you have not already done so, configure a remote that points to the upstream repository (you can check the current configured remote repos with `git remote -v`.)

```sh
git remote add upstream git@github.com:INRIA/geoarches.git
```

The following steps will help you pull the changes from main and then apply your changes on top. See Github's [Syncing a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) for explanations for each command.

1. Either commit or stash your changes on your dev branch:
    ```sh
    git stash push -m "message"
    ```

2. Update remote branches.
    ```sh
    git fetch upstream main
    ```

3. Checkout the branch you want to sync. And rebase or merge your changes. Prefer to rebase if you are the only user of this branch, otherwise merge.
    ```sh
    git checkout <branch>
    git rebase upstream/main # or git merge upstream/main
    ```

    Resolve merge conflicts if needed. You can always decide to abort to undo the rebase completely:
    ```sh
    git rebase –abort # or git merge --abort
    ```

5. If you ran `git stash` in step 1, you can now apply your stashed changes on top.
    ```sh
    git stash pop
    ```

    Resolve merge conflicts if needed. To undo applying the stash:
    ```sh
    git reset --merge
    ```
    This will discard stashed changes, but stash contents won't be lost and can be re-applied later.