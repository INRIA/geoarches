
# Contribute to geoarches
You can make changes on your own `dev` branch(s). This way you are not blocked by development on the `main` branch, but can still contribute to the `main` branch and can still incorporate updates from other team members.

## Setup

Create a `dev` branch from the `main` branch of geoarches to start making changes.

```shell
cd geoarches
git checkout main
git checkout -b dev_<name>
```

## Local testing

Every piece of code will need a corresponding test file under `tests/`.

Make sure tests pass by running:

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

## Code reviews

When your code is ready, please make a pull request on Github. You will only be able to merge with the `main` branch, once you receive approval from a team member.

## Pull code updates

When the `main` branch of geoarches gets updated, and you want to incorporate changes.
This is important for both:
- Allowing you to take advantage of new code.
- Preemptively resolving any merge conflicts before merge requests.

The following steps will help you pull the changes from main and then apply your changes on top.
1. Either commit or stash your changes on your dev branch:
    ```sh
    git stash push -m "message"
    ```

2. Pull new changes into local main branch:
    ```sh
    git checkout main
    git pull origin main
    ```

3. Rebase your changes on top of new commits from main branch:
    ```sh
    git checkout dev_<name>
    git rebase main
    ```

    Resolve merge conflicts if needed. You can always decide to abort to undo the rebase completely:
    ```sh
    git rebase –abort
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