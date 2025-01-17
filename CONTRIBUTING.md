
## Contribute to geoarches

You can make changes on your own `dev` branch(s). This way you are not blocked by development on the `main` branch, but can still contribute to the `main` branch if you want to and can still incorporate updates from other team members.

1. Create a `dev` branch from the `main` branch of geoarches to start making changes.
    ```sh
    cd geoarches
    git checkout main
    git checkout -b dev_<name>
    ```

2. Commit and push your changes. 
3. Make sure tests pass by running `pytest tests/`.
4. Format your code with `ruff check --fix` and `ruff format`.
5. To incorporate your changes into the `main` branch, make a merge request and wait for review.

## Pull code updates.

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