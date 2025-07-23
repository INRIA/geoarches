# Release Process

## How to create a release

1. Make sure you're on main and up to date:
   ```bash
   git checkout main
   git pull
   ```

2. Create and push a tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. GitHub automatically creates a release with notes

## Version numbers

We use semantic versioning: `vMAJOR.MINOR.PATCH`

* **PATCH**: Bug fixes (v0.1.0 → v0.1.1)
* **MINOR**: New features (v0.1.0 → v0.2.0)  
* **MAJOR**: Breaking changes (v0.1.0 → v1.0.0)

## Managing releases

### View and edit releases
* Go to: https://github.com/INRIA/geoarches/releases
* Click the ✏️ Edit button on any release to modify it

### Delete a release if needed
1. Delete the release on GitHub (from the releases page)
2. Delete the tag locally and remotely:
   ```bash
   git push --delete origin v0.1.0
   git tag -d v0.1.0
   ```