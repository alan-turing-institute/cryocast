# Create a release

CryoCast uses [CalVer](https://calver.org/) of the format `YYYY.MM` (e.g. `2026.10`) for version numbers. Releases are made at least once per milestone, or sooner if there have been significant changes or improvements to the model.

## 1. Prepare the release

Create a new branch containing the changes to be included in the release. Increment the version in `pyproject.toml` and any other metadata that differs for the new release.

Open a pull request and merge it to `main`.

## 2. Publish the release on GitHub

On the main repo page, go to **Releases** -> **Draft a new release**.

1. Click **Choose a tag** and create a new tag named the version number.
2. Choose the commit to target the release at (the one you just merged to `main`).
3. Fill in the descriptive fields and publish the release.

## 3. Run the model comparison suite

Once the release has been published, train the default set of models on HPC, using the tagged release version.

This should use the full dataset, with separate runs for the northern hemisphere and the southern hemisphere, and all other defaults. The models tested should be the basic model suite, i.e.:

- persistence (or similar climatology baseline)
- the best performing models at release time

For each training run, set the W&B run `name` to the version and the model type (e.g. `2026-10-persistence`) via the local config (or command line):

```yaml
reporting:
  loggers:
    wandb:
      name: <name>
```

The baseline configs have been tuned to run as either:
- a single-stage job completing inside a single 24h GPU run on JASMIN
- a multistage job completing inside two linked 24h GPU runs on JASMIN

Evaluate each trained model, again using the version-based name for the evaluation run, and upload summary evaluation plots/metrics to GitHub as release attachments.

Finally, copy the trained/evaluated run folders to Sharepoint as a backup.
