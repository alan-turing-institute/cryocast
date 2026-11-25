# CryoCast notebooks

These notebooks are supplementary teaching examples and research artifacts, not
disposable generated files. Their explanatory text, scientific code, and cell
order must be preserved when performing notebook maintenance.

See the [notebook guide](https://alan-turing-institute.github.io/cryocast/user-guide/notebooks/)
for prerequisites, execution options, and the current CI coverage.

The full [`demo_pipeline.ipynb`](demo_pipeline.ipynb) walkthrough uses the
`demo_notebook` configuration, requiring CDS data access and Weights & Biases.
Do not mistake it for an account-free example or remove its explanations to
make it run on CI. The separate `cryocast/config/synthetic.yaml` configuration
supports an account-free CLI workflow.

## Saving changes safely

From the repository root, after working interactively:

```bash
python scripts/notebook_hygiene.py --clean
python scripts/notebook_hygiene.py
```

This removes only generated cell outputs, execution counts, and timing metadata.
It **does not remove or rewrite code cells, Markdown, figures referenced by
Markdown, notebook metadata, or research instructions**. Empty cells are reported
rather than silently deleted. Already-clean files are left byte-for-byte unchanged,
including their original JSON layout and line endings.

CI checks the structure and cleanliness of all notebooks, and executes the
self-contained `metrics_examples.ipynb` without network credentials. Notebooks
that require CDS, third-party datasets or trained checkpoints still require
their documented external prerequisites and are not claimed to execute in CI.

The metrics execution job uses the inline plotting backend and checks that every
explicit `plt.show()` cell produces a PNG figure. It also rejects skipped cells,
error outputs, and changes to cell contents or order in the executed copy.
