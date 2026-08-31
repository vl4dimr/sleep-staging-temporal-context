# Temporal context, not encoder capacity, drives automatic sleep staging

Code, fold definitions and per-fold results for the manuscript:

> Temporal context, not encoder capacity, drives automatic sleep staging:
> a subject-disjoint evaluation of compact models on Sleep-EDF-78.

Milton Vladimir Mamani Calisaya, Universidad Nacional del Altiplano, Puno,
Peru. ORCID: 0000-0002-0676-0989.

## Headline results

Five-fold subject-level cross-validation over all 78 subjects of
Sleep-EDF-78 (195,479 thirty-second epochs, each scored exactly once):

| Model   | Parameters | Accuracy        | Cohen's kappa   |
|---------|-----------:|-----------------|-----------------|
| full    |    362,085 | 0.800 +/- 0.023 | 0.733 +/- 0.029 |
| compact |     76,085 | 0.789 +/- 0.027 | 0.720 +/- 0.033 |

## What is in this deposit

- `folds.json` - explicit person-to-fold assignment. Both nights of a
  subject always share a fold; this file lets you audit the partitions
  without re-running GroupKFold.
- `code/models/` - the per-epoch encoder and the sequence model.
- `code/scripts/` - preprocessing (with sleep-period cropping and EDF
  integrity verification), local training/evaluation, benchmarks, and the
  scripts that regenerate every figure and table of the manuscript from
  the JSON results.
- `colab/SleepStaging_CV.ipynb` - the full GPU experiment battery
  (cross-validation, encoder/sequence variants, ablation).
- `results/` - per-fold metrics, confusion matrices and training histories
  for every run reported in the manuscript, plus measured efficiency.

## Data

The Sleep-EDF Expanded database is not redistributed here. Download it from
PhysioNet (https://physionet.org/content/sleep-edfx/1.0.0/); the
preprocessing script verifies every EDF file against the record count
declared in its own header before use - in our copy, 42 of 160 PSG files
were silently truncated downloads until re-fetched.

## Reproducing

1. `pip install -r code/requirements.txt`
2. Download Sleep-EDF Cassette into `data/raw/`.
3. `python code/scripts/preprocess_v2.py`
4. `python code/scripts/package_for_colab.py`
5. Run `colab/SleepStaging_CV.ipynb` on a GPU (or
   `code/scripts/cv_reduced_local.py` on CPU for the compact model).
6. `python code/scripts/report_from_colab.py --dir <results>` regenerates
   tables and figures; `build_manuscript.py` regenerates the manuscript.
   Every number in the paper is read from these JSON files at build time.

## License

Code: MIT. Results and documentation: CC-BY-4.0.
