# Dataset

## Source

The experiments are based on the public PhysioNet/CinC Challenge 2016 heart
sound dataset:

```text
https://physionet.org/content/challenge-2016/1.0.0/
```

The dataset contains phonocardiogram recordings from multiple training subsets
with normal and abnormal labels. Please follow the official PhysioNet page for
download instructions, citation requirements, and license terms.

## Why Raw Data Is Not Included

This repository does not redistribute the original `.wav`, `.hea`, or `.mat`
files. Keeping raw medical recordings out of the repository makes the code
package lightweight and avoids mixing dataset redistribution with the algorithm
implementation.

The included file `examples/synthetic_pcg.wav` is a synthetic signal generated
only for testing the command-line demo. It is not part of the PhysioNet dataset.

## Suggested Local Layout

After downloading the official dataset, keep it outside the Git repository and
point your scripts to the local dataset root:

```text
<dataset_root>/
|-- training-a/
|-- training-b/
|-- training-c/
|-- training-d/
|-- training-e/
`-- training-f/
```

Each subset should contain the original waveform and metadata files from the
official dataset release.

## Reproducibility Notes

- Do not commit raw recordings, patient-level files, or derived large feature
  caches to GitHub.
- Keep trained checkpoints in a local `checkpoints/` directory or a separate
  artifact store.
- Report the dataset version as PhysioNet/CinC Challenge 2016 v1.0.0 when
  reproducing experiments.
