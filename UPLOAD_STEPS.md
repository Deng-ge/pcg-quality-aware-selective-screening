# GitHub Upload Steps

## Standard Public Repository

Create an empty GitHub repository named:

```text
pcg-quality-aware-selective-screening
```

Then run:

```bash
cd pcg-quality-aware-selective-screening
git init
git add .
git commit -m "Initial release of quality-aware PCG screening code"
git branch -M main
git remote add origin https://github.com/<your-account>/pcg-quality-aware-selective-screening.git
git push -u origin main
```

## Anonymous Review Use

If the repository is used during anonymous peer review:

- use an anonymous or neutral GitHub account when required by the journal;
- do not upload the submitted manuscript PDF;
- do not add author names, affiliations, acknowledgements, or funding text;
- do not upload raw PCG recordings, trained checkpoints, or local logs;
- keep the repository description neutral, for example:

```text
Quality-aware selective prediction pipeline for phonocardiogram screening
```

## Before Uploading

Check the repository content:

```bash
git status --ignored
```

Files under `outputs/`, raw data folders, checkpoints, and manuscript artifacts
should remain ignored or outside the repository.
