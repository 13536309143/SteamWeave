# Contributing to SteamWeave

Start from a reproducible issue: include the command, Python version, provider name, dataset fingerprint and the relevant error. Do not include API keys or raw authenticated HTTP headers.

Changes to features or model training should state the information available at the observation time, preserve grouped splitting, and add a focused regression test when behavior changes. Keep diagnostic device-ID results separate from leakage-controlled results.

Run `python -m pytest -q`. Install `requirements-research.txt` to exercise the reconstruction architecture tests. Full experiments are opt-in and use the configuration recorded in `reports/experiment_manifest.json`.

Report results even when a proposed method loses. Avoid tuning on the outer test groups. Preserve paired predictions, training histories and configuration changes so comparisons remain reviewable.
