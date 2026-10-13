# SteamWeave technical manual

[中文](TECHNICAL_MANUAL.zh-CN.md) · [Project overview](../README.md) · [Experiment protocol](EXPERIMENT_PROTOCOL.md)

## 1. Task and information boundary

SteamWeave predicts the degradation level at the next scheduled inspection from the information available at observation time (t). The target is the supplied `nextinslevel.txt`, measured in arbitrary degradation units (adu); time is measured in arbitrary time units (atu). It does not estimate a calendar failure date or remaining useful life.

$$
\hat y_{i,t}=f(x_{i,t}^{num},x_{i,t}^{img},x_{i,t}^{text},H_{i,\le t},T_{next}-t).
$$

`H` contains earlier input observations of the same device. Target labels, including historical target labels, are excluded from feature reconstruction and LLM requests. The dataset's target convention is retained; the software does not invent an inspection/repair ordering absent from the source files.

## 2. Repository and runtime

| Directory | Responsibility |
| --- | --- |
| `sg_agents/` | data contracts, causal reconstruction, experts, fusion, provider adapters and API |
| `static/` | localhost dashboard, history chart, uploads and API selection for explanations |
| `research/` | neural reconstruction, frozen experiment runner, pressure tests and public feature assets |
| `reports/` | experiment manifests, results, paired analysis and standalone figures |
| `config/` | provider-specific environment examples without credentials |
| `tests/` | information-boundary, missing-data, protocol and reconstruction tests |
| `Multimodal data for prognostics/` | original dataset, retained without rewriting |
| `artifacts/` | generated models, caches, logs and PID files; excluded from Git |

Tested environment: Python 3.14, macOS ARM64, scikit-learn histogram boosting and optional PyTorch MPS. The standard Python service also targets Linux; Windows users can use WSL or Docker. Native Windows locking is not implemented. Docker is supplied as a deployment recipe; the local machine's Docker runtime was unavailable for validation.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m sg_agents.cli train --offline
.venv/bin/python scripts/service.py start
```

The included semantic asset permits the default MiniMax feature pipeline to train offline. An API key is needed for new uncached reports and live explanations. Open `http://127.0.0.1:8787` or `/docs`. On macOS, `scripts/start.command` opens the workbench. The detached service survives closing the terminal, but does not automatically restart after a reboot unless the optional login-startup configuration is installed.

## 3. Data contract and audit

Each device folder contains time, text, target and four numerical vectors. Observation index `X` corresponds to image `Image/X.tiff`, using one-based indexing. Numeric/text lengths must agree. Times must be finite and strictly increasing. Counts and durations must be finite and nonnegative.

The local copy contains 7,626 records in 50 directories. Canonicalization trims outer text quotes; 794 raw report strings become 766 unique canonical strings. Devices 1–20 and 31–50 have paired identical time, numerical and text histories. Thus, evaluation uses 30 input-trajectory groups. Grouping excludes labels and does not treat TIFF metadata differences as independent simulation evidence. The audit concerns this copy, not an assertion about the original authors' experimental data version.

Group construction is offline split preparation. The group ID is not a predictive feature. All source records remain available; same-group inspection events are inversely weighted by their record count during leakage-controlled training. This prevents repeated directories and densely observed inspections from dominating the loss.

## 4. Time reconstruction and image processing

The next planned inspection and latest repair times are inferred from current numerical fields:

$$
T_{next}=t+\Delta t_{next},\qquad T_{repair}=t-\Delta t_{repair}.
$$

A forward update in inferred latest repair time signals an observed repair change. Cleaning-count increases and mechanical-count resets provide additional context. Multiple events between observations cannot necessarily be recovered; event interval bounds are kept rather than fabricated exact timestamps.

For text, the software stores first-seen time, current consecutive-repeat segment start, both observed ages, reuse flag, and whether a repair occurred after the repeat segment began. True report generation time remains unknown. Reappearing identical wording is not automatically assigned its earlier generation timestamp.

The curve encoder locates the plot rectangle, removes axes/tick regions, and searches for the trace. A file containing only axes is missing evidence, even though the TIFF exists. Valid traces yield 32 samples plus 8 shape statistics. Height is normalized to the common plot rectangle, preserving amplitude differences across records. These features are approximate image-derived geometry, not reconstructed original signal values. Unreadable files and undetected axes are explicitly unavailable.

## 5. Seven bounded agent roles

| Role | Input | Output |
| --- | --- | --- |
| Dispatcher | current request and same-device history | time-bounded context and availability masks |
| Numerical expert | four original numeric fields | supervised point estimate |
| Curve expert | curve geometry and horizon/repair duration | point estimate or unavailable state |
| Text expert | six semantic fields, 64 hashed lexical features, horizon/repair duration | point estimate and evidence provenance |
| Maintenance expert | numeric fields and 13 reconstructed state features | point estimate |
| Fusion agent | expert predictions, numeric fields and 12 quality/context fields | point prediction and interval |
| Audit/explanation agent | whitelisted prediction and evidence fields | deterministic quality flags and optional LLM explanation |

The semantic fields separate pre-maintenance severity, post-maintenance severity, mechanical cleaning, chemical cleaning, explicit urgency and unknown maintenance type. Severity is an ordinal language feature; it is not an adu conversion. JSON schema/domain checks and a source-substring check validate structure and grounding, not the correctness of every semantic label. Case-only quotation differences are mapped back to the exact source substring.

The workflows use explicit contracts and a fixed graph. The predictive experts are supervised regressors; the language model is used for extraction and explanation. A seven-role workflow does not require seven paid LLM calls per record.

## 6. Expert and fusion training

Histogram boosting uses 180 iterations, 15 leaf nodes, 25 minimum samples per leaf, learning rate 0.065 and L2 regularization 4. Internal random-row early stopping is disabled. Columns entirely unobserved in a fitting partition are removed by that partition's learned selector. Other NaNs remain available to native missing-value routing. No validation/test rows determine the selector.

Four-fold inner cross-validation generates out-of-fold expert estimates for fusion. Duplicate-aware trajectory groups are shared between outer and inner protocols. Expert predictions for unavailable current modalities are NaN. Fusion inputs also retain explicit quality flags; a blank curve cannot silently become a normal curve.

Fusion augmentation weights clean examples by 0.7 and each current-image/current-text/both-missing pattern by 0.1. The missing masks remove current expert evidence; earlier maintenance context may remain. The learned fusion is not an attention model, and its expert estimates are not causal contribution measures.

Production training uses seed 20261013 and a 21/3/6 trajectory-group fitting/calibration/test split. Research cross-validation and temporal experiments have separate outputs and do not overwrite the serving model.

## 7. Predictive intervals

Separate 5% and 95% quantile models operate on the fusion representation. On held-out calibration groups, the maximum interval violation within each target inspection event forms a calibration score. The selected score quantile expands both bounds. Interval crossing is corrected by ordering the two estimates.

Report row coverage, event coverage and interval width. Nominal 90% coverage is empirical here: events inside a trajectory are dependent, and future-period shifts can change behavior. Negative lower bounds are possible because bounds are not silently clipped to a physical range unspecified by the runtime contract.

## 8. API suppliers, credentials and cache provenance

| Provider | Key environment variable | Model variable | Endpoint variable |
| --- | --- | --- | --- |
| MiniMax | `MINIMAX_API_KEY` | `MINIMAX_MODEL` | `MINIMAX_BASE_URL` |
| OpenAI GPT | `OPENAI_API_KEY` | `OPENAI_MODEL` | `OPENAI_BASE_URL` |
| Qwen | `DASHSCOPE_API_KEY` | `QWEN_MODEL` | `QWEN_BASE_URL` |
| Custom compatible service | `CUSTOM_API_KEY` | `CUSTOM_MODEL` | `CUSTOM_BASE_URL` |

`LLM_PROVIDER` selects extraction. `LLM_AUDIT_PROVIDER`, or the explanation request's `provider` field, independently selects explanation. For example, retain MiniMax-derived model inputs and use GPT only for audit:

```dotenv
LLM_PROVIDER=minimax
LLM_AUDIT_PROVIDER=openai
OPENAI_API_KEY=replace_with_your_key
OPENAI_MODEL=your-gpt-model
```

No adapter falls back to another supplier's key. Keys are backend-only and omitted from JSON responses and Git. Model names are explicit; the adapter does not silently replace a requested model. Use endpoint/key combinations from the same Qwen region and billing plan.

`*_API_STYLE=chat` uses Chat Completions; `responses` parses final `output_text` blocks from Responses. OpenAI Chat defaults to `max_completion_tokens`; the other presets default to `max_tokens`. `*_TOKEN_PARAMETER` changes this when required. `*_EXTRA_BODY` accepts optional model parameters while forbidding overrides of credentials, model, input and output limits. Chat reasoning uses `reasoning_effort`; Responses reasoning uses a `reasoning` object. Only send fields supported by the selected model.

```bash
.venv/bin/python -m sg_agents.cli check-api --provider minimax
.venv/bin/python -m sg_agents.cli check-api --provider openai
.venv/bin/python -m sg_agents.cli train --provider qwen
```

Semantic caches are isolated by provider, model, endpoint and API style. The original M2.7 standard/high-speed aliases share one documented legacy namespace. Switching the extraction namespace requires rebuilding features and retraining. Switching only the audit API preserves the predictive model. The bundled default feature cache is not used as an OpenAI/Qwen cache.

MiniMax has been validated with real requests. GPT/Qwen/custom have transport-contract and error-path tests; without those suppliers' keys, no claim of live-account validation is made. Official protocol references: [OpenAI Chat](https://developers.openai.com/api/reference/resources/chat), [Aliyun endpoint overview](https://help.aliyun.com/zh/model-studio/base-url), [MiniMax model calls](https://platform.minimax.cn/docs/guides/text-generation).

## 9. Service API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health`, `/api/providers` | readiness and non-secret provider configuration |
| `GET /api/summary`, `/api/devices` | data audit and evaluation overview |
| `GET /api/devices/{id}/records` | observation selection |
| `GET /api/devices/{id}/history?through=t` | predictions for observations no later than `t` |
| `POST /api/predict` | known-record inference with optional current-modality masks |
| `POST /api/explain` | whitelisted evidence explanation with optional supplier |
| `POST /api/predict/custom` | multipart numeric/text/image upload |
| `GET /api/export/{id}` | device predictions as UTF-8 CSV |
| `POST /api/train`, `GET /api/train/status`, `POST /api/reload` | training lifecycle |
| `POST /api/provider/check?provider=name` | live API verification |

Custom predictions use strictly earlier same-device observations; a new device has empty history. Image uploads are limited to 10 MB and 20 million pixels. Finite nonnegative quantities and integer cleaning counts are enforced. Live explanation never receives the evaluation target appended by the display layer.

The tested service binds localhost. Cross-origin mutation requests are rejected. Concurrent cache writes use a file lock and atomic replacement. Pipeline tasks have a nonblocking lock. A public HTTP deployment needs its own identity and request-budget controls.

## 10. Research commands and comparison scope

```bash
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m research.glove
.venv/bin/python -m research.run_suite
.venv/bin/python -m research.extend_stress
.venv/bin/python -m research.summarize
```

The suite evaluates five grouped folds across three seeds, three future-period runs, and five diagnostic directory-ID folds. Six ablations and paired pressure tests are evaluated on all five grouped folds for seed 17. The reconstructed neural baseline uses one seed, not three. CPU users can set `--device cpu`; a different schedule should use a new `--run-name` so completed runs cannot be silently reused under changed settings.

The original paper reports its own three-branch model's MAE/MSE/MRE. Those values remain a literature reference with the source protocol identified. The local reconstruction and the current system are rerun on shared partitions. Differences in data realization, preprocessing, optimization and hardware prevent declaring a reproduction of the published result or a cross-paper superiority claim.

See [experiment results](../reports/EXPERIMENTS.md) for actual metrics, confidence intervals, ablations, pressure tests, raw predictions and training-budget limitations.

## 11. Verification and publication

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_deployment.py --with-minimax
.venv/bin/python scripts/check_publication.py --tracked
```

Unit tests protect causal reconstruction, label exclusion, missing-image detection, duplicate isolation, fully missing columns, provider isolation and reconstruction dimensions. The live service verifier exercises existing predictions, masks, history cutoff, upload, CSV and real explanations.

Publish original data, notices, code, fixed feature assets and experiment outputs. `.env`, local logs/PIDs, virtual environments, private generated caches and model binaries are excluded. The bundled semantic asset contains simulated-report evidence; it does not contain authentication material.

## 12. Limits and next investigations

The benchmark is synthetic and contains repeated input trajectories. Effective group count is thirty. LLM semantic labels are not human-validated ground truth. Some maintenance/report-age interactions have no natural examples in this copy; synthetic report-delay tests test sensitivity, not a proven operational distribution. The neural reconstruction has disclosed missing implementation details and a capped budget. No module is assumed beneficial before its ablation is examined.

Further work should pre-register development-set model selection, expand independent simulation seeds or industrial data, assess report-label quality, and test inference cost alongside forecasting error. Keep outer test groups for final assessment.
