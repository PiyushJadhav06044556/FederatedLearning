## Secure Federated Finance ML Platform

This repo contains a hub-and-spoke Federated Learning prototype (server + N dockerized clients) for finance datasets (e.g., credit card fraud), with:

- Binary logistic regression for centralized and local client training (no external ML libs; pure Python stdlib for math and CSV I/O)
- Plain FedAvg orchestration over HTTP (Flask)
- Secure Aggregation (Bonawitz-style) building blocks with X25519 ECDH, HKDF, and HMAC-based PRG, plus unit tests and a simple dropout recovery simulation
- Docker Compose setup to run server and 3 clients; scale to more clients as needed
- Notebooks for centralized baseline and FL demos

Important constraints satisfied:
- Logistic regression is implemented without external ML libraries
- Data artifacts are stored as CSV only (no NPY files)

### Quickstart

1) Preprocess CSVs and train centralized baseline

```
python3 -m scripts.preprocess_data --input data/creditcard.csv --clients 3 --out-root data --seed 42 --test-size 0.2
python3 -m scripts.train_centralized --train data/processed/train.csv --test data/processed/test.csv --label Class --epochs 2 --lr 0.1 --batch-size 512
```

2) Run the federated demo (FedAvg plain) with Docker (lightweight image)

```
./run_demo.sh
```

This launches the Flask server and 3 clients. The Docker image is minimal:
- We install only runtime deps and do not COPY source into the image; code is bind-mounted at runtime.
- Clients reuse the same built image (`federated-runtime:py311-slim`) to avoid duplicate builds.
- `.dockerignore` trims build context size.

3) Enable Secure Aggregation (prototype)

Set `SECAGG=1` for the clients in `docker/docker-compose.yml` to switch clients to masked uploads via Bonawitz-style masking and integer encoding. The server aggregates masked integers modulo q and decodes the aggregate.

4) Per-round logging and convergence plots

- The server writes per-round entries to `server/state/round_log.csv`. If you set `EVAL_TEST_CSV=data/processed/test.csv` in the server environment, it also appends `server/state/metrics_log.csv` with AUC/Precision/Recall/F1 each round.
- After a run, generate a simple convergence plot (AUC/F1 vs round):

```
python3 -m scripts.plot_convergence
```

Plot is saved to `outputs/plots/convergence.png`.

5) Performance and accuracy comparison

- Server performance (per-round): `server/state/perf_log.csv` includes duration and CPU time. Client-side metrics (per-round): `server/state/client_perf.csv` includes local training time, CPU time, and upload bytes by client.
- Create performance plots and a summary table:

```
python3 -m scripts.plot_performance
```

Artifacts:
- `outputs/plots/performance.png` (latency and bytes per client)
- `outputs/metrics/performance_summary.csv`

- Compare centralized baseline vs federated accuracy and generate an overlay plot:

```
python3 -m scripts.compare_baseline_vs_fl
```

Artifacts:
- `outputs/plots/centralized_vs_fl.png`
- `outputs/metrics/baseline_vs_fl.csv`

- Optional: Open the notebook `notebooks/performance_and_comparison.ipynb` to reproduce these plots and print a simple equivalence decision (±0.01 tolerance on AUC and F1).

### Keep images small

- We use `python:3.11-slim`, avoid apt packages, and avoid copying the repo into the image.
- Only the server builds the image; clients share it via `image:` setting.
- Clean up dangling images occasionally:

```
docker system prune -f
```

### Repo Highlights

- `models/logistic_regression.py`: pure-Python logistic regression (SGD/minibatch, L2, CSV save/load)
- `scripts/preprocess_data.py`: CSV-only preprocessing and per-client partitioning
- `scripts/train_centralized.py`: centralized training and metrics (AUC/Precision/Recall/F1) computed in pure Python
- `server/server.py`, `server/fedavg.py`: FL orchestration endpoints
- `server/secure_aggregation/*`: ECDH, HKDF+PRG, and mask composition helpers
- `clients/base_client.py`, `clients/client.py`: client loop (global fetch, local train, delta submit); SECAGG optional
- `scripts/encoding.py`: clipping and fixed-point encoding/decoding utilities (CSV-friendly)
- `tests/`: unit tests for encoding, mask cancellation, ECDH-based masks, and dropout simulation
- `docker/docker-compose.yml`: compose services for server and 3 clients
- `notebooks/`: centralized baseline and FL experiment runners

### Tests

```
python3 -m unittest discover -s tests -p 'test_*.py'
```

### Notes

- Pure-Python training is intentionally simple and may be slower than NumPy-based versions. Keep batch sizes and epochs moderate for demos.
- All intermediate outputs are CSV-only to satisfy reproducibility constraints without binary formats.
