
### Docker setup

```bash
docker run --rm -it --ipc host -p 1234:1234 \
           --network mlflow-tracking-server_default \
           -v $PWD:/app \
           -v mlflow-tracking-server_artifact-store:/artifact-store \
           continuumio/anaconda3 bash
```

### Before running

```bash
export MLFLOW_TRACKING_URI='http://mlflow-tracking-server:5000'
```

### Model serving

```bash
mlflow models serve -m runs:/<run-id>/<model-path> -h 0.0.0.0 -p 1234
```

### Request

```bash
curl http://0.0.0.0:1234/invocations -H 'Content-Type: application/json' -d '{
    "columns": ["review"],
    "data": [["<text>"]]
}'
```
