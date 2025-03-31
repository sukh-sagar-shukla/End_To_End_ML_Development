
### Docker setup

```bash
docker run --rm -it --ipc host \
           --network mlflow-tracking-server_default \
           -v $PWD:/app \
           -v mlflow-tracking-server_artifact-store:/artifact-store \
           continuumio/anaconda3 bash
```

### Before running

```bash
export MLFLOW_EXPERIMENT_NAME='mlflow-projects-demo'
export MLFLOW_TRACKING_URI='http://mlflow-tracking-server:5000'
```

### Running

```bash
mlflow run .
```
