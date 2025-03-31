import mlflow
import numpy as np


mlflow.set_tracking_uri("http://mlflow-tracking-server:5000")
logged_model = 'runs:/129a339c0a2e46c1aa4d22ca12f3c38c/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)
random_data = np.random.randn(1, 87931)
prediction = loaded_model.predict(random_data)
print(prediction)
