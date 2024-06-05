FROM python:3.9

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /mlruns

EXPOSE 5000

ENTRYPOINT ["mlflow", "serve", "--backend-store-uri", "sqlite:////mlruns/mlflow.db", "--default-artifact-root", "file:///mlruns", "--serve-artifacts", "--host", "0.0.0.0", "--port", "5000"]