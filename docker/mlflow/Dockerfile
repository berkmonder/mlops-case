FROM python:3.9

WORKDIR /mlruns

RUN pip install mlflow

EXPOSE 5000

ENTRYPOINT ["mlflow", "serve", "--backend-store-uri", "postgresql://user:password@localhost:5432/mlflowdb", "--host", "0.0.0.0", "--port", "5000"]