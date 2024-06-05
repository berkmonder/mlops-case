FROM python:3.9

WORKDIR /mlruns

RUN pip install mlflow psycopg2-binary

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", "--backend-store-uri", "postgresql://user:password@postgres:5432/mlflowdb", "--host", "0.0.0.0", "--port", "5000"]