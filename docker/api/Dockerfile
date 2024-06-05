FROM python:3.9

COPY requirements.txt app/

WORKDIR /app

RUN pip install -r requirements.txt

COPY data /app/data
COPY models /app/models

COPY app.py model_train.py logger.py /app/

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]