FROM python:3.9

COPY requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

COPY data /app/data

COPY model_app.py model_train.py /app/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "model_app.py", "--server.port", "8501"]