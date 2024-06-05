FROM python:3.9

COPY requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

RUN pip install --force-reinstall --no-deps bokeh==2.4.3 

COPY data /app/data

COPY model_app.py model_train.py logger.py /app/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "model_app.py", "--server.port", "8501"]