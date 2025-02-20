FROM python:3.10-slim

RUN useradd -m mlflowuser

USER mlflowuser
WORKDIR /home/mlflowuser

RUN pip install mlflow==2.17.2 psycopg2-binary==2.9.10

ENV PATH="/home/mlflowuser/.local/bin:$PATH"
