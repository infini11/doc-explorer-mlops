FROM apache/airflow:2.8.1  

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

USER airflow

RUN pip install --upgrade pip setuptools wheel

RUN pip install python-dev-tools
RUN pip install --no-cache-dir -r /requirements.txt
