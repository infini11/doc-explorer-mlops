FROM postgres:17
COPY ../../scripts/create_multiple_databases.sh /docker-entrypoint-initdb.d/
RUN chmod +x /docker-entrypoint-initdb.d/create_multiple_databases.sh