FROM python:3.11-slim

WORKDIR /app
COPY server.py .
RUN useradd --create-home --uid 10001 labuser
USER labuser
CMD ["python", "server.py"]
