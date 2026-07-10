FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY client.py .
RUN useradd --create-home --uid 10001 labuser
USER labuser

CMD ["python", "client.py", "--help"]
