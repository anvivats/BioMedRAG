FROM python:3.10-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for Alpine
RUN apk add --no-cache \
    gcc \
    g++ \
    make \
    libffi-dev \
    openssl-dev \
    musl-dev \
    linux-headers

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "api/app.py"]