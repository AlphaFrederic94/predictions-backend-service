# Docker Setup for Medical Prediction Services

This document provides detailed information about the Docker setup for the Medical Prediction Services application.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, but recommended)

## Docker Image

The Docker image for this application is built using a Python 3.12 slim base image. It includes all the necessary dependencies for running the application, including:

- FastAPI for the API framework
- Uvicorn as the ASGI server
- Scikit-learn for machine learning models
- Pandas and NumPy for data processing
- Other dependencies as specified in the requirements.txt file

## Building and Running with Docker

### Using Docker Directly

1. Build the Docker image:
   ```bash
   docker build -t medical-prediction-api .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 --name medical-api medical-prediction-api
   ```

3. Stop and remove the container:
   ```bash
   docker stop medical-api
   docker rm medical-api
   ```

### Using Docker Compose

1. Start the application:
   ```bash
   docker compose up -d
   ```

2. Stop the application:
   ```bash
   docker compose down
   ```

## Configuration

### Environment Variables

The following environment variables can be set to configure the application:

- `ENVIRONMENT`: Set to `production` for production mode, `development` for development mode
- Additional environment variables can be added in the docker-compose.yml file

### Volumes

The Docker Compose setup includes a volume mapping for the data directory:

```yaml
volumes:
  - ./data:/app/data
```

This ensures that the data files are persisted even if the container is removed.

## Accessing the Application

Once the container is running, you can access the application at:

- API Root: http://localhost:8000/
- API Documentation: http://localhost:8000/docs

## Troubleshooting

### Common Issues

1. **Port already in use**:
   If port 8000 is already in use on your system, you can change the port mapping in the docker-compose.yml file:
   ```yaml
   ports:
     - "8001:8000"  # Maps host port 8001 to container port 8000
   ```

2. **Container not starting**:
   Check the logs for errors:
   ```bash
   docker logs medical-api
   # or with docker compose
   docker compose logs
   ```

3. **Missing data files**:
   Make sure the data directory contains all necessary files and is properly mounted in the container.

## Advanced Configuration

### Custom Dockerfile

You can customize the Dockerfile to suit your needs. For example, you might want to use a multi-stage build to reduce the image size:

```dockerfile
# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Health Checks

You can add health checks to your Docker Compose configuration to ensure the application is running correctly:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Security Considerations

When deploying to production, consider the following security measures:

1. Use specific versions for base images instead of `latest` tags
2. Run the container as a non-root user
3. Implement proper authentication and authorization for the API
4. Use environment variables for sensitive configuration
5. Regularly update dependencies to patch security vulnerabilities
