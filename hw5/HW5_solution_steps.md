# HW5 Solution Steps

## Question 1: Install `uv`
- Created a clean homework directory at `hw5/`.
- Installed uv via `pip install uv` in the base environment (see terminal history).
- Verified installation using `uv --version` to report the installed version.

## Question 2: Add scikit-learn 1.6.1
- Initialized uv project (`uv init`) which produced `pyproject.toml` and `uv.lock`.
- Added the dependency with `uv add scikit-learn==1.6.1`.
- Located the first wheel hash in `uv.lock` under the `scikit-learn` package section and recorded the `sha256` value.

## Question 3: Local Scoring Script
- Downloaded pipeline via PowerShell `Invoke-WebRequest` to `pipeline_v1.bin`.
- Implemented the loader and scorer in `predict.py` (module referencing the FastAPI schema’s payload keys).
- Executed `uv run python predict.py` to get the conversion probability for the provided lead.

## Question 4: FastAPI Service
- Added FastAPI, Uvicorn, and Requests to project dependencies (`uv add fastapi uvicorn requests`).
- Implemented service endpoints in `service.py` (defines `Lead` model, `/health`, `/predict`).
- Ran API locally with `uv run uvicorn service:app --host 0.0.0.0 --port 8000`.
- Tested the `/predict` route via PowerShell `Invoke-RestMethod` to capture the response probability.

## Question 5: Docker Base Image Size
- Pulled base image from Docker Hub: `docker pull agrigorev/zoomcamp-model:2025`.
- Retrieved size using `docker images agrigorev/zoomcamp-model:2025` to answer the size question.

## Question 6: Containerized Service
- Authored `Dockerfile` (see `hw5/Dockerfile`) extending the base image, installing dependencies with uv, and launching Uvicorn.
- Built container image with `DOCKER_BUILDKIT=0 docker build -t hw5-service .` (WSL terminal).
- Ran container via `docker run -p 8000:8000 hw5-service`.
- Sent POST request using `curl` to the running container and recorded the probability (0.534).
- Stopped the container after recording results.

All referenced files reside in the `hw5/` directory:
- Scoring script: `hw5/predict.py`
- FastAPI service: `hw5/service.py`
- Docker setup: `hw5/Dockerfile`
- Lockfile with hashes: `hw5/uv.lock`
