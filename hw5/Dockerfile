FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Copy project metadata and source so uv can build the local package
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install uv && uv sync --frozen --no-dev

COPY pipeline_v1.bin predict.py service.py ./

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]