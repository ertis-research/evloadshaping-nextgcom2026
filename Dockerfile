FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# The dataset is not baked into the image (see .dockerignore) since it is
# large and not distributed with the source. Mount it at run time, e.g.:
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs ev-load-shaping
CMD ["python", "pipeline.py", "--data-path", "data/20260514_uma_adabyron_data.csv"]