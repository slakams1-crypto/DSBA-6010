from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI
from fastapi.responses import Response

# Define metrics
REQUEST_COUNT = Counter("app_requests_total", "Total requests")
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency in seconds")

# Create FastAPI app for metrics endpoint
app = FastAPI()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.middleware("http")
async def add_metrics(request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(latency)
    return response