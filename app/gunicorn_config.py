import multiprocessing

# Server socket
bind = "0.0.0.0:8000"

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout
timeout = 120

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr