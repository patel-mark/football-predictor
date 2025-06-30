#!/bin/bash

# Start Gunicorn with Uvicorn workers
exec gunicorn -c deployment/gunicorn_config.py deployment.fastapi_app:app
