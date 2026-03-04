#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH=/Users/nishitareddy/Trading-Platform
cd /Users/nishitareddy/Trading-Platform
exec python3 -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --log-level info
