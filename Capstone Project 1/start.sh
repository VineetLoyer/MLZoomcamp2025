#!/bin/bash
exec gunicorn --bind 0.0.0.0:${PORT:-9696} --workers 2 --timeout 120 predict:app
