#!/bin/bash
set -e

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db init

# Create admin user with admin/admin credentials
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists"

# Start scheduler in the background, webserver as the foreground (PID 1) process.
# Avoid `--daemon` mode: its pidfile lives on /opt/airflow, which survives a plain
# container restart, so a stale pidfile from a prior run can wrongly block startup
# and, since it was launched with `&`, that failure was silently swallowed.
echo "Starting Airflow scheduler..."
airflow scheduler &

echo "Starting Airflow webserver..."
exec airflow webserver --port 8080