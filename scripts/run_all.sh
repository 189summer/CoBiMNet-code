#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_etth1.sh
bash scripts/run_etth2.sh
bash scripts/run_ettm1.sh
bash scripts/run_ettm2.sh
bash scripts/run_weather.sh
bash scripts/run_electricity.sh
bash scripts/run_traffic.sh
