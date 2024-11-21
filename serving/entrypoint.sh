#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script launches the serving framework, run as the entrypoint.

# Exit if any command fails or if expanding an undefined variable.
set -eu

export MODEL_REST_PORT=8600
export LOCAL_MODEL_PATH=/model

echo "Serving framework start, launching model server"

/server-env/bin/python3.12 serving_framework/model_transfer.py \
    --gcs_path="${AIP_STORAGE_URI}" \
    --local_path="${LOCAL_MODEL_PATH}/1"
/usr/bin/tensorflow_model_server \
    --xla_cpu_compilation_enabled=true \
    --port=8500 \
    --rest_api_port="${MODEL_REST_PORT}" \
    --model_name=default \
    --model_base_path="${LOCAL_MODEL_PATH}" &

echo "Launching front end"

/server-env/bin/python3.12 server_gunicorn.py --alsologtostderr \
    --verbosity=1 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?