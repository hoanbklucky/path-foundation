# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is used to build a Docker image that includes the necessary dependencies
# for running the Path Foundation as a microservice.

FROM tensorflow/serving:2.18.0

COPY ./python/serving /serving
COPY ./LICENSE /LICENSE
RUN chmod a+x /serving/entrypoint.sh

# Install python3.12 and git
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y git python3.12 python3.12-venv

# Get pypi requirements
RUN python3.12 -m venv /server-env && \
    /server-env/bin/python3.12 -m pip install --upgrade pip && \
    /server-env/bin/python3.12 -m pip install -r /serving/requirements.txt

# Clone python-certifi to meet MPL 2.0 License terms for source code mirroring
RUN git clone https://github.com/certifi/python-certifi.git

ENTRYPOINT ["/serving/entrypoint.sh"]
