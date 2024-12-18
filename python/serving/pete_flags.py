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

"""Pete flags."""

import json
import os
import sys
from typing import List, Optional, Union

from absl import flags


def _load_multi_string(val: Optional[str]) -> Optional[Union[List[str], str]]:
  if val is None:
    return None
  try:
    return json.loads(val)
  except json.decoder.JSONDecodeError:
    return val


ENDPOINT_LOG_NAME_FLAG = flags.DEFINE_string(
    'endpoint_log_name',
    os.environ.get('ENDPOINT_LOG_NAME', ''),
    'Optional name write in endpoint logs to easily identify endpoints.',
)

# If true and Redis host is defined stores ICC Profile bytes in redis.
ICC_PROFILE_CACHE_GCS_BUCKET_FLAG = flags.DEFINE_string(
    'icc_profile_cache_gcs_bucket',
    os.environ.get('ICC_PROFILE_CACHE_GCS_BUCKET', ''),
    'Name of gcs bucket to cache icc profile to.',
)

ICC_PROFILE_CACHE_REDIS_IP_FLAG = flags.DEFINE_string(
    'icc_profile_cache_redis_ip',
    os.environ.get('ICC_PROFILE_CACHE_REDIS_IP', ''),
    'IP address of REDIS server to cache cache icc profile to.',
)

ICC_PROFILE_CACHE_REDIS_PORT_FLAG = flags.DEFINE_integer(
    'icc_profile_cache_redis_port',
    int(os.environ.get('ICC_PROFILE_CACHE_REDIS_PORT', '6379')),
    'Port of REDIS server to cache cache icc profile to.',
)

# If true and Redis host is defined stores ICC Profile bytes in redis.
STORE_ICC_PROFILE_BYTES_IN_REDIS_FLAG = flags.DEFINE_bool(
    'store_icc_profile_bytes_in_redis',
    bool(os.environ.get('STORE_ICC_PROFILE_BYTES_IN_REDIS', False)),
    'bool cache icc profile bytes in redis',
)

# If true and Redis host is defined stores ICC Profile bytes in redis.
IS_DEBUGGING_FLAG = flags.DEFINE_bool(
    'is_debugging',
    bool(
        os.environ.get(
            'IS_DEBUGGING',
            'UNITTEST_ON_FORGE' in os.environ or 'unittest' in sys.modules,
        )
    ),
    'internal flag for unit tests detects if running in debugger.',
)

APPROVED_GCS_SOURCE_LIST_FLAG = flags.DEFINE_multi_string(
    'approved_gcs_source_list',
    _load_multi_string(os.environ.get('APPROVED_GCS_SOURCE_LIST', None)),
    'List of GCS buckets endpoints can read from; all are allowed if'
    ' undefined.',
)


APPROVED_DICOM_STORE_SOURCE_LIST_FLAG = flags.DEFINE_multi_string(
    'approved_dicom_store_source_list',
    _load_multi_string(
        os.environ.get('APPROVED_DICOM_STORE_SOURCE_LIST', None)
    ),
    'List of DICOM stores endpoint can read from; all are allowed if'
    ' undefined.',
)

ENDPOINT_INPUT_WIDTH_FLAG = flags.DEFINE_integer(
    'endpoint_input_width',
    int(
        os.environ.get('ENDPOINT_INPUT_WIDTH', 224)
    ),
    'Width in pixels of input image to endpoint.',
)

ENDPOINT_INPUT_HEIGHT_FLAG = flags.DEFINE_integer(
    'endpoint_input_height',
    int(
        os.environ.get('ENDPOINT_INPUT_HEIGHT', 224)
    ),
    'Height in pixels of input image to endpoint.',
)
