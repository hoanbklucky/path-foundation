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
# ==============================================================================
"""Mock Embedding Endpoint."""

from collections.abc import Set
import http
import io
import json
from typing import List, Mapping, Optional, Sequence

import numpy as np
import requests
import requests_mock

from serving.serving_framework import model_runner
from serving import abstract_pete_predictor


class _MockModel(model_runner.ModelRunner):
  """Mocks model."""

  def mock_model(self, data: np.ndarray) -> np.ndarray:
    return np.mean(data, axis=(1, 2))

  def run_model_multiple_output(
      self,
      model_input: Mapping[str, np.ndarray] | np.ndarray,
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_keys: Set[str],
  ) -> Mapping[str, np.ndarray]:
    raise NotImplementedError("Not implemented.")

  def run_model(
      self,
      model_input: Mapping[str, np.ndarray] | np.ndarray,
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_key: str = "output_0",
  ) -> np.ndarray:
    if not isinstance(model_input, np.ndarray):
      raise ValueError("Model input must be a numpy array.")
    return self.mock_model(model_input)

  def batch_model(
      self,
      model_inputs: Sequence[Mapping[str, np.ndarray]] | Sequence[np.ndarray],
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_key: str = "output_0",
  ) -> List[np.ndarray]:
    if not isinstance(model_inputs[0], np.ndarray):
      raise ValueError("Model input must be a Sequence of numpy array.")
    return [self.mock_model(model_input) for model_input in model_inputs]


class EndpointMock:
  """Mocks Pathology Embedding Enpoint."""

  def __init__(
      self,
      mock_request: requests_mock.Mocker,
      mock_endpoint_url: str,
      pete_endpoint: abstract_pete_predictor.AbstractPetePredictor,
  ):
    self._mock_endpoint_url = mock_endpoint_url
    mock_request.add_matcher(self._handle_request)
    self._pete_endpoint = pete_endpoint
    self._mock_model_runner = _MockModel()

  def _handle_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Handles a request for the mock embedding endpoint.

    Args:
      request: The request to handle.

    Returns:
      None if request not handled otherwise mock V1 embedding response.
      Mock embedding is mean channel value per patch.
    """
    if not request.url.startswith(self._mock_endpoint_url):
      return None
    result = self._pete_endpoint.predict(
        request.json(), self._mock_model_runner
    )
    resp = requests.Response()
    resp.status_code = http.HTTPStatus.OK
    msg = json.dumps(result).encode("utf-8")
    resp.raw = io.BytesIO(msg)
    return resp
