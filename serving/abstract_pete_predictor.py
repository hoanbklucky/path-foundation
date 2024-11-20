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

"""Defines abstract pathology embedding prediction interface."""
import abc
from typing import Any, Mapping

from prediction_container import model_runner


class AbstractPetePredictor(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def predict(
      self,
      prediction_input: Mapping[str, Any],
      model: model_runner.ModelRunner,
  ) -> Mapping[str, Any]:
    """Returns embeddings for embedding prediction requests."""
