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

"""Shared dataclasses across requests and responses for Pete."""

import dataclasses

from serving import pete_errors
from serving import pete_flags


@dataclasses.dataclass(frozen=True)
class PatchCoordinate:
  """A coordinate of a patch."""

  x_origin: int
  y_origin: int
  width: int
  height: int

  def __post_init__(self):
    if (
        self.width != pete_flags.ENDPOINT_INPUT_WIDTH_FLAG.value
        or self.height != pete_flags.ENDPOINT_INPUT_HEIGHT_FLAG.value
    ):
      raise pete_errors.PatchDimensionsDoNotMatchEndpointInputDimensionsError(
          'Patch coordinate width and height must be'
          f' {pete_flags.ENDPOINT_INPUT_WIDTH_FLAG.value}x{pete_flags.ENDPOINT_INPUT_HEIGHT_FLAG.value}.'
      )


def create_patch_coordinate(
    x_origin: int,
    y_origin: int,
    width: int = -1,
    height: int = -1,
) -> PatchCoordinate:
  """Creates a patch coordinate."""
  if width == -1:
    width = pete_flags.ENDPOINT_INPUT_WIDTH_FLAG.value
  if height == -1:
    height = pete_flags.ENDPOINT_INPUT_HEIGHT_FLAG.value
  return PatchCoordinate(
      x_origin=x_origin,
      y_origin=y_origin,
      width=width,
      height=height,
  )
