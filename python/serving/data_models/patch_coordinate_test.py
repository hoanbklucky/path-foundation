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

"""Tests for patch coordinate."""

from absl.testing import absltest
from absl.testing import parameterized
from serving import pete_errors
from serving.data_models import patch_coordinate


class PatchCoordinateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._patch_coordinate_zero_dimensions = (
        patch_coordinate.create_patch_coordinate(
            x_origin=1,
            y_origin=3,
        )
    )
    self._patch_coordinate_zero_dimensions_dict = {
        'x_origin': 1,
        'y_origin': 3,
        'width': 224,
        'height': 224,
    }

  def test_dicom_embedding_patch_coordinate_invalid_dimensions(self):
    with self.assertRaises(
        pete_errors.PatchDimensionsDoNotMatchEndpointInputDimensionsError
    ):
      _ = patch_coordinate.PatchCoordinate(
          x_origin=1,
          y_origin=3,
          width=11,
          height=10,
      )

  def test_dicom_embedding_patch_coordinate_default_dimensions(self):
    parameters = self._patch_coordinate_zero_dimensions

    self.assertEqual(
        parameters.__dict__, self._patch_coordinate_zero_dimensions_dict
    )


if __name__ == '__main__':
  absltest.main()
