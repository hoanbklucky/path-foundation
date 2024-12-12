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

"""Tests fpr dicom embedding request."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from health_foundations.path_foundation.serving.data_models import embedding_request
from health_foundations.path_foundation.serving.data_models import patch_coordinate


# Necessary to avoid flag parsing errors during unit tests.
def setUpModule():
  flags.FLAGS(['./program'])


class DicomEmbeddingRequestTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._dicom_embedding_parameters = embedding_request.EmbeddingParameters(
        model_size='SMALL', model_kind='LOW_PIXEL_SPACING'
    )
    self._instance_uids_1 = ['1.2.3.4', '2.3.4.5']
    self._patch_coordinate_1 = patch_coordinate.PatchCoordinate(
        x_origin=1,
        y_origin=1,
        width=224,
        height=224,
    )
    self._instance_uids_2 = ['3.4.5.6', '4.5.6.7']
    self._patch_coordinate_2 = patch_coordinate.PatchCoordinate(
        x_origin=2,
        y_origin=3,
        width=224,
        height=224,
    )

    self._dicom_embedding_instance = embedding_request.EmbeddingInstanceV1(
        dicom_web_store_url='potato',
        dicom_study_uid='10.11.12.13',
        dicom_series_uid='11.12.13.14',
        bearer_token='xx_pear_xx',
        ez_wsi_state={'hello': 'world'},
        instance_uids=self._instance_uids_1,
        patch_coordinates=[
            self._patch_coordinate_1,
            self._patch_coordinate_2,
        ],
    )

    self._dicom_embedding_instance_2 = embedding_request.EmbeddingInstanceV1(
        dicom_web_store_url='potato_2',
        dicom_study_uid='12.13.14.15',
        dicom_series_uid='13.14.15.16',
        bearer_token='xx_pineapple_xx',
        ez_wsi_state={'hello': 'goodbye'},
        instance_uids=self._instance_uids_2,
        patch_coordinates=[
            self._patch_coordinate_2,
            self._patch_coordinate_1,
        ],
    )

    self._dicom_embedding_parameters_dict = {
        'model_size': 'SMALL',
        'model_kind': 'LOW_PIXEL_SPACING',
    }
    self._dicom_embedding_instance_dict = {
        'dicom_web_store_url': 'potato',
        'dicom_study_uid': '10.11.12.13',
        'dicom_series_uid': '11.12.13.14',
        'bearer_token': 'xx_pear_xx',
        'ez_wsi_state': {'hello': 'world'},
        'instance_uids': ['1.2.3.4', '2.3.4.5'],
        'patch_coordinates': [
            self._patch_coordinate_1,
            self._patch_coordinate_2,
        ],
    }
    self._dicom_embedding_instance_2_dict = {
        'dicom_web_store_url': 'potato_2',
        'dicom_study_uid': '12.13.14.15',
        'dicom_series_uid': '13.14.15.16',
        'bearer_token': 'xx_pineapple_xx',
        'ez_wsi_state': {'hello': 'goodbye'},
        'instance_uids': ['3.4.5.6', '4.5.6.7'],
        'patch_coordinates': [
            self._patch_coordinate_2,
            self._patch_coordinate_1,
        ],
    }

    self._dicom_embedding_request_dict = {
        'parameters': self._dicom_embedding_parameters,
        'instances': [
            self._dicom_embedding_instance,
            self._dicom_embedding_instance_2,
        ],
    }

  def test_dicom_embedding_parameters(self):
    parameters = self._dicom_embedding_parameters

    self.assertEqual(parameters.__dict__, self._dicom_embedding_parameters_dict)

  def test_dicom_embedding_instance(self):
    parameters = self._dicom_embedding_instance

    self.assertEqual(parameters.__dict__, self._dicom_embedding_instance_dict)

  def test_dicom_embedding_request(self):
    parameters = embedding_request.EmbeddingRequestV1(
        parameters=self._dicom_embedding_parameters,
        instances=[
            self._dicom_embedding_instance,
            self._dicom_embedding_instance_2,
        ],
    )

    self.assertEqual(parameters.__dict__, self._dicom_embedding_request_dict)


if __name__ == '__main__':
  absltest.main()
