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

"""Tests for embedding response."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import patch_embedding_endpoints
from health_foundations.path_foundation.serving import pete_errors
from health_foundations.path_foundation.serving.data_models import embedding_response
from health_foundations.path_foundation.serving.data_models import patch_coordinate

_EndpointJsonKeys = patch_embedding_endpoints.EndpointJsonKeys


# Necessary to avoid flag parsing errors during unit tests.
def setUpModule():
  flags.FLAGS(['./program'])


class DicomEmbeddingResponseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._model_version = 'model_version'

    self._instance_uids = ['eggplant', 'basil']
    self._patch_coordinate = patch_coordinate.PatchCoordinate(
        x_origin=1,
        y_origin=3,
        width=224,
        height=224,
    )
    self._instance_uids_2 = ['tomato', 'tomatillo']
    self._patch_coordinate_2 = patch_coordinate.PatchCoordinate(
        x_origin=2,
        y_origin=3,
        width=224,
        height=224,
    )
    self._pete_error = embedding_response.PeteErrorResponse(
        error_code=embedding_response.ErrorCode.TOO_MANY_PATCHES_ERROR
    )
    self.dicom_embedding_result_item_1 = embedding_response.PatchEmbeddingV1(
        embeddings=[i + 1 for i in range(384)],
        patch_coordinate=self._patch_coordinate,
    )
    self.dicom_embedding_result_item_2 = embedding_response.PatchEmbeddingV1(
        embeddings=[i + 13 for i in range(384)],
        patch_coordinate=self._patch_coordinate_2,
    )
    self.dicom_embedding_embedding_result_1 = (
        embedding_response.EmbeddingResultV1(
            dicom_study_uid='potato',
            dicom_series_uid='tomato',
            instance_uids=self._instance_uids,
            patch_embeddings=[
                self.dicom_embedding_result_item_1,
                self.dicom_embedding_result_item_2,
            ],
        )
    )
    self.dicom_embedding_embedding_result_2 = (
        embedding_response.EmbeddingResultV1(
            dicom_study_uid='grape',
            dicom_series_uid='berry',
            instance_uids=self._instance_uids_2,
            patch_embeddings=[
                self.dicom_embedding_result_item_2,
                self.dicom_embedding_result_item_1,
            ],
        )
    )
    self.dicom_embedding_response = embedding_response.EmbeddingResponseV1(
        model_version=self._model_version,
        error_response=None,
        embedding_result=[
            self.dicom_embedding_embedding_result_1,
            self.dicom_embedding_embedding_result_2,
        ],
    )
    self.dicom_embedding_error_response = (
        embedding_response.EmbeddingResponseV1(
            model_version=self._model_version,
            error_response=self._pete_error,
            embedding_result=None,
        )
    )

    self._patch_coordinate_dict = {
        'x_origin': 1,
        'y_origin': 3,
        'width': 224,
        'height': 224,
    }

    self._error_response_dict = {
        'model_version': self._model_version,
        'error_response': self._pete_error,
        'embedding_result': None,
    }
    self._response_dict = {
        'model_version': self._model_version,
        'error_response': None,
        'embedding_result': [
            self.dicom_embedding_embedding_result_1,
            self.dicom_embedding_embedding_result_2,
        ],
    }

  def test_pete_error_response(self):
    parameters = self.dicom_embedding_error_response

    self.assertEqual(parameters.__dict__, self._error_response_dict)

  def test_dicom_embedding_result_item(self):
    parameters = self.dicom_embedding_response

    self.assertEqual(parameters.__dict__, self._response_dict)

  def test_dicom_embedding_response_fails(self):
    with self.assertRaises(pete_errors.InvalidResponseError):
      embedding_response.EmbeddingResponseV1(
          model_version=self._model_version,
          error_response=None,
          embedding_result=None,
      )

  def test_embedding_instance_response_v2(self):
    embedding = embedding_response.PatchEmbeddingV2(
        [1, 2, 3, 4], patch_coordinate.PatchCoordinate(0, 10, 224, 224)
    )
    self.assertEqual(
        embedding_response.embedding_instance_response_v2([embedding] * 2),
        {
            'result': {
                'patch_embeddings': [
                    {
                        'embedding_vector': [1, 2, 3, 4],
                        'patch_coordinate': {
                            'x_origin': 0,
                            'y_origin': 10,
                            'width': 224,
                            'height': 224,
                        },
                    },
                    {
                        'embedding_vector': [1, 2, 3, 4],
                        'patch_coordinate': {
                            'x_origin': 0,
                            'y_origin': 10,
                            'width': 224,
                            'height': 224,
                        },
                    },
                ]
            },
        },
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='code_only',
          description='',
          expected={
              _EndpointJsonKeys.ERROR: {
                  _EndpointJsonKeys.ERROR_CODE: 'TOO_MANY_PATCHES_ERROR',
              },
          },
      ),
      dict(
          testcase_name='code_and_description',
          description='foo',
          expected={
              _EndpointJsonKeys.ERROR: {
                  _EndpointJsonKeys.ERROR_CODE: 'TOO_MANY_PATCHES_ERROR',
                  _EndpointJsonKeys.ERROR_CODE_DESCRIPTION: 'foo',
              },
          },
      ),
  ])
  def test_instance_error_response_v2(self, description, expected):
    self.assertEqual(
        embedding_response.instance_error_response_v2(
            embedding_response.ErrorCode.TOO_MANY_PATCHES_ERROR,
            description=description,
        ),
        expected,
    )

  def test_prediction_error_response_v2(self):
    self.assertEqual(
        embedding_response.prediction_error_response_v2(
            embedding_response.ErrorCode.TOO_MANY_PATCHES_ERROR
        ),
        {_EndpointJsonKeys.VERTEXAI_ERROR: 'TOO_MANY_PATCHES_ERROR'},
    )


if __name__ == '__main__':
  absltest.main()
