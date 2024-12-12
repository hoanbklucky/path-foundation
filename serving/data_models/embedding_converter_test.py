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

"""Tests for embedding models data transformer."""

import dataclasses
import json
import typing
from typing import Any, List, Mapping

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import patch_embedding_endpoints

from health_foundations.path_foundation.serving import pete_errors
from health_foundations.path_foundation.serving.data_models import embedding_converter
from health_foundations.path_foundation.serving.data_models import embedding_request
from health_foundations.path_foundation.serving.data_models import embedding_response
from health_foundations.path_foundation.serving.data_models import patch_coordinate


_EndpointJsonKeys = patch_embedding_endpoints.EndpointJsonKeys


# Necessary to avoid flag parsing errors during unit tests.
def setUpModule():
  flags.FLAGS(['./program'])


class EmbeddingConverterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._embedding_request_dict = {
        'parameters': {
            'model_size': 'SMALL',
            'model_kind': 'LOW_PIXEL_SPACING',
        },
        'instances': [
            {
                'dicom_web_store_url': 'potato',
                'dicom_study_uid': '10.11.12.13',
                'dicom_series_uid': '11.12.13.14',
                'bearer_token': 'xx_pear_xx',
                'ez_wsi_state': '{"hello": "world"}',
                'instance_uids': ['1.2.3.4'],
                'patch_coordinates': [
                    {
                        'x_origin': 1,
                        'y_origin': 1,
                        'width': 224,
                        'height': 224,
                    },
                    {
                        'x_origin': 2,
                        'y_origin': 3,
                        'width': 224,
                        'height': 224,
                    },
                ],
            },
            {
                'dicom_web_store_url': 'tomato',
                'dicom_study_uid': '10.11.12.13',
                'dicom_series_uid': '11.12.13.14',
                'bearer_token': 'xx_pear_xx',
                'ez_wsi_state': '{"hello": "world"}',
                'instance_uids': ['1.2.3.4', '5.6.7.8'],
                'patch_coordinates': [
                    {
                        'x_origin': 2,
                        'y_origin': 3,
                        'width': 224,
                        'height': 224,
                    },
                    {
                        'x_origin': 1,
                        'y_origin': 1,
                        'width': 224,
                        'height': 224,
                    },
                ],
            },
        ],
    }
    self._embedding_request = embedding_request.EmbeddingRequestV1(
        parameters=embedding_request.EmbeddingParameters(
            model_size='SMALL', model_kind='LOW_PIXEL_SPACING'
        ),
        instances=[
            embedding_request.EmbeddingInstanceV1(
                dicom_web_store_url='potato',
                dicom_study_uid='10.11.12.13',
                dicom_series_uid='11.12.13.14',
                bearer_token='xx_pear_xx',
                ez_wsi_state='{"hello": "world"}',
                instance_uids=['1.2.3.4'],
                patch_coordinates=[
                    patch_coordinate.PatchCoordinate(
                        x_origin=1,
                        y_origin=1,
                        width=224,
                        height=224,
                    ),
                    patch_coordinate.PatchCoordinate(
                        x_origin=2,
                        y_origin=3,
                        width=224,
                        height=224,
                    ),
                ],
            ),
            embedding_request.EmbeddingInstanceV1(
                dicom_web_store_url='tomato',
                dicom_study_uid='10.11.12.13',
                dicom_series_uid='11.12.13.14',
                bearer_token='xx_pear_xx',
                ez_wsi_state='{"hello": "world"}',
                instance_uids=['1.2.3.4', '5.6.7.8'],
                patch_coordinates=[
                    patch_coordinate.PatchCoordinate(
                        x_origin=2,
                        y_origin=3,
                        width=224,
                        height=224,
                    ),
                    patch_coordinate.PatchCoordinate(
                        x_origin=1,
                        y_origin=1,
                        width=224,
                        height=224,
                    ),
                ],
            ),
        ],
    )

    self._embedding_response = embedding_response.EmbeddingResponseV1(
        model_version='11.13.00',
        error_response=None,
        embedding_result=[
            embedding_response.EmbeddingResultV1(
                dicom_study_uid='1.2.3.4',
                dicom_series_uid='5.6.7.8',
                instance_uids=['eggplant', 'basil'],
                patch_embeddings=[
                    embedding_response.PatchEmbeddingV1(
                        embeddings=[i + 1 for i in range(3)],
                        patch_coordinate=patch_coordinate.PatchCoordinate(
                            x_origin=1,
                            y_origin=1,
                            width=224,
                            height=224,
                        ),
                    ),
                    embedding_response.PatchEmbeddingV1(
                        embeddings=[i + 13 for i in range(3)],
                        patch_coordinate=patch_coordinate.PatchCoordinate(
                            x_origin=2,
                            y_origin=3,
                            width=224,
                            height=224,
                        ),
                    ),
                ],
            ),
            embedding_response.EmbeddingResultV1(
                dicom_study_uid='10.11.12.13',
                dicom_series_uid='14.15.16.17',
                instance_uids=['grape', 'cherry'],
                patch_embeddings=[
                    embedding_response.PatchEmbeddingV1(
                        embeddings=[i + 1 for i in range(3)],
                        patch_coordinate=patch_coordinate.PatchCoordinate(
                            x_origin=1,
                            y_origin=1,
                            width=224,
                            height=224,
                        ),
                    ),
                    embedding_response.PatchEmbeddingV1(
                        embeddings=[i + 13 for i in range(3)],
                        patch_coordinate=patch_coordinate.PatchCoordinate(
                            x_origin=2,
                            y_origin=3,
                            width=224,
                            height=224,
                        ),
                    ),
                ],
            ),
        ],
    )

    self._embedding_response_error = embedding_response.EmbeddingResponseV1(
        model_version='11.13.00',
        error_response=embedding_response.PeteErrorResponse(
            error_code=embedding_response.ErrorCode.TOO_MANY_PATCHES_ERROR
        ),
        embedding_result=None,
    )

    self._embedding_response_dict = {
        'predictions': {
            'model_version': '11.13.00',
            'error_response': None,
            'embedding_result': [
                {
                    'dicom_study_uid': '1.2.3.4',
                    'dicom_series_uid': '5.6.7.8',
                    'instance_uids': ['eggplant', 'basil'],
                    'patch_embeddings': [
                        {
                            'embeddings': [i + 1 for i in range(3)],
                            'patch_coordinate': {
                                'x_origin': 1,
                                'y_origin': 1,
                                'width': 224,
                                'height': 224,
                            },
                        },
                        {
                            'embeddings': [i + 13 for i in range(3)],
                            'patch_coordinate': {
                                'x_origin': 2,
                                'y_origin': 3,
                                'width': 224,
                                'height': 224,
                            },
                        },
                    ],
                },
                {
                    'dicom_study_uid': '10.11.12.13',
                    'dicom_series_uid': '14.15.16.17',
                    'instance_uids': ['grape', 'cherry'],
                    'patch_embeddings': [
                        {
                            'embeddings': [i + 1 for i in range(3)],
                            'patch_coordinate': {
                                'x_origin': 1,
                                'y_origin': 1,
                                'width': 224,
                                'height': 224,
                            },
                        },
                        {
                            'embeddings': [i + 13 for i in range(3)],
                            'patch_coordinate': {
                                'x_origin': 2,
                                'y_origin': 3,
                                'width': 224,
                                'height': 224,
                            },
                        },
                    ],
                },
            ],
        }
    }

    self._embedding_response_error_dict = {
        'predictions': {
            'model_version': '11.13.00',
            'error_response': {
                'error_code': 'INFERENCE_TIME_OUT',
            },
            'embedding_result': None,
        }
    }

    self._transformer = embedding_converter.EmbeddingConverterV1()

  def test_json_to_embedding_request(self):
    request = self._transformer.json_to_embedding_request(
        self._embedding_request_dict
    )

    self.assertEqual(
        dataclasses.asdict(request), dataclasses.asdict(self._embedding_request)
    )

  def test_embedding_response_to_json(self):
    response = embedding_converter.embedding_response_v1_to_json(
        self._embedding_response
    )
    self.assertEqual(response, self._embedding_response_dict)

  def test_embedding_response_to_json_minimum_dicom(self):
    instances = {
        'instances': [
            dict(
                dicom_path=dict(series_path='foo', instance_uids=['1.2']),
                patch_coordinates=[dict(x_origin=3, y_origin=4)],
            )
        ]
    }
    result = (
        embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
            instances
        )
    )
    self.assertEqual(
        result,
        embedding_request.EmbeddingRequestV2([
            embedding_request.DicomImageV2(
                'foo',
                '',
                {},
                ['1.2'],
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=4, width=224, height=224
                    )
                ],
            )
        ]),
    )

  def test_embedding_response_to_json_full_dicom(self):
    instances = {
        'instances': [
            dict(
                dicom_path=dict(
                    series_path='foo',
                    instance_uids=['1.2', '1.3'],
                ),
                bearer_token='ABCD',
                extensions={'EGF': 'QRS'},
                patch_coordinates=[
                    dict(x_origin=3, y_origin=0, width=224, height=224),
                    dict(x_origin=5, y_origin=6, width=224, height=224),
                ],
            )
        ]
    }
    result = (
        embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
            instances
        )
    )
    self.assertEqual(
        result,
        embedding_request.EmbeddingRequestV2([
            embedding_request.DicomImageV2(
                'foo',
                'ABCD',
                {'EGF': 'QRS'},
                ['1.2', '1.3'],
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=0, width=224, height=224
                    ),
                    patch_coordinate.PatchCoordinate(
                        x_origin=5, y_origin=6, width=224, height=224
                    ),
                ],
            )
        ]),
    )

  def test_embedding_response_to_json_minimum_gcs_image(self):
    instances = {
        'instances': [
            dict(
                image_file_uri='foo',
                patch_coordinates=[dict(x_origin=3, y_origin=4)],
            )
        ]
    }
    result = (
        embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
            instances
        )
    )
    self.assertEqual(
        result,
        embedding_request.EmbeddingRequestV2([
            embedding_request.GcsImageV2(
                'foo',
                '',
                {},
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=4, width=224, height=224
                    )
                ],
            )
        ]),
    )

  def test_embedding_response_to_json_full_gcs_image(self):
    instances = {
        'instances': [
            dict(
                image_file_uri='foo',
                bearer_token='ABCD',
                extensions={'EGF': 'QRS'},
                patch_coordinates=[
                    dict(x_origin=3, y_origin=0, width=224, height=224),
                    dict(x_origin=5, y_origin=6, width=224, height=224),
                ],
            )
        ]
    }
    result = (
        embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
            instances
        )
    )
    self.assertEqual(
        result,
        embedding_request.EmbeddingRequestV2([
            embedding_request.GcsImageV2(
                'foo',
                'ABCD',
                {'EGF': 'QRS'},
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=0, width=224, height=224
                    ),
                    patch_coordinate.PatchCoordinate(
                        x_origin=5, y_origin=6, width=224, height=224
                    ),
                ],
            )
        ]),
    )

  def test_embedding_response_to_json_minimum_image_bytes(self):
    instances = {
        'instances': [
            dict(
                raw_image_bytes='foo',
                patch_coordinates=[dict(x_origin=3, y_origin=4)],
            )
        ]
    }
    result = (
        embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
            instances
        )
    )
    self.assertEqual(
        result,
        embedding_request.EmbeddingRequestV2([
            embedding_request.EmbeddedImageV2(
                'foo',
                {},
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=4, width=224, height=224
                    )
                ],
            )
        ]),
    )

  def test_embedding_response_to_json_full_image_bytes(self):
    instances = {
        'instances': [
            dict(
                raw_image_bytes='foo',
                extensions={'EGF': 'QRS'},
                patch_coordinates=[
                    dict(x_origin=3, y_origin=0, width=224, height=224),
                    dict(x_origin=5, y_origin=6, width=224, height=224),
                ],
            )
        ]
    }
    result = (
        embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
            instances
        )
    )
    self.assertEqual(
        result,
        embedding_request.EmbeddingRequestV2([
            embedding_request.EmbeddedImageV2(
                'foo',
                {'EGF': 'QRS'},
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=0, width=224, height=224
                    ),
                    patch_coordinate.PatchCoordinate(
                        x_origin=5, y_origin=6, width=224, height=224
                    ),
                ],
            )
        ]),
    )

  @parameterized.parameters([
      ({'dicom_web_store_url': ''},),
      ({'dicom_web_store_url': 123},),
      ({'dicom_study_uid': 123},),
      ({'dicom_study_uid': ''},),
      ({'dicom_series_uid': 123},),
      ({'dicom_series_uid': ''},),
      ({'bearer_token': ''},),
      ({'bearer_token': 123},),
      ({'ez_wsi_state': 123},),
      ({'instance_uids': 123},),
      ({'instance_uids': [123]},),
      ({'instance_uids': []},),
      ({'instance_uids': ['abc', 123]},),
      ({'instance_uids': ['abc', '']},),
      ({'patch_coordinates': []},),
      ({'patch_coordinates': [{}]},),
      ({'patch_coordinates': [{'x_origin': '1', 'y_origin': 1}]},),
      ({'patch_coordinates': [{'x_origin': 1, 'y_origin': '1'}]},),
      ({'patch_coordinates': [{'x_origin': 1, 'y_origin': 1.2}]},),
      ({'patch_coordinates': [{'x_origin': 1.2, 'y_origin': 1}]},),
  ])
  def test_v1_dicom_embedding_response_bad_value_raises(self, param):
    test_dict = dict(
        dicom_web_store_url='foo',
        dicom_study_uid='1.2.3',
        dicom_series_uid='1.2.4',
        bearer_token='ABCD',
        ez_wsi_state='',
        instance_uids=['1.2', '1.3'],
        patch_coordinates=[
            dict(x_origin=3, y_origin=4, width=224, height=224),
            dict(x_origin=5, y_origin=6, width=224, height=224),
        ],
    )
    test_dict.update(param)
    instances = {
        'parameters': {
            'model_size': 'SMALL',
            'model_kind': 'LOW_PIXEL_SPACING',
        },
        'instances': [test_dict],
    }
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      embedding_converter.EmbeddingConverterV1().json_to_embedding_request(
          instances
      )

  @parameterized.parameters([
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 223, 'height': 224}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 225, 'height': 224}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 223}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 225}
              ]
          },
      ),
  ])
  def test_v1_dicom_embedding_patch_patch_dim_raises(self, param):
    test_dict = dict(
        dicom_web_store_url='foo',
        dicom_study_uid='1.2.3',
        dicom_series_uid='1.2.4',
        bearer_token='ABCD',
        ez_wsi_state='',
        instance_uids=['1.2', '1.3'],
        patch_coordinates=[
            dict(x_origin=3, y_origin=4, width=224, height=224),
            dict(x_origin=5, y_origin=6, width=224, height=224),
        ],
    )
    test_dict.update(param)
    instances = {
        'parameters': {
            'model_size': 'SMALL',
            'model_kind': 'LOW_PIXEL_SPACING',
        },
        'instances': [test_dict],
    }
    with self.assertRaises(
        pete_errors.PatchDimensionsDoNotMatchEndpointInputDimensionsError
    ):
      embedding_converter.EmbeddingConverterV1().json_to_embedding_request(
          instances
      )

  @parameterized.parameters([
      ({'series_path': ''},),
      ({'series_path': 123},),
      ({'bearer_token': 123},),
      ({'extensions': 123},),
      ({'extensions': '123'},),
      ({'instance_uids': 123},),
      ({'instance_uids': [123]},),
      ({'instance_uids': []},),
      ({'instance_uids': ['abc', 123]},),
      ({'instance_uids': ['abc', '']},),
      ({'patch_coordinates': []},),
      ({'patch_coordinates': [{}]},),
      ({'patch_coordinates': [{'x_origin': '1', 'y_origin': 1}]},),
      ({'patch_coordinates': [{'x_origin': 1, 'y_origin': '1'}]},),
      ({'patch_coordinates': [{'x_origin': 1, 'y_origin': 1.2}]},),
      ({'patch_coordinates': [{'x_origin': 1.2, 'y_origin': 1}]},),
  ])
  def test_dicom_embedding_v2_response_bad_value_raises(self, param):
    test_dict = dict(
        series_path='foo',
        bearer_token='ABCD',
        extensions={'EGF': 'QRS'},
        instance_uids=['1.2', '1.3'],
        patch_coordinates=[
            dict(x_origin=3, y_origin=4, width=224, height=224),
            dict(x_origin=5, y_origin=6, width=224, height=224),
        ],
    )
    test_dict.update(param)
    instances = {'instances': [test_dict]}
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
          instances
      )

  @parameterized.parameters([
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 223, 'height': 224}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 225, 'height': 224}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 223}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 225}
              ]
          },
      ),
  ])
  def test_dicom_v2_embedding_response_bad_patch_dim_value_raises(self, param):
    test_dict = dict(
        series_path='foo',
        bearer_token='ABCD',
        extensions={'EGF': 'QRS'},
        instance_uids=['1.2', '1.3'],
        patch_coordinates=[
            dict(x_origin=3, y_origin=4, width=224, height=224),
            dict(x_origin=5, y_origin=6, width=224, height=224),
        ],
    )
    test_dict.update(param)
    instances = {'instances': [test_dict]}
    with self.assertRaises(
        pete_errors.PatchDimensionsDoNotMatchEndpointInputDimensionsError
    ):
      embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
          instances
      )

  def test_dicom_embedding_unidentified_type(self):
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      embedding_converter.EmbeddingConverterV2().json_to_embedding_request({
          'instances': [{
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 224}
              ]
          }]
      })

  @parameterized.parameters([
      ({'image_file_uri': ''},),
      ({'image_file_uri': 123},),
      ({'bearer_token': 123},),
      ({'extensions': 123},),
      ({'extensions': '123'},),
      ({'patch_coordinates': []},),
      ({'patch_coordinates': [{}]},),
      ({'patch_coordinates': [{'x_origin': '1', 'y_origin': 1}]},),
      ({'patch_coordinates': [{'x_origin': 1, 'y_origin': '1'}]},),
      ({'patch_coordinates': [{'x_origin': 1, 'y_origin': 1.2}]},),
      ({'patch_coordinates': [{'x_origin': 1.2, 'y_origin': 1}]},),
  ])
  def test_gcs_image_embedding_response_bad_value_raises(self, param):
    test_dict = dict(
        image_file_uri='foo',
        bearer_token='ABCD',
        extensions={'EGF': 'QRS'},
        patch_coordinates=[
            dict(x_origin=3, y_origin=4, width=224, height=224),
            dict(x_origin=5, y_origin=6, width=224, height=224),
        ],
    )
    test_dict.update(param)
    instances = {'instances': [test_dict]}
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
          instances
      )

  @parameterized.parameters([
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 223, 'height': 224}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 225, 'height': 224}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 223}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 225}
              ]
          },
      ),
  ])
  def test_gcs_image_embedding_bad_patch_dim_raises(self, param):
    test_dict = dict(
        image_file_uri='foo',
        bearer_token='ABCD',
        extensions={'EGF': 'QRS'},
        patch_coordinates=[
            dict(x_origin=3, y_origin=4, width=224, height=224),
            dict(x_origin=5, y_origin=6, width=224, height=224),
        ],
    )
    test_dict.update(param)
    instances = {'instances': [test_dict]}
    with self.assertRaises(
        pete_errors.PatchDimensionsDoNotMatchEndpointInputDimensionsError
    ):
      embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
          instances
      )

  @parameterized.parameters([
      ({'raw_image_bytes': ''},),
      ({'raw_image_bytes': 123},),
      ({'extensions': 123},),
      ({'extensions': '123'},),
      ({'patch_coordinates': []},),
      ({'patch_coordinates': [{}]},),
      ({'patch_coordinates': [{'x_origin': '1', 'y_origin': 1}]},),
      ({'patch_coordinates': [{'x_origin': 1, 'y_origin': '1'}]},),
      ({'patch_coordinates': [{'x_origin': 1, 'y_origin': 1.2}]},),
      ({'patch_coordinates': [{'x_origin': 1.2, 'y_origin': 1}]},),
  ])
  def test_image_bytes_embedding_response_bad_value_raises(self, param):
    test_dict = dict(
        raw_image_bytes='foo',
        extensions={'EGF': 'QRS'},
        patch_coordinates=[
            dict(x_origin=3, y_origin=4, width=224, height=224),
            dict(x_origin=5, y_origin=6, width=224, height=224),
        ],
    )
    test_dict.update(param)
    instances = {'instances': [test_dict]}
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
          instances
      )

  @parameterized.parameters([
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 223, 'height': 224}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 225, 'height': 224}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 223}
              ]
          },
      ),
      (
          {
              'patch_coordinates': [
                  {'x_origin': 0, 'y_origin': 0, 'width': 224, 'height': 225}
              ]
          },
      ),
  ])
  def test_image_bytes_invalid_patch_dim_raises(self, param):
    test_dict = dict(
        raw_image_bytes='foo',
        extensions={'EGF': 'QRS'},
        patch_coordinates=[
            dict(x_origin=3, y_origin=4, width=224, height=224),
            dict(x_origin=5, y_origin=6, width=224, height=224),
        ],
    )
    test_dict.update(param)
    instances = {'instances': [test_dict]}
    with self.assertRaises(
        pete_errors.PatchDimensionsDoNotMatchEndpointInputDimensionsError
    ):
      embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
          instances
      )

  def test_v2_multiple_sources(self):
    instances = {
        'instances': [
            dict(
                dicom_path=dict(series_path='foo', instance_uids=['1.2']),
                patch_coordinates=[dict(x_origin=3, y_origin=4)],
            ),
            dict(
                image_file_uri='foo',
                patch_coordinates=[dict(x_origin=3, y_origin=4)],
            ),
            dict(
                raw_image_bytes='foo',
                patch_coordinates=[dict(x_origin=3, y_origin=4)],
            ),
        ]
    }
    result = (
        embedding_converter.EmbeddingConverterV2().json_to_embedding_request(
            instances
        )
    )
    self.assertEqual(
        result,
        embedding_request.EmbeddingRequestV2([
            embedding_request.DicomImageV2(
                'foo',
                '',
                {},
                ['1.2'],
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=4, width=224, height=224
                    )
                ],
            ),
            embedding_request.GcsImageV2(
                'foo',
                '',
                {},
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=4, width=224, height=224
                    )
                ],
            ),
            embedding_request.EmbeddedImageV2(
                'foo',
                {},
                [
                    patch_coordinate.PatchCoordinate(
                        x_origin=3, y_origin=4, width=224, height=224
                    )
                ],
            ),
        ]),
    )

  def test_return_error_response(self):
    self.assertEqual(
        embedding_converter.embedding_response_v1_to_json(
            embedding_response.EmbeddingResponseV1(
                '1233',
                embedding_response.PeteErrorResponse(
                    embedding_response.ErrorCode.INVALID_RESPONSE_ERROR
                ),
                [],
            )
        ),
        {
            'predictions': {
                'model_version': '1233',
                'error_response': {'error_code': 'INVALID_RESPONSE_ERROR'},
                'embedding_result': [],
            }
        },
    )

  def test_get_patch_coord_cast_float_to_int(self):
    self.assertEqual(
        embedding_converter._get_patch_coord([
            dict(x_origin=3.0, y_origin=4.0, width=224.0, height=224.0),
        ]),
        [
            patch_coordinate.PatchCoordinate(
                x_origin=3, y_origin=4, width=224, height=224
            ),
        ],
    )

  @parameterized.parameters([1, 'abc', ({},)])
  def test_get_patch_coord_not_passed_list_raises(self, val):
    with self.assertRaises(embedding_converter._InvalidCoordinateError):
      embedding_converter._get_patch_coord(
          typing.cast(List[Mapping[str, Any]], val)
      )

  def test_get_patch_coord_not_passed_list_of_dict_raises(self):
    with self.assertRaises(embedding_converter._InvalidCoordinateError):
      embedding_converter._get_patch_coord([[]])

  def test_embedding_converter_v1_missing_key_raises(self):
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      embedding_converter.EmbeddingConverterV1().json_to_embedding_request({})

  def test_embedding_converter_v2_missing_key_raises(self):
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      embedding_converter.EmbeddingConverterV2().json_to_embedding_request({})

  @parameterized.parameters([1, 'abc', ({1: 'abc'},)])
  def test_validate_str_key_dict_raises(self, val):
    with self.assertRaises(embedding_converter.ValidationError):
      embedding_converter.validate_str_key_dict(val)

  @parameterized.parameters([({},), ({'abc': 123},), ({'abc': 'efg'},)])
  def test_validate_str_key_dict_succeeds(self, val):
    self.assertEqual(val, embedding_converter.validate_str_key_dict(val))

  def test_embedding_response_to_json_v2(self):
    self.assertEqual(
        embedding_converter.embedding_response_v2_to_json(['a', 'b', 'c']),
        {'predictions': ['a', 'b', 'c']},
    )

  def test_generate_instance_metadata_error_string(self):
    test_input = {
        'a': 1,
        _EndpointJsonKeys.EXTENSIONS: {
            'b': 2,
            _EndpointJsonKeys.EZ_WSI_STATE: 3,
        },
        _EndpointJsonKeys.BEARER_TOKEN: 'abcd',
    }
    self.assertEqual(
        embedding_converter._generate_instance_metadata_error_string(
            test_input,
            _EndpointJsonKeys.EXTENSIONS,
            'a',
            _EndpointJsonKeys.BEARER_TOKEN,
        ),
        json.dumps(
            {
                'a': 1,
                _EndpointJsonKeys.EXTENSIONS: {'b': 2},
                _EndpointJsonKeys.BEARER_TOKEN: 'PRESENT',
            },
            sort_keys=True,
        ),
    )


if __name__ == '__main__':
  absltest.main()
