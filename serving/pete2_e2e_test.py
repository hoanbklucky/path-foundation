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

"""E2E tests for ez_wsi_dicomweb -> Pete and back."""

from collections.abc import Sequence
import os
import shutil

from absl.testing import absltest
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import local_image
from ez_wsi_dicomweb import patch_embedding
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import pydicom
import requests_mock

from health_foundations.path_foundation.serving import pete_predictor_v2
from health_foundations.path_foundation.serving.test_utils import pete_mock
from health_foundations.path_foundation.serving.test_utils import test_files
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


def _round(embeddings: Sequence[float], decimals: int = 3) -> Sequence[float]:
  return [round(e, decimals) for e in embeddings]


class EzWsiPete2E2eTest(absltest.TestCase):

  @requests_mock.Mocker()
  def test_ez_wsi_dicom_embeddings(self, mock_request):
    instance = pydicom.dcmread(
        test_files.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{test_files.TEST_STORE_PATH}/dicomWeb/studies/{instance.StudyInstanceUID}/series/{instance.SeriesInstanceUID}'
    )
    store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(
        store_path, mock_request=mock_request
    ) as mock_store:
      mock_store[store_path].add_instance(instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.DefaultCredentialFactory()
          ),
          series_path,
      )
      endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
      with pete_predictor_v2.PetePredictor() as predictor:
        pete_mock.EndpointMock(mock_request, endpoint.end_point_url, predictor)
        patch = slide.get_patch(slide.native_level, 0, 0, 224, 224)
        embedding = patch_embedding.get_patch_embedding(endpoint, patch)
        self.assertEqual(_round(embedding.tolist(), 3), [0.775, 0.716, 0.827])

  @requests_mock.Mocker()
  def test_ez_wsi_gcs_embeddings(self, mock_request):
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('dcm_frame_1.jpg'),
        os.path.join(temp_dir, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          image_dimensions=gcs_image.ImageDimensions(224, 224),
          credential_factory=credential_factory.NoAuthCredentialsFactory(),
      )
      endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
          credential_factory=credential_factory.NoAuthCredentialsFactory()
      )
      with pete_predictor_v2.PetePredictor() as predictor:
        pete_mock.EndpointMock(mock_request, endpoint.end_point_url, predictor)
        patch = image.get_patch(0, 0, 224, 224)
        embedding = patch_embedding.get_patch_embedding(endpoint, patch)
        self.assertEqual(_round(embedding.tolist(), 3), [0.776, 0.713, 0.826])

  @requests_mock.Mocker()
  def test_ez_wsi_local_image(self, mock_request):
    mem = np.zeros((224, 224), dtype=np.uint8)
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
        credential_factory=credential_factory.NoAuthCredentialsFactory()
    )
    with pete_predictor_v2.PetePredictor() as predictor:
      pete_mock.EndpointMock(mock_request, endpoint.end_point_url, predictor)
      image = local_image.LocalImage(mem)
      patch = image.get_patch(0, 0, 224, 224)
      embedding = patch_embedding.get_patch_embedding(endpoint, patch)
      self.assertEqual(_round(embedding.tolist(), 3), [0.0, 0.0, 0.0])


if __name__ == '__main__':
  absltest.main()
