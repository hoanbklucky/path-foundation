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

"""Mappings between errors in python and error_codes returned in API responses."""

import pete_errors
from data_models import embedding_response


_ERROR_MAPPINGS = {
    pete_errors.InvalidRequestFieldError: (
        embedding_response.ErrorCode.INVALID_REQUEST_FIELD_ERROR
    ),
    pete_errors.InvalidResponseError: (
        embedding_response.ErrorCode.INVALID_RESPONSE_ERROR
    ),
    pete_errors.InstancesNotConcatenatedError: (
        embedding_response.ErrorCode.INSTANCES_NOT_CONCATENATED_ERROR
    ),
    pete_errors.InvalidCredentialsError: (
        embedding_response.ErrorCode.INVALID_CREDENTIALS_ERROR
    ),
    pete_errors.TooManyPatchesError: (
        embedding_response.ErrorCode.TOO_MANY_PATCHES_ERROR
    ),
    pete_errors.LevelNotFoundError: (
        embedding_response.ErrorCode.LEVEL_NOT_FOUND_ERROR
    ),
    pete_errors.EzWsiStateError: (
        embedding_response.ErrorCode.EZ_WSI_STATE_ERROR
    ),
    pete_errors.PatchOutsideOfImageDimensionsError: (
        embedding_response.ErrorCode.PATCH_OUTSIDE_OF_IMAGE_DIMENSIONS_ERROR
    ),
    pete_errors.ImageError: embedding_response.ErrorCode.IMAGE_ERROR,
    pete_errors.HttpError: embedding_response.ErrorCode.HTTP_ERROR,
    pete_errors.InvalidIccProfileTransformError: (
        embedding_response.ErrorCode.INVALID_ICC_PROFILE_TRANSFORM_ERROR
    ),
    pete_errors.ImageDimensionError: (
        embedding_response.ErrorCode.IMAGE_DIMENSION_ERROR
    ),
    pete_errors.DicomTiledFullError: (
        embedding_response.ErrorCode.DICOM_TILED_FULL_ERROR
    ),
    pete_errors.DicomError: embedding_response.ErrorCode.DICOM_ERROR,
    pete_errors.DicomImageDownsamplingTooLargeError: (
        embedding_response.ErrorCode.DICOM_IMAGE_DOWNSAMPLING_TOO_LARGE_ERROR
    ),
    pete_errors.DicomPathError: embedding_response.ErrorCode.DICOM_PATH_ERROR,
    pete_errors.GcsImagePathFormatError: (
        embedding_response.ErrorCode.GCS_IMAGE_PATH_FORMAT_ERROR
    ),
    pete_errors.UnapprovedDicomStoreError: (
        embedding_response.ErrorCode.UNAPPROVED_DICOM_STORE_ERROR
    ),
    pete_errors.UnapprovedGcsBucketError: (
        embedding_response.ErrorCode.UNAPPROVED_GCS_BUCKET_ERROR
    ),
    pete_errors.PatchDimensionsDoNotMatchEndpointInputDimensionsError: (
        embedding_response.ErrorCode.PATCH_DIMENSIONS_DO_NOT_MATCH_ENDPOINT_INPUT_DIMENSIONS_ERROR
    ),
}


def get_error_code(
    error: pete_errors.PeteError,
) -> embedding_response.ErrorCode:
  """Maps PeteErrors to ERROR_CODES."""
  return _ERROR_MAPPINGS[type(error)]
