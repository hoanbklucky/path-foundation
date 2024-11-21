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

"""Callable responsible for running Inference on provided patches."""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import math
import threading
import time
import types
import typing
from typing import Any, List, Mapping, Optional, Union

from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb import local_dicom_slide_cache_types
from ez_wsi_dicomweb import local_image
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
from PIL import ImageCms

from serving_framework import model_runner
import abstract_pete_predictor
import pete_error_mapping
import pete_errors
import pete_flags
import pete_icc_profile_cache
import pete_logging
from data_models import embedding_converter
from data_models import embedding_request
from data_models import embedding_response
from data_models import patch_coordinate as patch_coordinate_module
from logging_lib import cloud_logging_client


_EndpointJsonKeys = patch_embedding_endpoints.EndpointJsonKeys

_UINT8_MAX_VALUE = 255.0
_MAX_PATCHES_PER_BATCH = 3000
_MAX_PATCHES_PER_REQUEST = 4000  # Avoid exceeding Vertex response limit
_MAX_RESIZE_DIMENSION_FOR_GCS_IMAGE_AND_EMBEDDED_IMAGE = 10000
_MAX_DICOM_LEVEL_DOWNSAMPLE = 8.0
_TRUE = 'TRUE'
_FALSE = 'FALSE'
_MAX_INSTANCE_READ_THREADS = 4


def _validate_gcs_image_source(uri: str) -> None:
  gcs_source_list = pete_flags.APPROVED_GCS_SOURCE_LIST_FLAG.value
  if gcs_source_list is None:
    return
  test_path = uri.lower()
  for val in gcs_source_list:
    if test_path.startswith(val.lower()):
      return
  msg = f'GCS source {uri} is not in the allowed list.'
  cloud_logging_client.info(msg, {'connection_attempted': uri})
  raise pete_errors.UnapprovedGcsBucketError(msg)


def _validate_dicom_image_source(path: str) -> None:
  dicom_source_list = pete_flags.APPROVED_DICOM_STORE_SOURCE_LIST_FLAG.value
  if dicom_source_list is None:
    return
  test_path = path.lower()
  for val in dicom_source_list:
    if test_path.startswith(val.lower()):
      return
  msg = f'DICOM store {str(path)} is not in the allowed list.'
  cloud_logging_client.info(msg, {'connection_attempted': path})
  raise pete_errors.UnapprovedDicomStoreError(msg)


class _RequestPatchCountSizeMonitor:
  """Monitors total number of patch performed during request."""

  def __init__(self, request: embedding_request.EmbeddingRequestV2):
    """Counts the total number of patch_coordinates objects in an request."""
    self._lock = threading.Lock()
    total_patches = 0
    for instance in request.instances:
      if (
          isinstance(instance, embedding_request.DicomImageV2)
          or isinstance(instance, embedding_request.GcsImageV2)
          or isinstance(instance, embedding_request.EmbeddedImageV2)
      ):
        total_patches += len(instance.patch_coordinates)
      else:
        # should never happen unless source code is being refactored.
        msg = 'Unexpected class passed to request patch count size monitor.'
        cloud_logging_client.error(msg)
        raise pete_errors.InternalBugError(msg)
    self._total_patches = total_patches
    self._test_patch_count()

  @property
  def count(self) -> int:
    with self._lock:
      return self._total_patches

  def _test_patch_count(self):
    """Raises if the number of patches being requested exceeds endpoint theshold."""
    if self._total_patches > _MAX_PATCHES_PER_REQUEST:
      msg = (
          'The number of patch embeddings exceeds the maximum allowed by the'
          ' endpoint. Reduce the total number of patches in the request.'
      )
      cloud_logging_client.info(
          msg,
          {
              'patches_requested': self._total_patches,
              'max_patches_per_request': _MAX_PATCHES_PER_REQUEST,
          },
      )
      raise pete_errors.TooManyPatchesError(msg)

  def add_patch_count(self, additional_patches: int) -> None:
    """Updates patch request size and checks against threshold."""
    with self._lock:
      self._total_patches += additional_patches
      self._test_patch_count()


def _get_required_fully_in_source_image_extension(
    extensions: Mapping[str, Any],
) -> bool:
  """Returns true (default) if patches are required to be fully in image."""
  value = extensions.get(
      _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE,
      True,
  )
  if isinstance(value, str):
    value = value.upper()
    if value == _TRUE:
      return True
    elif value == _FALSE:
      return False
  if not isinstance(value, bool):
    msg = (
        f'{_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE} is'
        ' not a boolean.'
    )
    cloud_logging_client.info(msg)
    raise pete_errors.InvalidRequestFieldError(msg)
  return value


def _get_resize_image_dimensions(
    extensions: Mapping[str, Any], max_dimension: Optional[int] = None
) -> Optional[dicom_slide.ImageDimensions]:
  """Returns optional dimension to resize input imaging level to."""
  value = extensions.get(_EndpointJsonKeys.IMAGE_DIMENSIONS, {})
  if not value:
    return None
  try:
    image_dim = dicom_slide.ImageDimensions(**value)
  except TypeError as exp:
    if not isinstance(value, dict):
      msg = f'{_EndpointJsonKeys.IMAGE_DIMENSIONS} is not a dictionary.'
      cloud_logging_client.info(msg, exp)
      raise pete_errors.InvalidRequestFieldError(msg) from exp
    keys = ', '.join(
        list(dataclasses.asdict(dicom_slide.ImageDimensions(0, 0)))
    )
    msg = (
        f'{_EndpointJsonKeys.IMAGE_DIMENSIONS} dict'
        f' has invalid keys; expecting: {keys}'
    )
    cloud_logging_client.info(msg, exp)
    raise pete_errors.InvalidRequestFieldError(msg) from exp
  try:
    width = embedding_converter.validate_int(image_dim.width_px)
    height = embedding_converter.validate_int(image_dim.height_px)
  except embedding_converter.ValidationError as exp:
    msg = 'Invalid dimensions(width and/or height).'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.InvalidRequestFieldError(msg) from exp
  if height <= 0 or width <= 0:
    msg = 'Image dimensions(width and/or height) are not positive integers.'
    cloud_logging_client.info(msg)
    raise pete_errors.InvalidRequestFieldError(msg)
  if max_dimension is not None and (
      height > max_dimension or width > max_dimension
  ):
    msg = (
        f'Image width and/or height exceeds max dimension ({max_dimension} px)'
        ' supported by the endpoint.'
    )
    cloud_logging_client.info(msg)
    raise pete_errors.ImageDimensionError(msg)
  return image_dim


def _get_ez_wsi_state(extensions: Mapping[str, Any]) -> Mapping[str, Any]:
  """Returns optional state for EZ-WSI DICOM web."""
  try:
    return embedding_converter.validate_str_key_dict(
        extensions.get(_EndpointJsonKeys.EZ_WSI_STATE, {})
    )
  except embedding_converter.ValidationError as exp:
    msg = 'Invalid EZ-WSI state metadata.'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.EzWsiStateError(msg) from exp


def _get_transform_imaging_to_icc_profile(extensions: Mapping[str, Any]) -> str:
  """Returns optional state for EZ-WSI DICOM web."""
  state = extensions.get(
      _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE,
      'NONE',
  )
  if not isinstance(state, str):
    msg = (
        f'{_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE} value'
        ' is not a string.'
    )
    cloud_logging_client.info(msg)
    raise pete_errors.InvalidRequestFieldError(msg)
  return state.upper()


def _validate_server_and_client_side_icc_profile_normalization(
    server_side_norm: str,
    icc_profile_transform: Optional[ImageCms.ImageCmsTransform],
    client_side_normalization: str,
) -> None:
  """Validates that client and server use same icc profile transfrom."""
  if not client_side_normalization or client_side_normalization == 'NONE':
    return
  if server_side_norm != client_side_normalization:
    # Client side ICC profile transformation is only performed using EZ-WSI
    # DICOM web. If this occurs it would be a bug in the ez-wsi interface.
    msg = (
        'Error different ICC profile transformations defined by client and'
        ' server.'
    )
    cloud_logging_client.error(msg)
    raise pete_errors.EzWsiStateError(msg)
  if icc_profile_transform is not None:
    # Client side ICC profile transformation is only performed using EZ-WSI
    # DICOM web. If this occurs it would be a bug in the ez-wsi interface.
    msg = 'Duplicate ICC profile transformations defined by client and server.'
    cloud_logging_client.error(msg)
    raise pete_errors.EzWsiStateError(msg)


def _get_icc_profile(
    extensions: Mapping[str, Any],
) -> Optional[ImageCms.core.CmsProfile]:
  """Returns optional state for EZ-WSI DICOM web."""
  state = _get_transform_imaging_to_icc_profile(extensions)
  if state == 'NONE':
    return None
  if state == 'ADOBERGB':
    return dicom_slide.get_adobergb_icc_profile()
  if state == 'ROMMRGB':
    return dicom_slide.get_rommrgb_icc_profile()
  if state == 'SRGB':
    return dicom_slide.get_srgb_icc_profile()
  msg = (
      f'{_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE} value'
      ' is not valid; expecting: ADOBERGB, ROMMRGB, SRGB, or NONE.'
  )
  cloud_logging_client.info(msg)
  raise pete_errors.InvalidIccProfileTransformError(msg)


def _fetch_patch_bytes(
    patch: Union[dicom_slide.DicomPatch, gcs_image.GcsPatch],
    icc_profile_transformation: Optional[ImageCms.ImageCmsTransform],
) -> np.ndarray:
  """Returns patch bytes for GCS imaging."""
  try:
    normalized_patch = (
        patch.image_bytes(icc_profile_transformation).astype(np.float32)
        / _UINT8_MAX_VALUE
    )
  except (
      ez_wsi_errors.HttpForbiddenError,
      ez_wsi_errors.HttpUnauthorizedError,
  ) as exp:
    if isinstance(patch.source, gcs_image.GcsImage):
      msg = (
          'Invalid credentials reading image bytes from google cloud storage'
          f' image: {patch.source.uri}'
      )
    elif isinstance(patch.source, dicom_slide.DicomSlide):
      msg = f'Invalid credentials reading DICOM image: {str(patch.source.path)}'
    else:
      msg = 'Invalid credentials accessing unrecognized data source'
    msg = f'{msg}; HTTP status code: {exp.status_code}; reason: {exp.reason}.'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.InvalidCredentialsError(msg) from exp
  except ez_wsi_errors.HttpError as exp:
    if isinstance(patch.source, gcs_image.GcsImage):
      msg = f'HTTP error occurred accessing GCS image: {patch.source.uri}'
    elif isinstance(patch.source, dicom_slide.DicomSlide):
      msg = (
          f'HTTP error occurred accessing DICOM image: {str(patch.source.path)}'
      )
    else:
      msg = 'HTTP error Coccurred accessing unrecognized data source'
    cloud_logging_client.info(msg, exp)
    msg = f'{msg}; HTTP status code: {exp.status_code}; reason: {exp.reason}.'
    raise pete_errors.HttpError(msg) from exp
  return np.expand_dims(
      patch_embedding_endpoints.normalized_patch_channels(
          pete_flags.ENDPOINT_INPUT_WIDTH_FLAG.value,
          pete_flags.ENDPOINT_INPUT_HEIGHT_FLAG.value,
          normalized_patch,
      ),
      axis=0,
  )


def _get_normalized_patches(
    patches: List[Union[dicom_slide.DicomPatch, gcs_image.GcsPatch]],
    icc_profile_transformation: Optional[ImageCms.ImageCmsTransform],
) -> np.ndarray:
  """Gets normalized patches from DICOM slide."""
  if icc_profile_transformation is not None:
    cloud_logging_client.info(
        'Transforming RGB values in patches to target ICC Profile.'
    )
  return np.concatenate(
      [
          _fetch_patch_bytes(patch, icc_profile_transformation)
          for patch in patches
      ],
      axis=0,
  )


def _get_dicom_patches(
    instance: embedding_request.DicomImageV2,
    patch_count_monitor: _RequestPatchCountSizeMonitor,
) -> np.ndarray:
  """Returns image patch bytes from DICOM series."""
  cloud_logging_client.info('Generating embedding from DICOM.')
  require_fully_in_source_image = _get_required_fully_in_source_image_extension(
      instance.extensions
  )
  resize_level_dim = _get_resize_image_dimensions(instance.extensions)
  ez_wsi_state = _get_ez_wsi_state(instance.extensions)
  target_icc_profile = _get_icc_profile(instance.extensions)
  if instance.bearer_token:
    dcf = credential_factory.TokenPassthroughCredentialFactory(
        instance.bearer_token
    )
  else:
    dcf = credential_factory.NoAuthCredentialsFactory()
  dwi = dicom_web_interface.DicomWebInterface(dcf)

  try:
    path = dicom_path.FromPath(
        dicom_path.FromString(instance.series_path),
        instance_uid='',
    )
  except ValueError as exp:
    msg = f'DICOM path is invalid: {instance.series_path}.'
    cloud_logging_client.info(
        msg,
        {'slide_path_requested': instance.series_path},
        exp,
    )
    raise pete_errors.DicomPathError(msg) from exp
  if not path.series_uid or not path.study_uid:
    msg = f'DICOM path is invalid: {instance.series_path}.'
    cloud_logging_client.info(
        msg, {'slide_path_requested': instance.series_path}
    )
    raise pete_errors.DicomPathError(
        f'Slide path is invalid: {instance.series_path}.'
    )
  # Load DICOM Slide only if slide path changes. Avoid unneeded
  # state requests and optionally init pyramid level metadata from
  # parameter to avoid DICOM store slide metadata query.
  _validate_dicom_image_source(str(path))
  try:
    start_time = time.time()
    series_images = dicom_slide.DicomMicroscopeSeries(
        dwi=dwi,
        path=path,
        enable_client_slide_frame_decompression=True,
        json_metadata=ez_wsi_state,
        instance_uids=instance.instance_uids,
        logging_factory=pete_logging.EZWSILoggingInterfaceFactory(
            cloud_logging_client.get_log_signature()
        ),
    )
  except ez_wsi_errors.DicomSlideInitError as exp:
    msg = f'DICOM metadata error; Path: {path}; {exp}'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.DicomError(msg) from exp
  except (
      ez_wsi_errors.SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError
  ) as exp:
    msg = (
        'All DICOM instances in a pyramid level are required to have same'
        ' TransferSyntaxUID.'
    )
    cloud_logging_client.info(msg, exp)
    raise pete_errors.DicomError(msg) from exp
  except (
      ez_wsi_errors.DicomTagNotFoundError,
      ez_wsi_errors.InvalidDicomTagError,
  ) as exp:
    msg = f'DICOM instance missing required tags; Path: {path}; {exp}'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.DicomError(msg) from exp
  except ez_wsi_errors.UnexpectedDicomObjectInstanceError as exp:
    msg = 'DICOM metadata lacks SOP Instance UID.'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.DicomError(msg) from exp
  except (
      ez_wsi_errors.InvalidSlideJsonMetadataError,
      ez_wsi_errors.SlidePathDoesNotMatchJsonMetadataError,
  ) as exp:
    msg = 'Error decoding embedding request JSON metadata.'
    cloud_logging_client.error(
        msg,
        {'slide_path_requested': path},
        exp,
    )
    # If this occurs it would be a bug in the ez-wsi interface.
    raise pete_errors.EzWsiStateError(msg) from exp
  cloud_logging_client.info(
      f'Retrieved metadata for slide: {path};'
      f' {time.time() - start_time} (sec).',
      {'slide_path_requested': path},
  )
  if series_images.dicom_slide is not None:
    if series_images.dicom_microscope_image is not None:
      msg = (
          'Cannot return embeddings for DICOM'
          ' VL_Whole_Slide_Microscopy_Images and DICOM Microscopic_Images in'
          ' the same instance request; split patch request across multiple'
          ' instances.'
      )
      cloud_logging_client.info(msg)
      raise pete_errors.DicomError(msg)
    ds = series_images.dicom_slide
  elif series_images.dicom_microscope_image is not None:
    ds = series_images.dicom_microscope_image
  else:
    msg = f'Could not find DICOM imaging; path: {path}.'
    cloud_logging_client.info(msg)
    raise pete_errors.DicomPathError(msg)
  # Initialize in-memory cache on slide to store frames requested in batch.
  # Optimization hints = MINIMIZE_LATENCY or MINIMIZE_DICOM_STORE_QPM
  # MINIMIZE_LATENCY: Batch load frames async. If a frame is
  # requested that is not in cache, issue an immediate request and return
  # data for the missing frame.
  # MINIMIZE_DICOM_STORE_QPM: Block, wait for cache to finish loading before
  # returning frames.

  ds.init_slide_frame_cache(
      optimization_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM
  )
  if len(instance.instance_uids) > 1 and not ds.are_instances_concatenated(
      instance.instance_uids
  ):
    msg = f'Instances {instance.instance_uids} are not concatenated.'
    cloud_logging_client.info(
        msg,
        {'dicom_instance_uids': instance.instance_uids},
    )
    raise pete_errors.InstancesNotConcatenatedError(msg)
  level = ds.get_instance_level(instance.instance_uids[0])
  if level is None:
    msg = (
        f'{instance.instance_uids[0]} is not part of DICOM WSI pyramid; path:'
        f' {path}.'
    )
    cloud_logging_client.info(msg)
    raise pete_errors.LevelNotFoundError(msg)
  if level.pixel_spacing.is_defined:
    level_pixel_spacing = (
        f'{level.pixel_spacing.column_spacing_mm},'
        f' {level.pixel_spacing.row_spacing_mm}'
    )
  else:
    level_pixel_spacing = 'undefined'
  cloud_logging_client.info(
      'Retrieved pyramid level for embedding generation.',
      {
          'level_index': level.level_index,
          'level_width': level.width,
          'level_height': level.height,
          'pixel_spacing': level_pixel_spacing,
          'frame_number_min': level.frame_number_min,
          'frame_number_max': level.frame_number_max,
          'transfer_syntax_uid': level.transfer_syntax_uid,
          'study_instance_uid': path.study_uid,
          'series_instance_uid': path.series_uid,
          'sop_instance_uids': level.get_level_sop_instance_uids(),
          _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
              require_fully_in_source_image
          ),
      },
  )
  if resize_level_dim is not None:
    cloud_logging_client.info(
        f'Resizing image level dimensions to: {resize_level_dim}'
    )
    downsample_scale_factor = max(level.scale_factors(resize_level_dim))
    if downsample_scale_factor > _MAX_DICOM_LEVEL_DOWNSAMPLE:
      msg = (
          f'Image downsampling, {round(downsample_scale_factor,5)}X,'
          ' exceeds 8x.'
      )
      cloud_logging_client.info(msg)
      raise pete_errors.DicomImageDownsamplingTooLargeError(msg)
    # maximum number of additional patch requests which could
    # occure based on scaling factor.
    patch_request_size_update = (math.ceil(downsample_scale_factor) ** 2) * len(
        instance.patch_coordinates
    )
    # cannot request more than total number of frames in level
    patch_request_size_update = min(
        patch_request_size_update, level.number_of_frames
    )
    # account for patches in current request
    patch_request_size_update = max(
        0, patch_request_size_update - len(instance.patch_coordinates)
    )
    patch_count_monitor.add_patch_count(patch_request_size_update)
    level = level.resize(resize_level_dim)
  # init patches from level
  try:
    patches = []
    for patch_coordinate in instance.patch_coordinates:
      try:
        patches.append(
            ds.get_patch(
                level,
                x=patch_coordinate.x_origin,
                y=patch_coordinate.y_origin,
                width=patch_coordinate.width,
                height=patch_coordinate.height,
                require_fully_in_source_image=require_fully_in_source_image,
            )
        )
      except ez_wsi_errors.PatchOutsideOfImageDimensionsError as exp:
        msg = (
            f'Patch dimensions {dataclasses.asdict(patch_coordinate)} fall'
            ' outside of DICOM level pyramid imaging dimensions'
            f' ({level.width} x {level.height}).'
        )
        cloud_logging_client.info(msg, exp)
        raise pete_errors.PatchOutsideOfImageDimensionsError(msg) from exp
      except ez_wsi_errors.DicomPatchGenerationError as exp:
        msg = (
            'Can not generate patches from DICOM instances with more than one'
            ' frame and Dimension Organization Type != TILED_FULL.'
        )
        cloud_logging_client.info(msg, exp)
        raise pete_errors.DicomTiledFullError(msg) from exp
    # Pre-fetch only the frames required for inference into the EZ-WSI frame
    # cache.
    fc = typing.cast(
        local_dicom_slide_cache.InMemoryDicomSlideCache, ds.slide_frame_cache
    )
    fc.reset_cache_stats()
    if target_icc_profile is None:
      ds.preload_patches_in_frame_cache(patches, blocking=True)
      icc_profile_transformation = None
    else:
      fc.optimization_hint = (
          local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY
      )
      ds.preload_patches_in_frame_cache(patches, blocking=False)
      try:
        dicom_slide_icc_profile = pete_icc_profile_cache.get_dicom_icc_profile(
            slide=ds, slide_level=level
        )
      except (
          ez_wsi_errors.HttpForbiddenError,
          ez_wsi_errors.HttpUnauthorizedError,
      ) as exp:
        msg = (
            'Failed to retrieve ICC profile from DICOM store. Invalid DICOM'
            f' store credentials; HTTP status code: {exp.status_code}; reason:'
            f' {exp.reason}.'
        )
        cloud_logging_client.info(msg, exp)
        raise pete_errors.InvalidCredentialsError(msg) from exp
      except ez_wsi_errors.HttpError as exp:
        msg = (
            'Failed to retrieve ICC profile from DICOM store. A HTTP error'
            f' occurred; HTTP status code: {exp.status_code}; reason:'
            f' {exp.reason}.'
        )
        cloud_logging_client.info(msg, exp)
        raise pete_errors.HttpError(msg) from exp
      if not dicom_slide_icc_profile:
        cloud_logging_client.info(
            'DICOM slide does not have ICC profile; imaging will not be'
            ' transformed to target ICC profile.'
        )
        icc_profile_transformation = None
      else:
        icc_profile_name = _get_transform_imaging_to_icc_profile(
            instance.extensions
        )
        cloud_logging_client.info(
            'Creating ICC profile transformation to transform RGB values in'
            f' DICOM patches to {icc_profile_name} colorspace.'
        )
        icc_profile_transformation = (
            dicom_slide.create_icc_profile_transformation(
                dicom_slide_icc_profile, target_icc_profile
            )
        )
      fc.block_until_frames_are_loaded()
      fc.optimization_hint = (
          local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM
      )
    result = _get_normalized_patches(patches, icc_profile_transformation)
    cloud_logging_client.debug(
        'DICOM image retrieval stats.', dataclasses.asdict(fc.cache_stats)
    )
    return result
  except (
      ez_wsi_errors.HttpForbiddenError,
      ez_wsi_errors.HttpUnauthorizedError,
  ) as exp:
    msg = (
        'Error retrieving DICOM patch imaging. Invalid DICOM store'
        f' credentials; HTTP status code: {exp.status_code}; reason:'
        f' {exp.reason}.'
    )
    cloud_logging_client.info(msg, exp)
    raise pete_errors.InvalidCredentialsError(msg) from exp
  except ez_wsi_errors.HttpError as exp:
    msg = (
        'Error retrieving DICOM patch imageing. A HTTP error occurred; HTTP'
        f' status code: {exp.status_code}; reason: {exp.reason}.'
    )
    cloud_logging_client.info(msg, exp)
    raise pete_errors.HttpError(msg) from exp
  except ez_wsi_errors.UnsupportedPixelFormatError as exp:
    msg = 'DICOM contains instances with imaging bits allocated != 8.'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.DicomError(msg) from exp
  except ez_wsi_errors.LevelNotFoundError as exp:
    msg = 'Cannot locate expected level. Request references invalid metadata.'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.LevelNotFoundError(msg) from exp
  except (
      ez_wsi_errors.CoordinateOutofImageDimensionsError,
      ez_wsi_errors.FrameNumberOutofBoundsError,
      ez_wsi_errors.InputFrameNumberOutOfRangeError,
      ez_wsi_errors.PatchIntersectionNotFoundError,
      ez_wsi_errors.SectionOutOfImageBoundsError,
  ) as exp:
    msg = (
        'Can not generate patches from DICOM instance. DICOM instance is'
        ' missing expected frames.'
    )
    cloud_logging_client.info(msg, exp)
    raise pete_errors.DicomError(msg) from exp
  except ez_wsi_errors.DicomInstanceReadError as exp:
    msg = 'Embedding request references a corrupt DICOM instance.'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.DicomError(msg) from exp


def _get_normalized_image_patches(
    image: Union[gcs_image.GcsImage, local_image.LocalImage, List[str]],
    instance: Union[
        embedding_request.GcsImageV2, embedding_request.EmbeddedImageV2
    ],
    require_fully_in_source_image: bool,
    icc_profile_transformation: Optional[ImageCms.ImageCmsTransform],
    image_dimensions: Optional[dicom_slide.ImageDimensions],
) -> np.ndarray:
  """Returns normalized patches for model input."""
  patches = []
  if isinstance(image, List):
    if len(image) != len(instance.patch_coordinates):
      # sanity check validate that coordinate and patch metadata have same
      # length; if this occurs it would be a bug in the ez-wsi interface.
      msg = 'Length of per-patch and patch coordinates data do not match.'
      cloud_logging_client.error(msg)
      raise pete_errors.EzWsiStateError(msg)
    for patch_metadata in image:
      try:
        cloud_logging_client.info('Request contains patch imaging metadata.')
        patch = gcs_image.GcsPatch.create_from_json(
            patch_metadata,
            require_fully_in_source_image,
            source_image_dimension=image_dimensions,
        )
      except ez_wsi_errors.PatchOutsideOfImageDimensionsError as exp:
        msg = 'Patch falls outside of image dimensions.'
        cloud_logging_client.info(msg, exp)
        raise pete_errors.PatchOutsideOfImageDimensionsError(msg) from exp
      except ez_wsi_errors.GcsImageError as exp:
        msg = f'Error occurred reading image; {exp}.'
        cloud_logging_client.info(msg, exp)
        raise pete_errors.ImageError(msg) from exp
      # Optimization remove image compressed bytes.
      # Bytes only needed to be retained to call embedding api.
      patch.source.clear_source_image_compressed_bytes()
      patches.append(patch)
  else:
    for patch_coordinate in instance.patch_coordinates:
      try:
        patches.append(
            image.get_patch(
                x=patch_coordinate.x_origin,
                y=patch_coordinate.y_origin,
                width=patch_coordinate.width,
                height=patch_coordinate.height,
                require_fully_in_source_image=require_fully_in_source_image,
            )
        )
      except (
          ez_wsi_errors.HttpForbiddenError,
          ez_wsi_errors.HttpUnauthorizedError,
      ) as exp:
        msg = (
            'Invalid Google Cloud Storage credentials; HTTP status code:'
            f' {exp.status_code}; reason: {exp.reason}.'
        )
        cloud_logging_client.info(msg, exp)
        raise pete_errors.InvalidCredentialsError(msg) from exp
      except ez_wsi_errors.HttpError as exp:
        msg = (
            'HTTP error occurred while retrieving imaging from Google Cloud'
            f' Storage; HTTP status code: {exp.status_code}; reason:'
            f' {exp.reason}.'
        )
        cloud_logging_client.info(msg, exp)
        raise pete_errors.HttpError(msg) from exp
      except ez_wsi_errors.PatchOutsideOfImageDimensionsError as exp:
        msg = (
            f'Patch dimensions {dataclasses.asdict(patch_coordinate)} fall'
            f' outside of image dimensions ({image.width} x {image.height}).'
        )
        cloud_logging_client.info(msg, exp)
        raise pete_errors.PatchOutsideOfImageDimensionsError(msg) from exp
  return _get_normalized_patches(patches, icc_profile_transformation)


def _create_image_icc_profile_transformation(
    instance: Union[
        embedding_request.GcsImageV2, embedding_request.EmbeddedImageV2
    ],
    image: Union[gcs_image.GcsImage, local_image.LocalImage],
) -> Optional[ImageCms.ImageCmsTransform]:
  """Creates transform to convert non-dicom imageing to target colorspace."""
  target_icc_profile_bytes = _get_icc_profile(instance.extensions)
  if not target_icc_profile_bytes:
    # Target name == 'NONE' or not set.
    return None
  icc_profile_transform = image.create_icc_profile_transformation(
      target_icc_profile_bytes
  )
  ez_wsi_state = _get_ez_wsi_state(instance.extensions)
  target_icc_profile_name = _get_transform_imaging_to_icc_profile(
      instance.extensions
  )
  if ez_wsi_state:
    try:
      client_side_icc_profile_normalization = embedding_converter.validate_str(
          ez_wsi_state.get(
              _EndpointJsonKeys.ICC_PROFILE_METADATA_NORMALIZATION,
              '',
          )
      ).upper()
    except embedding_converter.ValidationError as exp:
      # If this occurs it would be a bug in the ez-wsi interface.
      msg = 'Invalid ICC profile metadata parameter.'
      cloud_logging_client.error(msg, exp)
      raise pete_errors.EzWsiStateError(msg) from exp
    if client_side_icc_profile_normalization:
      _validate_server_and_client_side_icc_profile_normalization(
          target_icc_profile_name,
          icc_profile_transform,
          client_side_icc_profile_normalization,
      )
  else:
    client_side_icc_profile_normalization = ''
  if icc_profile_transform is None:
    # target name != 'NONE'
    if client_side_icc_profile_normalization == target_icc_profile_name:
      # When raw image bytes are sent, performing client side ICC profile
      # transformation is optimal. Image bytes are saved and sent to server
      # without an embedding ICC profile, reducing image payload size.
      cloud_logging_client.info(
          f'No {target_icc_profile_name} ICC profile transformation created.'
          f' Image bytes were transformed to {target_icc_profile_name} by'
          ' client and, as expected, sent without an embedding ICC profile.'
      )
    else:
      cloud_logging_client.info(
          f'No {target_icc_profile_name} ICC profile transformation created.'
          ' Image bytes do not have an embeedded ICC profile.'
      )
    return None
  cloud_logging_client.info(
      'Creating ICC profile transformation to transform RGB values in'
      f' image patches to {target_icc_profile_name} colorspace.'
  )
  return icc_profile_transform


def _get_gcs_patches(
    instance: embedding_request.GcsImageV2,
) -> np.ndarray:
  """Returns image patche bytes from a image on GCS."""
  cloud_logging_client.info('Generating embedding from GCS image.')
  require_fully_in_source_image = _get_required_fully_in_source_image_extension(
      instance.extensions
  )
  resize_level_dim = _get_resize_image_dimensions(
      instance.extensions,
      _MAX_RESIZE_DIMENSION_FOR_GCS_IMAGE_AND_EMBEDDED_IMAGE,
  )
  if resize_level_dim is not None:
    cloud_logging_client.info(f'Resizing imaging to: {resize_level_dim}')
  ez_wsi_state = _get_ez_wsi_state(instance.extensions)
  if instance.bearer_token:
    dcf = credential_factory.TokenPassthroughCredentialFactory(
        instance.bearer_token
    )
  else:
    dcf = credential_factory.NoAuthCredentialsFactory()
  try:
    image = None
    icc_profile_transform = None
    image_dimensions = None
    if ez_wsi_state:
      image_dimensions = _get_resize_image_dimensions(ez_wsi_state)
      try:
        patch_metadata = embedding_converter.validate_str_list(
            ez_wsi_state.get(_EndpointJsonKeys.PATCHES, [])
        )
      except embedding_converter.ValidationError as exp:
        msg = 'Invalid patch JSON metadata.'
        cloud_logging_client.error(msg, exp)
        # If this occurs it would be a bug in the ez-wsi interface.
        raise pete_errors.EzWsiStateError(msg) from exp
      try:
        image_metadata = embedding_converter.validate_str(
            ez_wsi_state.get(_EndpointJsonKeys.IMAGE, '')
        )
      except embedding_converter.ValidationError as exp:
        # If this occurs it would be a bug in the ez-wsi interface.
        msg = 'Invalid image JSON metadata.'
        cloud_logging_client.error(msg, exp)
        raise pete_errors.EzWsiStateError(msg) from exp
      try:
        patch_source_width = embedding_converter.validate_int(
            ez_wsi_state.get(
                _EndpointJsonKeys.SOURCE_IMAGE_WIDTH_PX,
                0,
            )
        )
        patch_source_height = embedding_converter.validate_int(
            ez_wsi_state.get(
                _EndpointJsonKeys.SOURCE_IMAGE_HEIGHT_PX,
                0,
            )
        )
      except embedding_converter.ValidationError as exp:
        msg = 'Invalid patch dimensions (width and/or height).'
        cloud_logging_client.info(msg, exp)
        raise pete_errors.InvalidRequestFieldError(msg) from exp
      if (
          image_dimensions is None
          and patch_source_width > 0
          and patch_source_height > 0
      ):
        image_dimensions = dicom_slide.ImageDimensions(
            patch_source_width, patch_source_height
        )
      if image_metadata:
        try:
          image = gcs_image.GcsImage.create_from_json(image_metadata)
          # Optimization remove image compressed bytes.
          # Bytes only needed to be retained to call embedding api.
          image.clear_source_image_compressed_bytes()
          cloud_logging_client.info('Request contains image metadata.')
          icc_profile_transform = _create_image_icc_profile_transformation(
              instance, image
          )
        except ez_wsi_errors.GcsImageError as exp:
          # If this occurs it would be a bug in the ez-wsi interface.
          msg = 'Invalid image JSON metadata.'
          cloud_logging_client.info(msg, exp)
          raise pete_errors.ImageError(msg) from exp
      elif patch_metadata:
        # Pass patch metadata in as the image.
        image = patch_metadata
    if image is None:
      _validate_gcs_image_source(instance.image_file_uri)
      cloud_logging_client.info('Retrieving imaging from GCS.')
      image = gcs_image.GcsImage(
          image_source=instance.image_file_uri,
          credential_factory=dcf,
          image_dimensions=resize_level_dim,
      )
      icc_profile_transform = _create_image_icc_profile_transformation(
          instance, image
      )
    return _get_normalized_image_patches(
        image,
        instance,
        require_fully_in_source_image,
        icc_profile_transform,
        image_dimensions,
    )
  except ez_wsi_errors.GcsImagePathFormatError as exp:
    msg = f'Invalid GCS URI: {instance.image_file_uri}'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.GcsImagePathFormatError(msg) from exp
  except (
      ez_wsi_errors.HttpForbiddenError,
      ez_wsi_errors.HttpUnauthorizedError,
  ) as exp:
    msg = (
        'Invalid Google Cloud Storage credentials; HTTP status code:'
        f' {exp.status_code}; reason: {exp.reason}.'
    )
    cloud_logging_client.info(msg, exp)
    raise pete_errors.InvalidCredentialsError(msg) from exp
  except ez_wsi_errors.HttpError as exp:
    msg = (
        'HTTP error occurred while retrieving imaging from Google Cloud'
        f' Storage; HTTP status code: {exp.status_code}; reason: {exp.reason}.'
    )
    cloud_logging_client.info(msg, exp)
    raise pete_errors.HttpError(msg) from exp
  except ez_wsi_errors.GcsImageError as exp:
    msg = f'Error occurred reading/decoding GCS image. {exp}'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.ImageError(msg) from exp


def _get_embedded_image_patches(
    instance: embedding_request.EmbeddedImageV2,
) -> np.ndarray:
  """Returns image patche bytes from image bytes passed directly to service."""
  cloud_logging_client.info('Generating embedding from embedded image.')
  require_fully_in_source_image = _get_required_fully_in_source_image_extension(
      instance.extensions
  )
  resize_level_dim = _get_resize_image_dimensions(
      instance.extensions,
      _MAX_RESIZE_DIMENSION_FOR_GCS_IMAGE_AND_EMBEDDED_IMAGE,
  )
  if resize_level_dim is not None:
    cloud_logging_client.info(f'Resizing imaging to: {resize_level_dim}')
  try:
    image = local_image.LocalImage(
        image_source=base64.b64decode(instance.image_bytes),
        image_dimensions=resize_level_dim,
    )
  except ez_wsi_errors.GcsImageError as exp:
    msg = f'Error occurred reading/decoding image. {exp}'
    cloud_logging_client.info(msg, exp)
    raise pete_errors.ImageError(msg) from exp
  icc_profile_transform = _create_image_icc_profile_transformation(
      instance, image
  )
  return _get_normalized_image_patches(
      image,
      instance,
      require_fully_in_source_image,
      icc_profile_transform,
      dicom_slide.ImageDimensions(image.width, image.height),
  )


def _get_instance_patches(
    instance: embedding_request.DicomImageV2,
    patch_count_monitor: _RequestPatchCountSizeMonitor,
) -> np.ndarray:
  """Loads patches described by instance and returns np.array of imaging."""
  if isinstance(instance, embedding_request.DicomImageV2):
    return _get_dicom_patches(instance, patch_count_monitor)
  elif isinstance(instance, embedding_request.GcsImageV2):
    return _get_gcs_patches(instance)
  elif isinstance(instance, embedding_request.EmbeddedImageV2):
    return _get_embedded_image_patches(instance)
  msg = 'Unspported instance type.'
  cloud_logging_client.error(msg)
  # If this occurs it would be a bug in the ez-wsi interface.
  raise pete_errors.InternalBugError(msg)


class PetePredictor(abstract_pete_predictor.AbstractPetePredictor):
  """Callable responsible for generating embeddings."""

  def __init__(self):
    super().__init__()
    self._thread_pool = None

  def __enter__(self) -> PetePredictor:
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=_MAX_INSTANCE_READ_THREADS
    )
    return self

  def __exit__(
      self,
      exc_type: Optional[type[BaseException]],
      exc_value: Optional[BaseException],
      traceback: Optional[types.TracebackType],
  ) -> None:
    if self._thread_pool is not None:
      self._thread_pool.shutdown(wait=False, cancel_futures=True)

  def predict(
      self,
      prediction_input: Mapping[str, Any],
      model: model_runner.ModelRunner,
  ) -> Mapping[str, Any]:
    """Runs inference on provided patches.

    Args:
      prediction_input: JSON formatted input for embedding prediction.
      model: ModelRunner to handle model step.

    Returns:
      JSON formatted output.

    Raises:
      ERROR_LOADING_DICOM: If the provided patches are not concated.
    """
    if self._thread_pool is None:
      msg = 'PetePredictor must be run in context mangaged block.'
      cloud_logging_client.error(msg)
      # If this occurs it would be a bug in the ez-wsi interface.
      raise pete_errors.InternalBugError(msg)
    embedding_json_converter = embedding_converter.EmbeddingConverterV2()
    request = embedding_json_converter.json_to_embedding_request(
        prediction_input
    )
    request_patch_count_monitor = _RequestPatchCountSizeMonitor(request)

    future_patches = [None] if request.instances else []
    # request imaging associated with each instance after first requested in
    # thread.
    for instance in request.instances[1:]:
      future_patches.append(
          self._thread_pool.submit(
              _get_instance_patches, instance, request_patch_count_monitor
          )
      )
    embedding_results = []
    normalized_patches = []
    instance_input_coordinates_or_errors: List[
        Union[
            List[patch_coordinate_module.PatchCoordinate],
            pete_errors.PeteError,
        ]
    ] = []
    # request input using multiple threads
    for instance, instance_future_patches in zip(
        request.instances, future_patches
    ):
      try:
        if instance_future_patches is None:
          # optmization requested first in inline in running thread
          normalized_patches.append(
              _get_instance_patches(instance, request_patch_count_monitor)
          )
        else:
          normalized_patches.append(
              instance_future_patches.result()
          )  # pytype: disable=attribute-error
        instance_input_coordinates_or_errors.append(instance.patch_coordinates)
      except pete_errors.PeteError as exp:
        instance_input_coordinates_or_errors.append(exp)
    del request  # try save some memory by freeing non-coord request metadata.
    if normalized_patches:
      # run all patches for all instances through model at once.
      normalized_patches = np.concatenate(normalized_patches, axis=0)
      num_batches = int(
          np.ceil(len(normalized_patches) / _MAX_PATCHES_PER_BATCH)
      )
      # Split the array into batches
      patch_batches = np.array_split(normalized_patches, num_batches)
      cloud_logging_client.info('Prepared model input.')
      start_time = time.time()
      all_embeddings = model.batch_model(patch_batches)
      cloud_logging_client.info(
          f'Called embedding model; {time.time() - start_time} (sec).'
      )
      all_embeddings = np.concatenate(all_embeddings, axis=0)
    instance_offset = 0
    # build response for each instance.
    for coord_list_or_error in instance_input_coordinates_or_errors:
      if isinstance(coord_list_or_error, pete_errors.PeteError):
        error = coord_list_or_error
        embedding_results.append(
            embedding_response.instance_error_response_v2(
                pete_error_mapping.get_error_code(error), str(error)
            )
        )
        continue
      instance_patch_coordinates = coord_list_or_error
      try:
        patch_embeddings = [
            embedding_response.PatchEmbeddingV2(
                embedding_vector=all_embeddings[  # pylint: disable=undefined-variable
                    instance_offset + i, :
                ].tolist(),
                patch_coordinate=instance_patch,
            )
            for i, instance_patch in enumerate(instance_patch_coordinates)
        ]
        embedding_results.append(
            embedding_response.embedding_instance_response_v2(patch_embeddings)
        )
        cloud_logging_client.info('Processed. embedding model results')
      except pete_errors.PeteError as exp:
        embedding_results.append(
            embedding_response.instance_error_response_v2(
                pete_error_mapping.get_error_code(exp), str(exp)
            )
        )
      # Increment embedding offset unless error occureed during input
      # generation.
      instance_offset += len(instance_patch_coordinates)
    cloud_logging_client.info('Returning embeddings.')
    return embedding_converter.embedding_response_v2_to_json(embedding_results)
