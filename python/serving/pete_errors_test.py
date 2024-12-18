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

"""Tests for pete errors."""

import inspect
import sys

from absl.testing import absltest
from absl.testing import parameterized
from serving import pete_errors


class PeteErrorsTest(parameterized.TestCase):

  def test_all_errors_use_base_class(self):
    for _, cls in inspect.getmembers(
        sys.modules[pete_errors.__name__], inspect.isclass
    ):
      if cls.__name__ == 'InternalBugError':
        continue
      self.assertTrue(issubclass(cls, pete_errors.PeteError))


if __name__ == '__main__':
  absltest.main()
