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

"""Tests for pete flags."""

from absl.testing import absltest

from health_foundations.path_foundation.serving import pete_flags


class PeteFlagsTest(absltest.TestCase):

  def test_default_load_multi_string_returns_none(self):
    self.assertIsNone(pete_flags._load_multi_string(None))

  def test_load_multi_string_parse_json_string(self):
    self.assertEqual(pete_flags._load_multi_string('["a", "b"]'), ['a', 'b'])

  def test_bad_json_returns_value(self):
    self.assertEqual(pete_flags._load_multi_string('/abc'), '/abc')


if __name__ == '__main__':
  absltest.main()
