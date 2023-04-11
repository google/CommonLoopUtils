# Copyright 2023 The CLU Authors.
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

from unittest import mock

from absl.testing import absltest
from clu.internal import utils
import jax.numpy as jnp
import ml_collections


class TestError(BaseException):
  __test__ = False
  pass


class HelpersTest(absltest.TestCase):

  def test_log_activity(
      self,
  ):
    with self.assertLogs() as logs:
      with utils.log_activity("test_activity"):
        pass
    self.assertLen(logs.output, 2)
    self.assertEqual(logs.output[0], "INFO:absl:test_activity ...")
    self.assertRegex(logs.output[1],
                     r"^INFO:absl:test_activity finished after \d+.\d\ds.$")

  def test_log_activity_fails(
      self,
  ):
    with self.assertRaises(TestError):  # pylint: disable=g-error-prone-assert-raises, line-too-long
      with self.assertLogs() as logs:
        with utils.log_activity("test_activity"):
          raise TestError()
    self.assertLen(logs.output, 2)
    self.assertEqual(logs.output[0], "INFO:absl:test_activity ...")
    self.assertRegex(logs.output[1],
                     r"^ERROR:absl:test_activity FAILED after \d+.\d\ds")

  def test_logged_with(self):

    @utils.logged_with("test_activity")
    def test():
      pass

    with self.assertLogs() as logs:
      test()
    self.assertLen(logs.output, 2)
    self.assertEqual(logs.output[0], "INFO:absl:test_activity ...")
    self.assertRegex(logs.output[1],
                     r"^INFO:absl:test_activity finished after \d+.\d\ds.$")

  def test_logged_with_fails(self):

    @utils.logged_with("test_activity")
    def test():
      raise TestError()

    with self.assertRaises(TestError):  # pylint: disable=g-error-prone-assert-raises, line-too-long
      with self.assertLogs() as logs:
        test()
    self.assertLen(logs.output, 2)
    self.assertEqual(logs.output[0], "INFO:absl:test_activity ...")
    self.assertRegex(logs.output[1],
                     r"^ERROR:absl:test_activity FAILED after \d+.\d\ds")

  def test_check_param(self):
    a = jnp.array(0.)
    with self.assertRaisesRegex(ValueError, r"^Expected np.array or jnp.array"):
      utils.check_param(None, ndim=1)
    with self.assertRaisesRegex(ValueError, r"^Expected ndim"):
      utils.check_param(a, ndim=1)
    with self.assertRaisesRegex(ValueError, r"^Expected dtype"):
      utils.check_param(a, ndim=0, dtype=jnp.int32)
    utils.check_param(a, ndim=0)  # should work
    utils.check_param(a, ndim=0, dtype=jnp.float32)  # should also work

  def test_flatten_dict(self):
    self.assertEqual(
        utils.flatten_dict(
            ml_collections.ConfigDict({
                "x": 1,
                "y": None,
                "z": ml_collections.ConfigDict({
                    "a": "bc",
                })
            })), [("x", 1), ("y", ""), ("z.a", "bc")])


if __name__ == "__main__":
  absltest.main()
