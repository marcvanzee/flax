from absl.testing import absltest

from flax.core import scope
from flax.core.scope import Scope, init

from jax import random

import functools

class ScopeTest(absltest.TestCase):

  def test_in_find_filter(self):    
    self.assertTrue(scope.in_kind_filter(True, "test"))
    self.assertFalse(scope.in_kind_filter("123", "1234"))
    self.assertTrue(scope.in_kind_filter("123", "123"))

    self.assertFalse(scope.in_kind_filter(["1", "2"], "3"))
    self.assertFalse(scope.in_kind_filter([], ""))
    
    # This should be disallowed if we run pytyping, but passes here.
    self.assertTrue(scope.in_kind_filter([1, 5, 2], 2))
  
  def test_group_kinds(self):
    xs = {
        "1": { "11": 1 },
        "2": { "22": 2 },
        "3": { "33": 3 },
    }
    # Both variables are selected by the first kind filter.
    filter = [True, "1", "2", ["3", "4"]]
    expected = (xs, {}, {}, {})
    self.assertEqual(scope.group_kinds(xs, filter), expected)

    # Filter 1, 2, 3 select resp. variable 1, 2, 3.
    filter = filter[1:]
    expected = ({"1": xs["1"]}, {"2": xs["2"]}, {"3": xs["3"]})
    self.assertEqual(scope.group_kinds(xs, filter), expected)

    # Filter 1 selects variable 1 and 2. Variable 3 is ignored.
    filter = [["1", "2"], ["4"]]
    expected = ({"1": xs["1"], "2": xs["2"]}, {})
    self.assertEqual(scope.group_kinds(xs, filter), expected)

    filter = ["3", "1"]
    expected = ({"3": xs["3"]}, {"1": xs["1"]})
    self.assertEqual(scope.group_kinds(xs, filter), expected)

  def test_temporary_context(self):
    with Scope(variables=None).temporary() as s:
      pass
    self.assertTrue(s.invalid)

  def test_init(self):
    # Basic use of init.
    def simple_fn(scope: Scope):
      return scope
    
    # If no mutable variables are selected, only the scope object is returned.
    print('key', random.PRNGKey(42))
    s = init(simple_fn, mutable=False)(random.PRNGKey(1))
    self.assertTrue(isinstance(s, Scope))

    # A scope that does nothing has no params.
    _, params = init(simple_fn)(random.PRNGKey(1))
    self.assertEqual(params, {})

    # init can be used as a decorated (since it is using functools.wraps).
    @functools.partial(init, mutable=False)
    def test_fn(scope: Scope):
      return scope

    s = test_fn(random.PRNGKey(1))
    self.assertTrue(isinstance(s, Scope))
    
  def test_apply(self):
    pass


if __name__ == '__main__':
  absltest.main()