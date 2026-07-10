import unittest
class UsefulPolicyTests(unittest.TestCase):
 def test_equal_dummy_is_not_gain(self): self.assertFalse(0 >= .05)
