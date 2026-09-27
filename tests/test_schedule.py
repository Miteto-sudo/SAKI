import unittest

from saki_opd.runtime import resolve_epsilon_schedule


class EpsilonScheduleTest(unittest.TestCase):
    def test_linear50_reaches_exact_zero_at_step51(self):
        config = {
            "epsilon": 0.02,
            "epsilon_schedule": {
                "type": "linear",
                "start_step": 1,
                "end_step": 51,
                "start_epsilon": 0.02,
                "end_epsilon": 0.0,
            },
        }
        self.assertEqual(resolve_epsilon_schedule(config, 1), 0.02)
        self.assertAlmostEqual(resolve_epsilon_schedule(config, 26), 0.01)
        self.assertAlmostEqual(resolve_epsilon_schedule(config, 50), 0.0004)
        self.assertEqual(resolve_epsilon_schedule(config, 51), 0.0)
        self.assertEqual(resolve_epsilon_schedule(config, 200), 0.0)


if __name__ == "__main__":
    unittest.main()
