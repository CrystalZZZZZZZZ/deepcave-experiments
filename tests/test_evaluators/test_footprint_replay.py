import unittest

from deepcave.evaluators.footprint_replay import subset


class TestFootprintReplay(unittest.TestCase):
    def test_subset_hides_future_points_and_clamps_trial(self):
        cloud = {
            "point_meta": [
                {"category": "configs", "config_id": 1, "first_seen": 0, "x": 0.0, "y": 0.0},
                {"category": "configs", "config_id": 2, "first_seen": 2, "x": 1.0, "y": 1.0},
            ],
            "incumbent_prefix": [
                {"order": 0, "config_id": 1},
                {"order": 2, "config_id": 2},
            ],
            "t_max": 2,
        }

        frame = subset(cloud, trial=1)

        self.assertEqual(frame["trial"], 1)
        self.assertEqual([point["visible"] for point in frame["points"]], [True, False])
        self.assertEqual(frame["incumbent_prefix"], [{"order": 0, "config_id": 1}])
        self.assertEqual(subset(cloud, trial=100)["trial"], 2)


if __name__ == "__main__":
    unittest.main()