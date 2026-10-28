"""Ground-truth integration checks using the same raw JSON schema as nuScenes."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from contracts import CAMERAS, HORIZON, build_records, self_check


class ContractTests(unittest.TestCase):
    def test_raw_reverse_indexes_rotated_ego_and_future_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root/"v1.0-mini"
            meta.mkdir()
            tables = {name: [] for name in ("scene", "sample", "sample_data", "ego_pose",
                       "sensor", "calibrated_sensor", "sample_annotation", "instance", "category")}
            for camera in CAMERAS:
                tables["sensor"].append(dict(token=camera, channel=camera))
                tables["calibrated_sensor"].append(dict(token=camera, sensor_token=camera))
                (root/f"{camera}.jpg").write_bytes(b"fixture only; contract checks paths, not pixel decoding")
            for scene_index, (name, angle) in enumerate((("scene-0061", 0), ("scene-0103", math.pi/2))):
                scene_token = f"scene{scene_index}"
                tables["scene"].append(dict(token=scene_token, name=name, first_sample_token=f"{scene_token}-0"))
                for index in range(14):
                    token = f"{scene_token}-{index}"
                    timestamp = int((100+index*.5)*1e6)
                    tables["sample"].append(dict(token=token, scene_token=scene_token, timestamp=timestamp,
                        prev=f"{scene_token}-{index-1}" if index else "",
                        next=f"{scene_token}-{index+1}" if index < 13 else ""))
                    tables["ego_pose"].append(dict(token=token, timestamp=timestamp,
                        translation=[10+index*math.cos(angle), 20+index*math.sin(angle), 0],
                        rotation=[math.cos(angle/2), 0, 0, math.sin(angle/2)]))
                    for camera in CAMERAS:
                        tables["sample_data"].append(dict(token=token+camera, sample_token=token,
                            timestamp=timestamp+5000, ego_pose_token=token, is_key_frame=True,
                            calibrated_sensor_token=camera, filename=f"{camera}.jpg"))
            for name, values in tables.items():
                (meta/f"{name}.json").write_text(json.dumps(values))
            records, manifest = build_records(root)
            self.assertEqual(manifest["train"], 6)
            self.assertEqual(manifest["val"], 6)
            for record in records:
                np.testing.assert_allclose(record["trajectory"], np.stack([HORIZON*2, HORIZON*0], -1), atol=1e-5)
                np.testing.assert_allclose(record["state"], [2, 0, 0, 0, 0, 2], atol=1e-5)
                self.assertEqual(record["action"], 0)
                self.assertGreater(record["future_times_s"][0], record["input_max_timestamp_s"])
            self.assertFalse({r["scene"] for r in records if r["split"] == "train"} &
                             {r["scene"] for r in records if r["split"] == "val"})


if __name__ == "__main__":
    self_check()
    unittest.main()
