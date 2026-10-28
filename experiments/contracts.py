"""nuScenes mini scene-isolated, 3-second ego trajectory contract."""
import json
import math
from pathlib import Path

import numpy as np

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
TRAIN_SCENES = {"scene-0061", "scene-0553", "scene-0655", "scene-0757",
                "scene-0796", "scene-1077", "scene-1094", "scene-1100"}
VAL_SCENES = {"scene-0103", "scene-0916"}
ACTION_NAMES = tuple(f"{turn}_{speed}" for turn in ("straight", "left", "right")
                     for speed in ("keep", "accelerate", "brake"))
HORIZON = np.arange(1, 7, dtype=np.float64) * .5


def yaw(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))


def local_xy(xy, origin, angle):
    delta = np.asarray(xy, dtype=np.float64)[..., :2] - np.asarray(origin)[:2]
    c, s = math.cos(angle), math.sin(angle)
    return delta @ np.array([[c, -s], [s, c]])


def wrap(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi


def action_class(trajectory, initial_speed):
    trajectory = np.asarray(trajectory)
    # Heading of the final displacement is less noisy than stationary ego yaw.
    angle = np.arctan2(trajectory[-1, 1], max(trajectory[-1, 0], .1))
    turn = 1 if angle > .12 else 2 if angle < -.12 else 0
    final_speed = np.linalg.norm(trajectory[-1] - trajectory[-2]) / .5
    speed = 1 if final_speed-initial_speed > 1.0 else 2 if final_speed-initial_speed < -1.0 else 0
    return int(turn*3+speed)


def build_records(root):
    root = Path(root)
    meta = root / "v1.0-mini"
    tables = {name: {r["token"]: r for r in json.loads((meta/f"{name}.json").read_text())}
              for name in ("scene", "sample", "sample_data", "ego_pose", "sensor", "calibrated_sensor",
                           "sample_annotation", "instance", "category")}
    # The devkit builds these reverse indexes; the official raw JSON does not contain them.
    for sample in tables["sample"].values():
        sample["data"], sample["anns"] = {}, []
    for data in tables["sample_data"].values():
        if data["is_key_frame"]:
            calibration = tables["calibrated_sensor"][data["calibrated_sensor_token"]]
            sensor = tables["sensor"][calibration["sensor_token"]]
            tables["sample"][data["sample_token"]]["data"][sensor["channel"]] = data["token"]
    for annotation in tables["sample_annotation"].values():
        tables["sample"][annotation["sample_token"]]["anns"].append(annotation["token"])
    records, excluded = [], {"no_history": 0, "short_future": 0}
    for scene in tables["scene"].values():
        assert scene["name"] in TRAIN_SCENES | VAL_SCENES, scene["name"]
        samples = []
        token = scene["first_sample_token"]
        while token:
            sample = tables["sample"][token]
            samples.append(sample)
            token = sample["next"]
        poses = [tables["ego_pose"][tables["sample_data"][s["data"]["CAM_FRONT"]]["ego_pose_token"]]
                 for s in samples]
        timestamps = np.array([p["timestamp"] for p in poses], dtype=np.float64) / 1e6
        positions = np.array([p["translation"] for p in poses])
        for index, sample in enumerate(samples):
            if index < 2:
                excluded["no_history"] += 1
                continue
            # Absolute Unix timestamps need float64 to preserve half-second offsets.
            target_times = float(timestamps[index]) + HORIZON
            if target_times[-1] > timestamps[-1]:
                excluded["short_future"] += 1
                continue
            angle = yaw(poses[index]["rotation"])
            dt = timestamps[index] - timestamps[index-1]
            previous_dt = timestamps[index-1] - timestamps[index-2]
            velocity = local_xy(positions[index], positions[index-1], angle) / dt
            old_velocity = local_xy(positions[index-1], positions[index-2], angle) / previous_dt
            acceleration = (velocity - old_velocity) / dt
            yawrate = wrap(angle-yaw(poses[index-1]["rotation"])) / dt
            state = np.array([*velocity, *acceleration, yawrate, np.linalg.norm(velocity)], dtype=np.float32)
            future_global = np.stack([np.interp(target_times, timestamps, positions[:, axis])
                                      for axis in range(2)], axis=-1)
            trajectory = local_xy(future_global, positions[index], angle).astype(np.float32)
            images = [tables["sample_data"][sample["data"][cam]]["filename"] for cam in CAMERAS]
            assert all((root/path).is_file() for path in images)
            # Oracle future occupied-circle proxy, not a simulator collision score.
            objects = []
            for target_time in target_times:
                future_index = int(np.argmin(abs(timestamps-target_time)))
                occupied = []
                for annotation_token in samples[future_index]["anns"]:
                    ann = tables["sample_annotation"][annotation_token]
                    inst = tables["instance"][ann["instance_token"]]
                    category = tables["category"][inst["category_token"]]["name"]
                    if category.startswith(("vehicle.", "human.")):
                        xy = local_xy(ann["translation"], positions[index], angle)
                        # Half-width circle with ego 1m radius; intentionally approximate.
                        occupied.append([float(xy[0]), float(xy[1]), float(ann["size"][0]/2+1.0)])
                objects.append(occupied)
            records.append(dict(token=sample["token"], scene=scene["name"],
                split="train" if scene["name"] in TRAIN_SCENES else "val",
                timestamp_s=float(timestamps[index]),
                input_max_timestamp_s=max(tables["sample_data"][sample["data"][cam]]["timestamp"]/1e6 for cam in CAMERAS),
                future_times_s=target_times.tolist(), images=images, state=state.tolist(),
                trajectory=trajectory.tolist(), action=action_class(trajectory, state[-1]),
                future_occupied_circles=objects))
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    assert train and val and not ({r["scene"] for r in train} & {r["scene"] for r in val})
    assert all(r["future_times_s"][0] > r["input_max_timestamp_s"] for r in records)
    return records, dict(train=len(train), val=len(val), excluded=excluded,
        train_scenes=sorted(TRAIN_SCENES), val_scenes=sorted(VAL_SCENES),
        horizon_s=HORIZON.tolist(), coordinates="ego current x forward, y left; meters",
        state_fields=["vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "yaw_rate_radps", "speed_mps"],
        action_names=ACTION_NAMES, action_thresholds=dict(turn_angle_rad=.12, delta_speed_mps=1.0),
        source="official nuScenes mini scene split; CAM_FRONT ego pose timestamps",
        safety_proxy="future annotation occupied circles at nearest sample, ego radius 1m",
        lane_violation_proxy=None, lane_proxy_reason="No nuScenes map expansion asset; not fabricated")


def self_check():
    assert np.allclose(local_xy([0, 2], [0, 0], math.pi/2), [2, 0], atol=1e-6)
    assert np.allclose(local_xy([2, 0], [1, 0], 0), [1, 0])
    assert action_class(np.stack([HORIZON*5, HORIZON*0], -1), 5) == 0
    assert not TRAIN_SCENES & VAL_SCENES
    print("DRIVING_CONTRACT_CHECK_OK", flush=True)


if __name__ == "__main__":
    self_check()
