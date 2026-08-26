import cv2
import re
import select
import time
from pathlib import Path

from evdev import InputDevice, ecodes
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import init_keyboard_listener, record_loop
from lerobot.teleoperators import Teleoperator
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.utils import log_say

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF270443-if00"
ROBOT_ID = "so101_follower_01"

DATASET_ROOT = Path.home() / "datasets" / f"so101_yellow_block_plate_{int(time.time())}"
REPO_ID = "ljt/so101-yellow-block-plate"

FPS = 30
NUM_EPISODES = 1
EPISODE_TIME_S = 9999
RESET_TIME_S = 5
TASK = "Pick up the yellow block and place it into the plate"

ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def find_joycon_event(device_name="Joy-Con (R)"):
    text = Path("/proc/bus/input/devices").read_text()
    blocks = text.split("\n\n")
    for block in blocks:
        if device_name in block and "(IMU)" not in block:
            m = re.search(r"Handlers=.*?(event\d+)", block)
            if m:
                return f"/dev/input/{m.group(1)}"
    raise RuntimeError(f"Could not find input event device for {device_name!r}")


class JoyConTeleop(Teleoperator):
    def __init__(self, event_path: str, initial_action: dict[str, float]):
        self.event_path = event_path
        self.target = initial_action.copy()
        self.dev = None
        self.last = None
        self.stick_x = 0.0
        self.stick_y = 0.0
        self.button_state = {
            ecodes.BTN_A: 0,
            ecodes.BTN_B: 0,
            ecodes.BTN_X: 0,
            ecodes.BTN_Y: 0,
        }
        self.rx_info = None
        self.ry_info = None
        self._is_connected = False
        self._is_calibrated = True

        self.PAN_RATE = 18.0
        self.LIFT_RATE = 18.0
        self.ELBOW_RATE = 20.0
        self.WRIST_FLEX_RATE = 20.0
        self.ROLL_STEP = 5.0
        self.GRIP_STEP = 6.0

    @property
    def is_connected(self):
        return self._is_connected

    @property
    def is_calibrated(self):
        return self._is_calibrated

    @property
    def action_features(self):
        return {k: {"dtype": "float32", "shape": (1,)} for k in ACTION_KEYS}

    @property
    def feedback_features(self):
        return {}

    def configure(self):
        return None

    def calibrate(self):
        self._is_calibrated = True
        return None

    def connect(self):
        print(f"Using Joy-Con event device: {self.event_path}")
        self.dev = InputDevice(self.event_path)
        self.dev.grab()
        self.rx_info = self.dev.absinfo(ecodes.ABS_RX)
        self.ry_info = self.dev.absinfo(ecodes.ABS_RY)
        self.last = time.perf_counter()
        self._is_connected = True

    def disconnect(self):
        if self.dev is not None:
            try:
                self.dev.ungrab()
            except Exception:
                pass
            self.dev.close()
            self.dev = None
        self._is_connected = False

    def send_feedback(self, obs):
        return None

    def _norm(self, v, info):
        lo, hi = info.min, info.max
        c = (lo + hi) / 2.0
        r = max((hi - lo) / 2.0, 1.0)
        x = (v - c) / r
        if abs(x) < 0.12:
            return 0.0
        return max(-1.0, min(1.0, x))

    def _clamp(self):
        self.target["gripper.pos"] = max(0.0, min(100.0, self.target["gripper.pos"]))

    def get_action(self):
        r, _, _ = select.select([self.dev.fd], [], [], 0.0)
        if r:
            for event in self.dev.read():
                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_RX:
                        self.stick_x = self._norm(event.value, self.rx_info)
                    elif event.code == ecodes.ABS_RY:
                        self.stick_y = self._norm(event.value, self.ry_info)

                elif event.type == ecodes.EV_KEY:
                    if event.code in self.button_state:
                        self.button_state[event.code] = event.value
                    elif event.value == 1:
                        if event.code == ecodes.BTN_TL:      # SL
                            self.target["wrist_roll.pos"] -= self.ROLL_STEP
                        elif event.code == ecodes.BTN_TL2:   # SR
                            self.target["wrist_roll.pos"] += self.ROLL_STEP
                        elif event.code == ecodes.BTN_TR:    # R
                            self.target["gripper.pos"] += self.GRIP_STEP
                        elif event.code == ecodes.BTN_TR2:   # ZR
                            self.target["gripper.pos"] -= self.GRIP_STEP

        now = time.perf_counter()
        dt = now - self.last
        self.last = now

        self.target["shoulder_pan.pos"] += self.stick_x * self.PAN_RATE * dt
        self.target["shoulder_lift.pos"] += (-self.stick_y) * self.LIFT_RATE * dt

        if self.button_state[ecodes.BTN_A]:
            self.target["elbow_flex.pos"] -= self.ELBOW_RATE * dt
        if self.button_state[ecodes.BTN_B]:
            self.target["elbow_flex.pos"] += self.ELBOW_RATE * dt
        if self.button_state[ecodes.BTN_X]:
            self.target["wrist_flex.pos"] += self.WRIST_FLEX_RATE * dt
        if self.button_state[ecodes.BTN_Y]:
            self.target["wrist_flex.pos"] -= self.WRIST_FLEX_RATE * dt

        self._clamp()
        return self.target.copy()


def main():
    print("starting main")

    robot_config = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        cameras={
            "global": OpenCVCameraConfig(
                index_or_path="/dev/video0",
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
                backend=cv2.CAP_V4L2,
                warmup_s=3,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path="/dev/video2",
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
                backend=cv2.CAP_V4L2,
                warmup_s=3,
            ),
        },
    )

    print("before robot init")
    robot = SO101Follower(robot_config)

    print("before robot connect")
    robot.connect(calibrate=False)
    print("after robot connect")

    obs = robot.get_observation()
    initial_action = {k: float(obs[k]) for k in ACTION_KEYS}

    joycon_event = find_joycon_event("Joy-Con (R)")
    teleop = JoyConTeleop(joycon_event, initial_action)

    print("before teleop connect")
    teleop.connect()
    print("after teleop connect")

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    print("before dataset create")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=2,
    )
    print("after dataset create")
    print(f"dataset root = {DATASET_ROOT}")

    _, events = init_keyboard_listener()
    print("events =", events)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    try:
        episode_idx = 0
        print("before while loop")
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1}/{NUM_EPISODES}")
            log_say("Task: pick up the yellow block and place it into the plate")
            log_say("Press n to save this episode")
            log_say("Press left arrow to discard and re-record")
            log_say("Press Esc to quit")

            print("before first record_loop")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_S,
                single_task=TASK,
                display_data=False,
            )
            print("after first record_loop")

            if events["rerecord_episode"]:
                log_say("Discarded current episode, please record it again")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            if events["stop_recording"]:
                break

            dataset.save_episode()
            episode_idx += 1
            log_say(f"Episode saved to {DATASET_ROOT}")

        log_say("Finalizing dataset...")
        dataset.finalize()
        log_say(f"Saved locally to: {DATASET_ROOT}")
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
