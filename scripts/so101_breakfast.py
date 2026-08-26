#!/usr/bin/env python3
import dataclasses
import logging
import sys

import cv2
import numpy as np

from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower


PROMPT = "first put the block onto the plate, move the plate to the center of the table, place the spoon on the right side of the plate, and place the cup on the left side of the plate"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 10093

ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0158628-if00"
ROBOT_ID = "so101_follower_01"

GLOBAL_CAMERA = "/dev/v4l/by-id/usb-Generic_USB_Camera3_200901010001-video-index0"
WRIST_CAMERA = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"

CAMERA_DEVICE_MAP = {
    "global": GLOBAL_CAMERA,
    "wrist": WRIST_CAMERA,
}

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
MODEL_IMAGE_SIZE = 224

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def open_camera(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {path}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_frame(cap: cv2.VideoCapture, name: str) -> np.ndarray:
    ok = False
    frame = None
    for _ in range(3):
        ok, frame = cap.read()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from {name}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = image_tools.resize_with_pad(frame, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    frame = image_tools.convert_to_uint8(frame)
    return np.ascontiguousarray(frame)


def get_state_array(obs: dict) -> np.ndarray:
    return np.asarray([obs[key] for key in JOINT_KEYS], dtype=np.float32)


def build_action_dict(target: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in zip(JOINT_KEYS, target.tolist())}


class SO101RealEnv(_environment.Environment):
    def __init__(self) -> None:
        self._robot = SOFollower(
            SOFollowerRobotConfig(
                port=ROBOT_PORT,
                id=ROBOT_ID,
            )
        )
        self._robot.connect(calibrate=False)
        self._global_cap = open_camera(CAMERA_DEVICE_MAP["global"])
        self._wrist_cap = open_camera(CAMERA_DEVICE_MAP["wrist"])
        self._step_idx = 0

        print(f"global camera device: {CAMERA_DEVICE_MAP['global']}")
        print(f"wrist camera device: {CAMERA_DEVICE_MAP['wrist']}")

    def reset(self) -> None:
        self._step_idx = 0

    def is_episode_complete(self) -> bool:
        return False

    def get_observation(self) -> dict:
        obs = self._robot.get_observation()
        state = get_state_array(obs)
        global_img = read_frame(self._global_cap, "global")
        wrist_img = read_frame(self._wrist_cap, "wrist")
        return {
            "observation/state": state,
            "observation/image": global_img,
            "observation/wrist_image": wrist_img,
            "prompt": PROMPT,
        }

    def apply_action(self, action: dict) -> None:
        if "actions" not in action:
            raise KeyError(f"Policy action missing 'actions', got keys: {list(action.keys())}")

        target = np.asarray(action["actions"], dtype=np.float32)
        if target.ndim != 1:
            raise ValueError(f"Expected one action step of shape (6,), got shape {target.shape}")

        live_obs = self._robot.get_observation()
        live_state = get_state_array(live_obs)

        print(
            f"step {self._step_idx} "
            f"action={np.round(target, 3)} "
            f"state_before={np.round(live_state, 3)}"
        )
        self._robot.send_action(build_action_dict(target))

        try:
            post_obs = self._robot.get_observation()
            post_state = get_state_array(post_obs)
            print(f"  state_after={np.round(post_state, 3)}")
        except Exception:
            print("  state_after=unavailable")

        self._step_idx += 1

    def close(self) -> None:
        try:
            self._global_cap.release()
        finally:
            self._wrist_cap.release()
            self._robot.disconnect()


@dataclasses.dataclass
class Args:
    host: str = SERVER_HOST
    port: int = SERVER_PORT
    action_horizon: int = 25
    max_hz: float = 30.0
    num_episodes: int = 1
    max_episode_steps: int = 100000


def main(args: Args) -> int:
    print("connecting remote policy...")
    ws_client_policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    print("server metadata:", ws_client_policy.get_server_metadata())

    print("opening cameras...")
    print("connecting robot...")
    env = SO101RealEnv()

    try:
        runtime = _runtime.Runtime(
            environment=env,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=args.max_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        print(f"starting runtime loop at max_hz={args.max_hz}, action_horizon={args.action_horizon}")
        runtime.run()
        return 0
    finally:
        try:
            ws_client_policy.close()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main(Args()))
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        raise
