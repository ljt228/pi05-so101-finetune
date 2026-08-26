import select
import time

from evdev import InputDevice, ecodes
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF270443-if00"
ROBOT_ID = "so101_follower_01"
EVENT_DEV = "/dev/input/event6"  # Joy-Con (L)

cfg = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
robot = SO101Follower(cfg)
robot.connect(calibrate=False)

dev = InputDevice(EVENT_DEV)
dev.grab()

obs = robot.get_observation()

keys = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
target = {k: float(obs[k]) for k in keys}

x_info = dev.absinfo(ecodes.ABS_X)
y_info = dev.absinfo(ecodes.ABS_Y)

def norm(v, info):
    lo, hi = info.min, info.max
    c = (lo + hi) / 2.0
    r = max((hi - lo) / 2.0, 1.0)
    x = (v - c) / r
    if abs(x) < 0.12:
        return 0.0
    return max(-1.0, min(1.0, x))

stick_x = 0.0
stick_y = 0.0

button_state = {
    544: 0,
    545: 0,
    546: 0,
    547: 0,
}

PAN_RATE = 18.0
LIFT_RATE = 18.0
ELBOW_RATE = 20.0
WRIST_FLEX_RATE = 20.0
ROLL_STEP = 5.0
GRIP_STEP = 6.0

print("Joy-Con (L) -> SO101 teleop started")
print("ABS_X: shoulder_pan")
print("ABS_Y: shoulder_lift")
print("544/545: elbow -/+")
print("546/547: wrist_flex +/-")
print("TL/TL2: wrist_roll -/+")
print("TR/TR2: gripper open/close")
print("Ctrl+C to quit")

last = time.perf_counter()

try:
    while True:
        r, _, _ = select.select([dev.fd], [], [], 0.02)
        if r:
            for event in dev.read():
                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_X:
                        stick_x = norm(event.value, x_info)
                    elif event.code == ecodes.ABS_Y:
                        stick_y = norm(event.value, y_info)

                elif event.type == ecodes.EV_KEY:
                    if event.code in button_state:
                        button_state[event.code] = event.value
                        print("button", event.code, "state", event.value)
                    elif event.value == 1:
                        if event.code == ecodes.BTN_TL:
                            target["wrist_roll.pos"] -= ROLL_STEP
                        elif event.code == ecodes.BTN_TL2:
                            target["wrist_roll.pos"] += ROLL_STEP
                        elif event.code == ecodes.BTN_TR:
                            target["gripper.pos"] += GRIP_STEP
                        elif event.code == ecodes.BTN_TR2:
                            target["gripper.pos"] -= GRIP_STEP

        now = time.perf_counter()
        dt = now - last
        last = now

        target["shoulder_pan.pos"] += stick_x * PAN_RATE * dt
        target["shoulder_lift.pos"] += (-stick_y) * LIFT_RATE * dt

        if button_state[544]:
            target["elbow_flex.pos"] -= ELBOW_RATE * dt
        if button_state[545]:
            target["elbow_flex.pos"] += ELBOW_RATE * dt
        if button_state[546]:
            target["wrist_flex.pos"] += WRIST_FLEX_RATE * dt
        if button_state[547]:
            target["wrist_flex.pos"] -= WRIST_FLEX_RATE * dt

        robot.send_action(target)

except KeyboardInterrupt:
    print("\nStopping teleop...")

finally:
    try:
        dev.ungrab()
    except Exception:
        pass
    robot.disconnect()
