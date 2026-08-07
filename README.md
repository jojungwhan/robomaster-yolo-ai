# RoboMaster Vision, Voice, LEGO, and Autonomy Demo

The application combines screen-captured RoboMaster video, YOLO object detection,
OpenCV LEGO detection, push-to-talk conversation, spoken mission prompts, and a
safety-gated autonomous motion planner.

## Modes and controls

- `1`: exploration patrol
- `2`: search and rescue; a person or confirmed red LEGO cross stops motion and starts
  a spoken check-in
- `3`: follow the selected YOLO target
- `4`: find an ArUco-tagged or colored LEGO target
- `V`, `Space`, or global `F8`: voice conversation
- `T`: cycle the selected target
- `M`: arm or disarm autonomous motion
- `E` or global `Esc`: latch the emergency stop
- `R`: reset the emergency stop; motion remains disarmed until `M` is pressed
- `N`: toggle automatic narration
- `Q`: stop and exit

## Safe default: no physical motion

```powershell
python robomaster_yolo_ai.py
```

The default backend only calculates and displays commands. It cannot move a robot.

To exercise the full arming and avoidance state machine without hardware:

```powershell
python robomaster_yolo_ai.py --motion-backend dry-run --enable-motion
```

Select a mission and press `M`. The UI will show simulated commands and four simulated
2-meter ToF readings.

## Physical motion prerequisites

Do not enable physical motion from screen pixels alone. The included hardware adapter
requires all of the following:

1. A compatible RoboMaster Python client and robot connection.
2. Four live ToF sensors mounted and mapped as front, left, right, and rear.
3. A flat, bounded test area with no stairs, pets, or bystanders inside the boundary.
4. A reachable emergency-stop keyboard and direct supervision.
5. Initial testing with the drive wheels raised off the floor.

DJI's public PC Python SDK documents EP/EP Core support. The consumer S1's built-in
Programming Lab is a different environment. A community S1-compatible SDK or ROS driver
may expose the same `robomaster` client API, but it must be validated independently.
The application intentionally does not automate RoboMaster PC keyboard controls.

After installing and validating a compatible client, hardware mode is:

```powershell
python robomaster_yolo_ai.py `
  --motion-backend robomaster `
  --conn-type ap `
  --tof-layout front,left,right,rear `
  --min-tof-count 4 `
  --enable-motion
```

The robot still starts disarmed. Confirm the displayed ToF directions and distances,
then press `M`. Every speed command expires after 0.35 seconds. Missing/stale ToF data,
an impact event, a failed command, the global `Esc` key, or a stalled vision loop stops
the chassis. Maximum configured speed is 0.12 m/s.

Odometry, arming, emergency stops, and rescue detections are written to
`mission_events.jsonl`.

## Lego-Identification model

The LEGO and Rescue modes use the
[`vsmidhun21/Lego-Identification`](https://github.com/vsmidhun21/Lego-Identification)
`FinalCoShSi.pt` checkpoint. It recognizes 34 color/shape/size combinations such as
`Red_Rectangle_Medium`; app labels are prefixed with `lego_piece_`.

The verified checkpoint is installed under
`models\lego-identification\FinalCoShSi.pt`. To restore it on another machine:

```powershell
.\setup_lego_identification.ps1
```

The setup script pins repository commit `ddd54ae077a8fed243065a1104ee14eb4aa5f5e2`
and verifies SHA256
`87591257D011CC7409CFF14BABF28A1D15402AB521E75F3D10BF5F7A1E013CF6`.
The model runs only in modes 2 and 4, in a background worker at 512px once every 0.75
seconds by default, with short-lived cached boxes between runs. This keeps the OpenCV
window and the main person/object detector responsive. Tune it if needed:

```powershell
python robomaster_yolo_ai.py `
  --lego-model-confidence 0.35 `
  --lego-model-interval 0.60 `
  --lego-model-imgsz 512
```

Use `--disable-lego-model` to run the OpenCV detectors without the third-party model.
The upstream repository had no license file at the pinned revision, so do not
redistribute its weights without permission from its author. PyTorch checkpoints are
executable serialization formats; this integration accepts the pinned checkpoint only
when its hash matches. A differently named custom checkpoint is treated as explicitly
user-supplied.

## Semantic LEGO rescue pattern

The first semantic pattern is a red 5-by-5 LEGO cross:

```text
..R..
..R..
RRRRR
..R..
..R..
```

Build it flat, keep it approximately upright and camera-facing, and put it on a neutral,
non-red background. Generate a screen/print reference with:

```powershell
python generate_lego_help_pattern.py --output lego_help_red_cross.png
```

OpenCV rejects plain red squares and rectangles, and requires the cross across three
consecutive frames. A confirmed cross appears as `lego_signal_help_needed`, disarms
autonomous motion, writes `lego_help_signal_confirmed` to the mission log, and asks
“Do you need help?” through the existing voice-conversation path. A single candidate
frame already requests zero motion while confirmation is pending. The pattern detector
runs in Rescue and LEGO modes even when the third-party model is disabled.

Automatic speech follows the `N` narration toggle. Recognition and stopping still work
when narration is off. Re-arm manually with `M` only after checking the scene and all ToF
readings.

## Other LEGO targets

Color/contour detection works best with isolated, brightly colored bricks on a neutral
background. It can confuse other rectangular colored objects with LEGO.

For autonomous approach, attach an ArUco marker to the LEGO build:

```powershell
python generate_lego_marker.py --id 0 --output lego_marker_0.png
```

Print the marker with its white border, attach it flat to the target, start LEGO mode
with `4`, and select the marker with `T` if necessary.

Public alternatives evaluated:

- `mw00/yolov7-lego` on Hugging Face provides LEGO weights but its model card warns
  about many false positives and only 1,000 single-class training images.
- `vsmidhun21/Lego-Identification` on GitHub is integrated for experimental piece
  labeling but is never used for collision clearance.

Neither third-party model should replace ToF collision sensing.
