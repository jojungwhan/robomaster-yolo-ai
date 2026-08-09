# RoboMaster Classroom Mission System

A safety-gated RoboMaster S1/EP classroom project that combines full-frame screen
capture, YOLO object detection, LEGO recognition, voice conversation, text-to-speech,
mission scenarios, autonomous navigation states, and an OpenCV operator dashboard.

Public repository: https://github.com/jojungwhan/robomaster-yolo-ai

> **Safety boundary:** the software reduces risk but cannot guarantee that a robot will
> never contact a wall or person. Camera detections are not collision sensors. Keep a
> teacher at the controls, use the emergency stop, validate every ToF direction, test
> with wheels raised first, and never operate near stairs or an unbounded edge.

## What is included

- Full monitor capture fitted into the dashboard without cropping or stretching.
- YOLOv8n detection for the 80 COCO object classes, with boxes, confidence, and a
  visible object list.
- Persistent, instance-level target lock: two-frame confirmation, short missed-frame
  hold, distractor rejection, bounded reacquisition, and visual close-target stop.
- Optional camera-gimbal tracking that is separate from chassis steering.
- Eleven data-driven classroom missions loaded from `scenarios.json`.
- An explicit navigation state machine with scan sweeps, timeouts, patrol, obstacle
  recovery, target approach, interaction stop, waypoint navigation, completion, and
  emergency-stop states.
- Four-direction ToF display, low-speed safety controller, command watchdog, impact
  latch, visual person clearance, and fail-closed SDK error handling.
- Odometry trail, range-derived obstacle points, named waypoints, mission-map save and
  resume, and return to the recorded home position.
- OpenCV classroom dashboard with scenario cards, clickable detections, captions,
  connection indicators, event timeline, map, teacher/student roles, and mouse controls.
- Local text-to-speech for prompts and detected-object reading, including mute, rate,
  and volume controls.
- Push-to-talk speech-to-text and multi-turn OpenAI conversation, with separate history
  for each scenario.
- Experimental `Lego-Identification` labels, colored-brick candidates, ArUco targets,
  and a semantic red LEGO cross for “help needed.”
- Local-only startup when no OpenAI API key is configured.

## Quick start

On Windows, the one-time setup script installs the RoboMaster PC application and the
project's Python dependencies:

```powershell
.\setup.ps1
.\.venv-robot\Scripts\Activate.ps1
python robomaster_yolo_ai.py
```

`setup.ps1` downloads `RoboMaster_x64_Installer_v1.1.5.exe` from the
[repository-provided Google Drive file](https://drive.google.com/file/d/1KaB71nUmsWfCn3udnFZWmD_qKw3eBObs/view?usp=drive_link),
verifies its pinned SHA256 and Windows publisher signature, and runs its documented
Inno Setup unattended installation. Windows may display an administrator approval
prompt. The installer is never executed if either verification fails, and the setup
uses `/NORESTART` so it cannot restart the computer automatically.

The same setup also installs a pinned 64-bit Python 3.8.10 runtime when needed,
creates `.venv-robot`, and runs:

```powershell
python -m pip install -r requirements-robot.txt
```

DJI published the Windows RoboMaster SDK only for Python 3.6-3.8. The repository
therefore keeps an offline, hash-pinned copy of the Python 3.8 SDK wheel and its native
codec DLLs under `vendor/robomaster-sdk/windows`. It also archives the two prerequisite
executables from DJI's pinned SDK commit. The DJI-bundled all-in-one VC runtime is
unsigned, so it is retained only as a recovery artifact and is never run by setup.
See [`vendor/robomaster-sdk/README.md`](vendor/robomaster-sdk/README.md) for sources,
hashes, signatures, and exact behavior.

To install or repair only the RoboMaster PC application, run:

```powershell
.\setup_robomaster_pc.ps1
```

To create only the local simulation/dashboard environment, with no desktop app or
robot SDK installation, run `.\setup.ps1 -LocalOnly`. To install the robot SDK but
skip the RoboMaster PC desktop app, run `.\setup.ps1 -SkipRoboMasterPc`. The equivalent
manual local-only Python setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python robomaster_yolo_ai.py
```

The first YOLO run may download `yolov8n.pt`. The application starts in dry-run mode,
opens the scenario menu, and **cannot move physical hardware**. The LEGO checkpoint is
optional; if it is absent, OpenCV LEGO and ArUco detection remain available.

### RoboMaster PC camera view

Connect the S1/EP in the RoboMaster PC application and place its live camera view on
the monitor selected by `--screen`. This project analyzes the selected monitor; it does
not control the RoboMaster PC application's keyboard or mouse input. Keep the Python
motion backend in its default dry-run mode when the PC application is being used only
as the camera source.

Cloud vision and conversation are also optional. Set the key only in the current shell
or another secret manager; never put a real key in source control:

```powershell
$env:OPENAI_API_KEY = "your-key-here"
python robomaster_yolo_ai.py
```

Without a key, object detection, scenarios, navigation simulation, the dashboard,
local TTS, and LEGO OpenCV features still start normally. TALK reports that cloud
conversation is unavailable instead of crashing the application.

### Detection classes and performance

The default `yolov8n.pt` is the small COCO model: it recognizes 80 broad classes such as
person, chair, bottle, backpack, laptop, TV, book, and cell phone. It does not know every
classroom item or arbitrary LEGO piece. Supply a compatible custom Ultralytics checkpoint
to add domain-specific classes:

```powershell
python robomaster_yolo_ai.py --yolo-model .\models\classroom-best.pt
```

YOLO latency is displayed at the bottom of the dashboard. On a CPU, reduce the inference
size and increase the interval; cached boxes keep the UI responsive between inferences:

```powershell
python robomaster_yolo_ai.py `
  --yolo-imgsz 416 `
  --yolo-interval 0.20 `
  --yolo-confidence 0.25
```

Use `--yolo-device 0` to select a supported GPU explicitly. Cached detector results carry
an observation ID, so replaying one result cannot satisfy the multi-frame target-lock
confirmation by itself.

## Dashboard

The preview is a single resizable mission console rather than several overlapping
OpenCV windows. It contains:

- the entire captured monitor, letterboxed into the camera panel;
- exact locked-target highlighting and a center-to-target guide line;
- robot, ToF, cloud, microphone, and TTS indicators;
- navigation, target, gimbal, and motion state;
- detected-object list and click-to-select regions in visual-target missions;
- four-direction ToF clearance view;
- odometry, home, waypoints, trail, and recent obstacle points;
- student/robot captions and a recent event timeline;
- scenario, speech, mission, safety, and teacher-only controls.

On Windows, the window is placed on top and excluded from the monitor capture where the
OS supports `SetWindowDisplayAffinity`, preventing recursive preview capture. Closing
the preview with its title-bar button latches the emergency stop and exits the loop.

If the wrong monitor is captured, select another one:

```powershell
python robomaster_yolo_ai.py --screen 2
```

The rendered dashboard adapts down to 960×640. Its default maximum is 1280×840:

```powershell
python robomaster_yolo_ai.py --dashboard-width 1100 --dashboard-height 700
```

## Missions

Selecting or changing a mission always stops and disarms motion. A teacher must inspect
the scene and ToF panel and then arm again.

| Key | Mission | Navigation behavior |
| --- | --- | --- |
| `1` | Exploration Patrol | Slow patrol, obstacle turns, object narration |
| `2` | Search & Rescue (SDG 3 + 11) | Search for a person/help cross, stop, ask if help is needed |
| `3` | Target Guidance | Confirm, lock, align, approach, stop at visual threshold |
| `4` | LEGO Search | Find LEGO pieces, semantic builds, or ArUco markers |
| `5` | Medical Supply Delivery | Navigate to the active teacher-recorded waypoint |
| `6` | Recycling Sort (SDG 12) | Select likely recyclables and discuss categories |
| `7` | Hazard Inspection | Conservative patrol and possible-hazard reporting |
| `8` | Inventory Search | Patrol with a live classroom object inventory |
| `9` | Follow the Leader | Person-only target lock with conservative clearance |
| `0` | LEGO Treasure Hunt | Search for teacher-selected LEGO/ArUco clues |
| `H` | Return to Base | Navigate toward the recorded mission home position |

Profiles are validated at startup. Edit `scenarios.json` or pass another version-1
profile file:

```powershell
python robomaster_yolo_ai.py --scenarios .\my_scenarios.json
```

Each profile can set its name, shortcut, objective, difficulty, prompts, navigation
policy, preferred/allowed targets, LEGO use, motion permission, cloud/microphone
requirements, and completion condition. Supported navigation policies are `patrol`,
`rescue`, `target`, `lego`, `waypoint`, `return_home`, and `stationary`. Duplicate IDs,
shortcuts, target labels, unsupported policies, and inconsistent preferred targets are
rejected before operation.

## Navigation behavior

```text
DISARMED / PAUSED
        |
        v
SCAN -> ACQUIRING -> TARGET LOCKED -> APPROACH -> STOP / INTERACT
  |          |             |             |
  +----------+--------- RECOVERY <-------+

WAYPOINT / RETURN HOME -> COMPLETE -> DISARMED
Any safety fault --------------------> ESTOP
```

Target motion does not follow whichever same-class box has the highest confidence on
each frame. It maintains one instance using bounding-box overlap, confirms it over
multiple frames, holds still during a short miss, and scans only after the hold period.
A distant same-label distractor cannot silently replace the selected target. The scan
alternates direction and stops after its timeout.

The approach planner stops before advancing when the selected box reaches the configured
close-height ratio. The safety controller independently blocks movement using ToF and
person-clearance checks. Reaching a waypoint or home sends zero motion and disarms.

The mission map is intentionally lightweight, not SLAM or an A* global planner. It uses
RoboMaster odometry and recent ToF endpoints for operator awareness, then drives directly
toward a waypoint with obstacle recovery. Odometry can drift; return-to-base is therefore
an approximate supervised classroom feature.

Record a waypoint with the dashboard or `W`. Start a new run with a fresh home position
by default, or explicitly resume a saved map:

```powershell
python robomaster_yolo_ai.py --resume-map --mission-map .\mission_map.json
```

## Voice, captions, and object reading

- Click **TALK**, press `V` or `Space` in the preview, or press global `F8` on Windows.
- The system records five seconds, transcribes speech, and sends the recent scenario
  conversation to OpenAI for a short spoken response.
- The student transcript and robot response remain visible as captions.
- Click **READ OBJECTS** or press `L` to speak current local detector labels without a
  cloud request.
- Dashboard controls mute/unmute TTS and adjust its rate and volume.
- `N` toggles automatic narration. Recognition and safety stopping remain active when
  narration is off.

Search & Rescue stops as soon as a person is detected, regardless of the narration
setting, and asks “Do you need help?” when narration is enabled. If no person is visible,
it names detected objects or says it is still scanning. A confirmed red LEGO cross also
stops motion and starts the help check. Conversation history is retained separately per
mission for natural back-and-forth turns.

## LEGO support

### Lego-Identification checkpoint

The optional integration uses
[`vsmidhun21/Lego-Identification`](https://github.com/vsmidhun21/Lego-Identification)
`FinalCoShSi.pt`, which labels 34 color/shape/size combinations. Install the pinned,
hash-verified checkpoint locally with:

```powershell
.\setup_lego_identification.ps1
```

The checkpoint runs in a background worker only for relevant missions. Tune or disable it:

```powershell
python robomaster_yolo_ai.py `
  --lego-model-confidence 0.35 `
  --lego-model-interval 0.60 `
  --lego-model-imgsz 512

python robomaster_yolo_ai.py --disable-lego-model
```

The upstream repository had no license file at the pinned revision. Do not redistribute
its weights without permission. PyTorch checkpoints are executable serialization; the
setup pins commit `ddd54ae077a8fed243065a1104ee14eb4aa5f5e2` and verifies SHA256
`87591257D011CC7409CFF14BABF28A1D15402AB521E75F3D10BF5F7A1E013CF6`.

### Help signal and tagged targets

The semantic help pattern is a red 5×5 LEGO cross:

```text
..R..
..R..
RRRRR
..R..
..R..
```

Generate references with:

```powershell
python generate_lego_help_pattern.py --output lego_help_red_cross.png
python generate_lego_marker.py --id 0 --output lego_marker_0.png
```

The red cross must be camera-facing and is confirmed over three consecutive frames.
Plain red squares and rectangles are rejected by the contour test. ArUco markers are a
more reliable autonomous target than an untagged colored brick. LEGO vision never
replaces ToF collision sensing.

## Controls

| Input | Action |
| --- | --- |
| Mouse | Scenario cards, target selection, all dashboard buttons |
| `0`–`9` | Select the corresponding scenario |
| `S` | Open scenario menu; pauses and disarms |
| `V`, `Space`, global `F8` | Start a voice turn |
| `L` | Read visible objects aloud |
| `T` | Cycle through targets allowed by the current profile |
| `W` | Record a waypoint in teacher mode |
| `H` | Select Return to Base in teacher mode |
| `P` | Pause/resume; pausing disarms |
| `M` | Arm/disarm in teacher mode |
| `E`, `Esc`, global `Esc` | Latch emergency stop |
| `R` | Reset emergency latch; remains disarmed |
| `N` | Toggle automatic narration |
| `Q` | Stop and exit |

Student mode cannot arm, reset, record waypoints, or initiate return-to-base. Disarming
and emergency stopping remain available. Scenario and target changes require re-arming.

## Dry-run and physical motion

Exercise navigation and arming logic without hardware:

```powershell
python robomaster_yolo_ai.py --motion-backend dry-run --enable-motion
```

Dry-run reports four simulated 2 m ToF ranges and records commands without moving.

Physical motion requires a compatible Python SDK/driver and four live, uniquely mapped
ToF channels. DJI's public PC SDK documents EP/EP Core support; an S1-compatible driver
must independently expose the same `robomaster` client API. The application does not
automate RoboMaster PC keyboard controls.

The default `.\setup.ps1` command installs the pinned SDK automatically. If setup was
run with `-LocalOnly`, install the robot environment separately with:

```powershell
.\setup_robot_sdk.ps1
.\.venv-robot\Scripts\Activate.ps1
```

After wheel-off-ground validation:

```powershell
python robomaster_yolo_ai.py `
  --motion-backend robomaster `
  --conn-type ap `
  --tof-layout front,left,right,rear `
  --min-tof-count 4 `
  --enable-motion
```

Optional gimbal tracking is a separate opt-in:

```powershell
python robomaster_yolo_ai.py `
  --motion-backend robomaster `
  --enable-motion `
  --enable-gimbal-tracking
```

Verify pitch/yaw direction with the chassis raised before floor operation. A gimbal SDK
error triggers the same latched emergency-stop path as a chassis command failure.

### Safety interlocks

- Motion is disabled by default and starts disarmed even with `--enable-motion`.
- Arming requires fresh, positive readings from the configured number of ToF sensors.
- ToF direction names must be supported and unique; duplicate mappings fail startup.
- Forward, reverse, lateral, and rotation commands each check the relevant clearance.
- A large person in the forward visual corridor independently blocks forward motion.
- A selected target that is visually close stops the approach even if a low LEGO object
  is below the front ToF beam.
- Commands are clamped to 0.12 m/s and 12°/s and use a 0.35 s SDK timeout.
- Asynchronous impact stops plus chassis and gimbal command sends share one lock,
  preventing a stale command from restarting an actuator after a safety callback.
- Emergency state is latched before SDK stop calls; a failed SDK call cannot leave the
  controller logically armed.
- Impact logging is idempotent, the vision-loop watchdog stops on a stall, and closing
  the preview stops operation.
- Mission selection, target changes, pause, student mode, scenario menu, rescue person,
  help signal, and mission completion all disarm or stop as appropriate.

Mission events are appended to `mission_events.jsonl`; maps are saved to
`mission_map.json`. Both paths are configurable.

## Architecture

| File | Responsibility |
| --- | --- |
| `robomaster_yolo_ai.py` | Runtime orchestration, screen capture, detectors, voice workers, UI actions |
| `autonomy.py` | Hardware/dry-run backends, safety controller, low-speed planner, mission log |
| `navigation.py` | Target tracker, navigation state machine, gimbal planner, waypoints/map |
| `dashboard.py` | OpenCV dashboard rendering and mouse hit-testing |
| `scenario_profiles.py` | Scenario schema, validation, and catalog |
| `scenarios.json` | User-editable mission definitions |
| `lego_vision.py` | OpenCV LEGO, ArUco, and red-cross recognition |
| `lego_identification.py` | Optional asynchronous third-party model wrapper |

## Tests

```powershell
python -m py_compile autonomy.py navigation.py dashboard.py scenario_profiles.py `
  lego_vision.py robomaster_yolo_ai.py
python -m unittest discover -s tests -v
```

The tests cover safety serialization, stop failures, impact latching, range-direction
validation, visual close stops, rescue/narration independence, target persistence,
scan timeout, obstacle recovery, map round trips, gimbal commands, dashboard clicks,
compact rendering, TTS failure handling, legacy ArUco, preview closure, and local-only
startup. They do **not** certify physical motion; hardware behavior must be validated on
the exact robot, sensors, firmware, SDK, surface, and test area.

The Windows GitHub Actions workflow runs the compile and unit-test commands on Python
3.11 and on the Python 3.8 robot-control environment. The Python 3.8 job also installs
`requirements-robot.txt` and verifies the native RoboMaster SDK import.
