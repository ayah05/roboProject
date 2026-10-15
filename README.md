# HandOver

HandOver is an affordable assistive robot system that helps people with paralysis, ALS, or stroke-related motor impairments perform everyday reaching and grasping tasks by voice command. Instead of steering a robot arm step-by-step with a joystick, the user simply says what they need (for example, “Give me the cup”), and the system understands, verifies, and performs the action autonomously.

## Key ideas

- Natural voice commands — Whisper converts spoken requests to text.
- Perception verification — YOLO checks the camera feed to confirm the requested object is present and estimates its location.
- Learned manipulation — an ACT policy trained with LeRobot executes the grasp; motions are learned from demonstrations so the robot adapts to where the object actually is rather than following a fixed scripted trajectory.
- Low-cost, open-source hardware — prototype built on the SO101 arm and runs on a regular laptop.
- Easy task creation — new tasks are added by demonstrating them, enabling caregivers to teach the robot tasks specific to a user’s home.

## Features

- Speech-to-text command processing
- Real-time object detection and verification
- Demonstration-driven learned policies for adaptive grasps
- Laptop-executable (no expensive proprietary controllers required)
- Designed for easy extension with new tasks and demonstrations

## How it works (overview)

1. The user speaks a request (e.g., “Give me the cup”).
2. Whisper transcribes the audio into text and the system extracts the intended object/task.
3. YOLO inspects the camera view to confirm the object is present and provides its pose/position estimate.
4. The learned ACT policy (trained with LeRobot) plans and executes a grasp adapted to the actual object pose.
5. The robot performs the handover to the user under supervision.

## Hardware

- Prototype uses the open-source SO101 arm (low-cost, research prototype).
- The software architecture is hardware-agnostic: for real-world deployments the same stack can run on a certified, industrial/medical-grade arm.

## Software requirements (example)

- Python 3.8+ (project is Python-based)
- GPU recommended for model inference and training (but can run on CPU for evaluation)
- Webcam or camera compatible with the laptop
- Install project dependencies: pip install -r requirements.txt
(See the repository for an up-to-date requirements file and hardware setup instructions.)

## Quick start (example)

1. Clone the repo:
   git clone https://github.com/ayah05/roboProject.git
2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Calibrate camera and arm according to the repo docs (see docs/ or scripts/ for calibration tools).
5. Run the main application (example placeholder):
   python run.py
6. Speak a command such as “Give me the cup” and follow on-screen prompts.

(Actual filenames and commands may differ — check the repository for exact usage and scripts.)

## Adding new tasks

- Record demonstrations of the task (teleoperation/kinesthetic or a guidance tool).
- Collect several examples from slightly different object positions so the policy can generalize.
- Train a new ACT policy using the provided training pipeline (LeRobot-based training).
- Validate the trained policy in the simulator or on the SO101 prototype before live use.

See the repository’s training and data folders for scripts and a recommended demo data format.

## Safety, limitations, and disclaimers

- This project is a research prototype. It is not certified for medical or independent unsupervised use.
- Always supervise the robot during operation, especially during initial deployments and when testing new tasks.
- Physical interactions may cause harm if not properly constrained, so limit motions in shared spaces and follow safe-startup calibration and emergency-stop procedures.

## Contributing

Contributions, issue reports, and improvements are welcome. Please:
1. Open an issue describing the change or bug.
2. Fork the project and make your changes on a feature branch.
3. Submit a pull request with a clear description and tests/examples when relevant.

See CONTRIBUTING.md (if present) for more details.

## License

See the LICENSE file in this repository for license terms.

## Acknowledgements

- Built on the SO101 open-source arm hardware.
- Uses Whisper for speech-to-text and YOLO for object detection.
- Policy learning performed with LeRobot tooling.

## Contact

For questions, collaboration, or support, please open an issue in this repository or contact the maintainers listed in the project metadata.
