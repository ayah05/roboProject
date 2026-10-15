"""
Train a LeRobot ACT policy on a cloud GPU with Modal.

The dataset is downloaded from the Hugging Face Hub, training runs on Modal,
and the finished policy is uploaded back to the Hub automatically.
Checkpoints are also kept in the Modal volume "lerobot-outputs".

One-time setup (on your laptop):
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx   # token with write access

Run training (uses the defaults at the bottom of this file):
    modal run --detach training/train_modal.py

Optional overrides:
    modal run --detach training/train_modal.py --steps 20000
    modal run --detach training/train_modal.py --dataset ayah055/other_dataset --policy ayah055/other_policy

--detach keeps the training running even if you close the terminal.
Watch progress in the Modal dashboard (link is printed when it starts).
"""

import subprocess
from datetime import datetime

import modal

# Use the same LeRobot version as on your laptop, otherwise the trained policy
# may not load locally. Get it with:  git -C lerobot rev-parse HEAD
# Leave as "main" to use the latest version.
LEROBOT_COMMIT = "main"

GPU = "A100"  # "L40S" is cheaper and also fine for ACT

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/huggingface/lerobot.git /lerobot",
        f"cd /lerobot && git checkout {LEROBOT_COMMIT}",
        # The [training] extra adds the dataset tools (datasets, pyarrow, av, ...)
        # that lerobot-train needs. A plain "pip install -e ." is not enough.
        'cd /lerobot && pip install -e ".[training]"',
    )
)

app = modal.App("lerobot-act-training", image=image)
outputs = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)


@app.function(
    gpu=GPU,
    timeout=8 * 60 * 60,  # 8 hours max
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/outputs": outputs},
)
def train(dataset: str, policy_repo: str, steps: int, save_freq: int):
    # LeRobot refuses to overwrite an existing output folder, so make each run unique
    job_name = f"act_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = f"/outputs/train/{job_name}"
    print(f"Training {dataset} -> {policy_repo}, output in {output_dir}")

    cmd = [
        "lerobot-train",
        "--policy.type=act",
        f"--dataset.repo_id={dataset}",
        "--policy.device=cuda",
        f"--policy.repo_id={policy_repo}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        # Keep checkpoints even if training crashes or is stopped
        outputs.commit()
    print(f"Done. Policy uploaded to https://huggingface.co/{policy_repo}")


@app.local_entrypoint()
def main(
    dataset: str = "ayah055/cup_task_20261014_213957",
    policy: str = "ayah055/cup_policy",
    steps: int = 30000,
    save_freq: int = 5000,
):
    train.remote(dataset, policy, steps, save_freq)