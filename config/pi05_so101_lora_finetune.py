# Fine-tune pi0.5 on a real SO-101 arm with dual-expert LoRA.
#
# This block is added to src/openpi/training/config.py of the openpi codebase
# (Physical-Intelligence/openpi). It starts from the public pi05_base weights,
# inserts LoRA adapters into both the PaliGemma 3B VLM and the 300M action
# expert, and trains on a local LeRobot v2.1 dataset recorded from
# teleoperated demonstrations.

from openpi.training.config import TrainConfig
from openpi.models import pi0_config
from openpi.training import weight_loaders

TrainConfig(
    name="pi05_so101_lora_finetune",
    model=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",      # LoRA into the 3B VLM
        action_expert_variant="gemma_300m_lora", # LoRA into the action expert
    ),
    data=LeRobotSO101DataConfig(
        repo_id="local/record-breakfast_49src_v21",
        assets=AssetsConfig(asset_id="local/record-breakfast_49src_v21"),
        default_prompt=(
            "first put the block onto the plate, move the plate to the center "
            "of the table, place the spoon on the right side of the plate, "
            "and place the cup on the left side of the plate."
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),   # everything except LoRA params stays frozen
    ema_decay=None,
    num_train_steps=30_000,
)
