#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""把 HIL-SERL actor 推理链裁剪成 AIC 可直接调用的运行时。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .config import HilSerlRuntimeConfig
from .types import HilSerlAction, ObservationDict


class HilSerlActorRuntime:
    """AIC 侧最小 HIL-SERL actor 运行时。

    迁移自 `train_rlpd.py --actor` 的核心路径：
    - 创建 agent
    - 从 checkpoint 恢复参数
    - 调 `sample_actions()` 执行单步推理
    """

    def __init__(self, config: HilSerlRuntimeConfig):
        self._config = config
        self._agent = None
        self._sampling_rng = None
        self._jax = None
        self._jnp = None
        self._checkpoints = None
        self._initialized = False

    def initialize(self, sample_observation: ObservationDict) -> None:
        if self._initialized:
            return

        self._ensure_import_paths()

        import jax
        import jax.numpy as jnp
        from flax.training import checkpoints
        from serl_launcher.utils.launcher import (
            make_bc_agent,
            make_sac_pixel_agent,
            make_sac_pixel_agent_hybrid_single_arm,
        )

        self._jax = jax
        self._jnp = jnp
        self._checkpoints = checkpoints

        rng = jax.random.PRNGKey(self._config.model.seed)
        rng, self._sampling_rng = jax.random.split(rng)

        sample_action = jnp.zeros((self._config.model.action_dim,), dtype=jnp.float32)
        sample_obs = {k: jnp.asarray(v) for k, v in sample_observation.items()}

        if self._config.model.setup_mode == "single-arm-fixed-gripper":
            self._agent = make_sac_pixel_agent(
                seed=self._config.model.seed,
                sample_obs=sample_obs,
                sample_action=sample_action,
                image_keys=self._config.model.image_keys,
                encoder_type=self._config.model.encoder_type,
            )
        elif self._config.model.setup_mode == "single-arm-learned-gripper":
            self._agent = make_sac_pixel_agent_hybrid_single_arm(
                seed=self._config.model.seed,
                sample_obs=sample_obs,
                sample_action=sample_action,
                image_keys=self._config.model.image_keys,
                encoder_type=self._config.model.encoder_type,
            )
        elif self._config.model.setup_mode == "bc":
            self._agent = make_bc_agent(
                seed=self._config.model.seed,
                sample_obs=sample_obs,
                sample_action=sample_action,
                image_keys=self._config.model.image_keys,
                encoder_type=self._config.model.encoder_type,
            )
        else:
            raise NotImplementedError(
                f"Unsupported HIL-SERL setup mode: {self._config.model.setup_mode}"
            )

        checkpoint_path = Path(self._config.model.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"HIL-SERL checkpoint path does not exist: {checkpoint_path}"
            )

        if self._config.model.checkpoint_step > 0:
            ckpt = checkpoints.restore_checkpoint(
                checkpoint_path.resolve(),
                self._agent.state,
                step=self._config.model.checkpoint_step,
            )
        else:
            ckpt = checkpoints.restore_checkpoint(
                checkpoint_path.resolve(),
                self._agent.state,
            )
        self._agent = self._agent.replace(state=ckpt)
        self._initialized = True

    def reset(self) -> None:
        if self._jax is None:
            return
        rng = self._jax.random.PRNGKey(self._config.model.seed)
        _, self._sampling_rng = self._jax.random.split(rng)

    def predict(self, observation: ObservationDict) -> HilSerlAction:
        if not self._initialized:
            raise RuntimeError(
                "HilSerlActorRuntime must be initialized before predict()."
            )

        self._sampling_rng, key = self._jax.random.split(self._sampling_rng)
        obs = {
            k: self._jax.device_put(self._jnp.asarray(v))
            for k, v in observation.items()
        }
        actions = self._agent.sample_actions(
            observations=obs,
            seed=key,
            argmax=self._config.model.argmax,
        )
        action_np = np.asarray(self._jax.device_get(actions), dtype=np.float32)
        return HilSerlAction(values=action_np)

    def _ensure_import_paths(self) -> None:
        hil_serl_root = str(self._config.hil_serl_root)
        examples_root = str(self._config.hil_serl_root / "examples")
        launcher_root = str(self._config.hil_serl_root / "serl_launcher")
        robot_infra_root = str(self._config.hil_serl_root / "serl_robot_infra")

        for path in [hil_serl_root, examples_root, launcher_root, robot_infra_root]:
            if path not in sys.path:
                sys.path.insert(0, path)
