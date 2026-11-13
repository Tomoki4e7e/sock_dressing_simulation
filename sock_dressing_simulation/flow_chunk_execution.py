"""Controlled Flow chunk execution with noise indexed by observation time."""
import hashlib

import torch


class FlowChunkExecutor:
    def __init__(self, model, low, high, mimic_multipliers, mimic_offsets,
                 ode_steps=8, execute_steps=1, seed=0):
        if ode_steps < 1 or execute_steps not in (1, 2, 4) or execute_steps > model.horizon:
            raise ValueError('invalid Flow integration/execution horizon')
        self.model = model
        self.low, self.high = low, high
        self.multipliers, self.offsets = mimic_multipliers, mimic_offsets
        self.ode_steps, self.execute_steps, self.seed = ode_steps, execute_steps, seed
        self.generator = None
        self.plan = None
        self.plan_start = None
        self.next_step = 0

    @torch.no_grad()
    def step(self, condition, normalized_angles, index):
        if index != self.next_step:
            raise ValueError('observations must be processed consecutively from zero')
        self.next_step += 1
        if self.generator is None:
            self.generator = torch.Generator(device=condition.device).manual_seed(self.seed)
        # Draw at EVERY observation, even while consuming an existing plan.
        # For execute_steps=1 this is exactly the original sampler's draw order.
        noise = torch.randn((condition.shape[0], self.model.horizon, self.model.action_dim),
                            device=condition.device, dtype=condition.dtype, generator=self.generator)
        fingerprint = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
        replanned = self.plan is None or index - self.plan_start >= self.execute_steps
        diagnostics = dict(replanned=replanned, ode_steps=self.ode_steps,
                           execute_steps=self.execute_steps, observation_index=index,
                           noise_sha256=fingerprint, plan_change_arm_rms_rad=None,
                           plan_change_finger_rms_m=None)
        if replanned:
            previous, previous_start = self.plan, self.plan_start
            offsets = self.model.sample_offsets(condition, steps=self.ode_steps, initial_noise=noise)
            # Keep this anchor for every slot; never re-anchor held offsets.
            angles = self.model.offsets_to_angles(offsets.float(), normalized_angles)
            physical = self.model.denormalize_actions(angles, self.low, self.high)
            self.plan = self.model.expand_mimic_joints(physical, self.multipliers, self.offsets)
            self.plan_start = index
            if previous is not None:
                same_time_slot = index - previous_start
                if same_time_slot < self.model.horizon:
                    delta = self.plan[:, 0] - previous[:, same_time_slot]
                    arms = list(range(7)) + list(range(9, 16))
                    diagnostics['plan_change_arm_rms_rad'] = delta[:, arms].square().mean().sqrt().item()
                    diagnostics['plan_change_finger_rms_m'] = delta[:, [7, 16]].square().mean().sqrt().item()
            diagnostics['planned_angles18'] = self.plan[0].cpu().tolist()
        slot = index - self.plan_start
        diagnostics.update(plan_start_observation=self.plan_start, plan_slot=slot)
        return self.plan[:, slot], diagnostics
