from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import random

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.utils.math import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

import time

import sys
sys.path.append(os.path.join(ASE_DIR, "ase"))
sys.path.append(os.path.join(ASE_DIR, "ase/utils"))
import cv2

from motion_lib import MotionLib
import torch_utils

class G1MimicPriv(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)

        # randomize demo velocity to augment the dataset
        self.cfg.g1_params.vel_factor = 1.0
        
        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        self.init_motions(cfg)
        if cfg.motion.num_envs_as_motions:
            self.cfg.env.num_envs = self._motion_lib.num_motions()
        
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self.obs_extra_history_buf = torch.zeros(self.num_envs, self.cfg.env.extra_history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.init_motion_buffers(cfg)

        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()

    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, 7:7+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, 7+self.num_dof:7+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec
    
    def init_motions(self, cfg):
        self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  #self._build_key_body_ids_tensor(key_bodies)
        # ['pelvis', 
        # 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 
        # 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 
        # 'waist_yaw_link', 'waist_roll_link', 'torso_link',
        # 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_rubber_hand', 
        # 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_rubber_hand']
        self._key_body_ids_sim = torch.tensor([1, 4, 5, # Left Hip yaw, Knee, Ankle
                                               7, 10, 11,
                                               16, 19, 20, # Left Shoulder pitch, Elbow, hand
                                               21, 24, 25], device=self.device)
        # self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        self._key_body_ids_sim_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)  # no ankle
        
        self._num_key_bodies = len(self._key_body_ids_sim_subset)

        # ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
        # 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
        # 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
        # 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
        # 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 
        # 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
        # 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
        self.dof_indices_sim = torch.tensor(   [0, 1, 2,  4, 5,  6, 7, 8,  10, 11,  12, 13, 14,  15, 16, 17,  20, 21, 22], device=self.device, dtype=torch.long)
        
        self._dof_ids_subset = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9,
                                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], device=self.device)  # wholebody

        self._n_demo_dof = len(self._dof_ids_subset)

        # exbody2
        self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                              4, 5, 6,
                              7,       # Torso
                              8, 9, 10, # Shoulder, Elbow, Hand
                              11, 12, 13]  # 13

        self._dof_offsets = [0, 3, 4, 6, 9, 10, 12, 
                             15, 
                             18, 19, 20, 23, 24, 25]  # 14

        self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+2*6, device=self.device, dtype=torch.bool)
        self._valid_dof_body_ids[-1] = 0
        self._valid_dof_body_ids[-6] = 0
        self.dof_indices_motion = torch.tensor([1, 0, 2,  4, 5,  7, 6, 8,  10, 11,  14, 12, 13,  16, 15, 17,  21, 20, 22], device=self.device, dtype=torch.long)

        if cfg.motion.motion_type == "single":
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/g1_retarget_npy/{cfg.motion.motion_name}.npy")
        else:
            assert cfg.motion.motion_type == "yaml"
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")
        
        self._load_motion(motion_file, cfg.motion.no_keybody)

    def init_motion_buffers(self, cfg):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        if cfg.motion.motion_curriculum:
            self._max_motion_difficulty = 9
        else:
            self._max_motion_difficulty = 9
        self._motion_times = self._motion_lib.sample_time(self._motion_ids)
        self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self._motion_difficulty = self._motion_lib.get_motion_difficulty(self._motion_ids)

        self._motion_left_ground_offsets = self._motion_lib.get_motion_left_ground_offset(self._motion_ids)
        self._motion_right_ground_offsets = self._motion_lib.get_motion_right_ground_offset(self._motion_ids)

        self._motion_dt = self.dt * 1.2 # a little bit tricky for data augmentation
        self._motion_num_future_steps = self.cfg.env.n_demo_steps
        self._motion_demo_offsets = torch.arange(0, self.cfg.env.n_demo_steps * self.cfg.env.interval_demo_steps, self.cfg.env.interval_demo_steps, device=self.device)
        self._demo_obs_buf = torch.zeros((self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo), device=self.device)
        self._curr_demo_obs_buf = self._demo_obs_buf[:, 0, :]
        self._next_demo_obs_buf = self._demo_obs_buf[:, 1, :]

        self._curr_demo_root_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._curr_demo_root_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_keybody = torch.zeros((self.num_envs, self._num_key_bodies, 3), device=self.device)
        self._in_place_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.dof_term_threshold = 3 * torch.ones(self.num_envs, device=self.device)
        self.keybody_term_threshold = 0.3 * torch.ones(self.num_envs, device=self.device)
        self.yaw_term_threshold = 0.5 * torch.ones(self.num_envs, device=self.device)
        self.height_term_threshold = 0.2 * torch.ones(self.num_envs, device=self.device)

        # contact info
        self.left_foot_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.right_foot_contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    
    def _load_motion(self, motion_file, no_keybody=False):
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device, 
                                     no_keybody=no_keybody, 
                                     regen_pkl=self.cfg.motion.regen_pkl)
        return
    
    def step(self, actions):
        actions = self.reindex(actions)
        actions.to(self.device)
        
        indices = -1 * torch.ones((self.num_envs,), device=self.device, dtype=torch.long)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter > self.cfg.domain_rand.delay_update_global_steps:
                self.delay = torch.randint(0, 10, (self.num_envs,), device=self.device, dtype=torch.long)
            else:
                self.delay = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
            if self.viewer:
                self.delay = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
            indices = -1 - self.delay

        self.global_counter += 1
        self.total_env_steps_counter += 1

        self.render()
        
        for _ in range(self.cfg.control.decimation):
            clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
            actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:, :].clone(), actions[:, None, :].clone()], dim=1)
            self.actions = self.action_history_buf[torch.arange(self.num_envs, device=self.device, dtype=torch.long), indices].squeeze()
            
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def resample_motion_times(self, env_ids):
        return self._motion_lib.sample_time(self._motion_ids[env_ids])
    
    def update_motion_ids(self, env_ids):
        self._motion_times[env_ids] = self.resample_motion_times(env_ids)
        self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
        self._motion_difficulty[env_ids] = self._motion_lib.get_motion_difficulty(self._motion_ids[env_ids])
        self._motion_left_ground_offsets[env_ids] = self._motion_lib.get_motion_left_ground_offset(self._motion_ids[env_ids])
        self._motion_right_ground_offsets[env_ids] = self._motion_lib.get_motion_right_ground_offset(self._motion_ids[env_ids])

    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return
        # Not used
        if self.cfg.motion.motion_curriculum:
            # ep_length = self.episode_length_buf[env_ids] * self.dt
            completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lengths[env_ids]
            completion_rate_mean = completion_rate.mean()
            relax_ids = completion_rate < 0.3
            strict_ids = completion_rate > 0.9
            # self.dof_term_threshold[env_ids[relax_ids]] += 0.05
            self.dof_term_threshold[env_ids[strict_ids]] -= 0.05
            self.dof_term_threshold.clamp_(1.5, 3)

            self.height_term_threshold[env_ids[relax_ids]] += 0.01
            self.height_term_threshold[env_ids[strict_ids]] -= 0.01
            self.height_term_threshold.clamp_(0.03, 0.1)

            relax_ids = completion_rate < 0.6
            strict_ids = completion_rate > 0.9
            self.keybody_term_threshold[env_ids[relax_ids]] -= 0.05
            self.keybody_term_threshold[env_ids[strict_ids]] += 0.05
            self.keybody_term_threshold.clamp_(0.1, 0.4)

            relax_ids = completion_rate < 0.4
            strict_ids = completion_rate > 0.8
            self.yaw_term_threshold[env_ids[relax_ids]] -= 0.05
            self.yaw_term_threshold[env_ids[strict_ids]] += 0.05
            self.yaw_term_threshold.clamp_(0.1, 0.6)


        self.update_motion_ids(env_ids)

        motion_ids = self._motion_ids[env_ids]
        motion_times = self._motion_times[env_ids]
        root_pos, root_rot, dof_pos_motion, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        # Intialize dof state from default position and reference position
        dof_pos_motion, dof_vel = self.reindex_dof_pos_vel(dof_pos_motion, dof_vel)

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # # clip initial height
        # self.cfg.g1_params.vel_factor = random.uniform(0.0, 1.5)
        # if self.cfg.g1_params.vel_factor < 0.2:
        #     self.cfg.g1_params.vel_factor = 0.0
        
        root_pos[:, :2] = self.cfg.g1_params.height_factor * root_pos[:, :2]
        root_pos[:, 2] = torch.clamp(root_pos[:, 2], self.cfg.g1_params.min_init_height, self.cfg.g1_params.max_init_height)
        # clip initial velocity
        root_vel = self.cfg.g1_params.height_factor * root_vel * self.cfg.g1_params.vel_factor
        root_vel[:, 0] = torch.clamp(root_vel[:, 0], -self.cfg.g1_params.max_vel, self.cfg.g1_params.max_vel)
        root_vel[:, 1] = torch.clamp(root_vel[:, 1], -self.cfg.g1_params.max_vel, self.cfg.g1_params.max_vel)
        root_vel[:, 2] = torch.clamp(root_vel[:, 2], -self.cfg.g1_params.max_vel, self.cfg.g1_params.max_vel)

        # reset robot states
        self._reset_dofs(env_ids, dof_pos_motion, dof_vel)
        self._reset_root_states(env_ids, root_vel, root_rot, root_pos[:, 2])

        if init:
            self.init_root_pos_global = self.root_states[:, :3].clone()
            self.init_root_pos_global_demo = root_pos[:].clone()
            self.target_pos_abs = self.init_root_pos_global.clone()[:, :2]
        else:
            self.init_root_pos_global[env_ids] = self.root_states[env_ids, :3].clone()
            self.init_root_pos_global_demo[env_ids] = root_pos[:].clone()
            self.target_pos_abs[env_ids] = self.init_root_pos_global[env_ids].clone()[:, :2]

        self._resample_commands(env_ids)  # no resample commands
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.obs_extra_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        self.extras["episode"]["curriculum_completion"] = completion_rate_mean
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        self.extras["episode"]["curriculum_motion_difficulty_level"] = self._max_motion_difficulty
        self.extras["episode"]["curriculum_dof_term_thresh"] = self.dof_term_threshold.mean()
        self.extras["episode"]["curriculum_keybody_term_thresh"] = self.keybody_term_threshold.mean()
        self.extras["episode"]["curriculum_yaw_term_thresh"] = self.yaw_term_threshold.mean()
        self.extras["episode"]["curriculum_height_term_thresh"] = self.height_term_threshold.mean()
        
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return
                                                                                                                                                                                                                                                                                                                                                                   
    def _reset_dofs(self, env_ids, dof_pos, dof_vel): 
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def post_physics_step(self):
        # self._motion_sync()
        super().post_physics_step()

        # step motion lib
        self._motion_times += self._motion_dt
        self._motion_times[self._motion_times >= self._motion_lengths] = 0.
        self.update_demo_obs()
        # self.update_mimic_obs()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_rigid_bodies_demo()

        return

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if self.common_step_counter % self.cfg.motion.resample_step_inplace_interval == 0:
            self.resample_step_inplace_ids()
    
    def resample_step_inplace_ids(self, ):
        self.step_inplace_ids = torch.rand(self.num_envs, device=self.device) < self.cfg.motion.step_inplace_prob
    
    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.motion.resample_step_inplace_interval = np.ceil(self.cfg.motion.resample_step_inplace_interval_s / self.dt)

    def _update_goals(self):
        reset_target_pos = self.episode_length_buf % (self.cfg.motion.global_keybody_reset_time // self.dt) == 0
        self.target_pos_abs[reset_target_pos] = self.root_states[reset_target_pos, :2]
        self.target_pos_abs += (self._curr_demo_root_vel * self.dt)[:, :2]
        self.target_pos_rel = global_to_local_xy(self.yaw[:, None], self.target_pos_abs - self.root_states[:, :2])
        r, p, y = euler_from_quaternion(self._curr_demo_quat)
        self.target_yaw = y.clone()

    
    def update_demo_obs(self):
        demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [num_envs, demo_dim]
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_key_body_pos \
            = self._motion_lib.get_motion_state(self._motion_ids.repeat_interleave(self._motion_num_future_steps), demo_motion_times.flatten(), get_lbp=True)
        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
        
        # clip initial velocity
        root_pos[:, :2] = self.cfg.g1_params.height_factor * root_pos[:, :2]
        root_vel = self.cfg.g1_params.height_factor * root_vel * self.cfg.g1_params.vel_factor
        root_vel[:, 0] = torch.clamp(root_vel[:, 0], -self.cfg.g1_params.max_vel, self.cfg.g1_params.max_vel)
        root_vel[:, 1] = torch.clamp(root_vel[:, 1], -self.cfg.g1_params.max_vel, self.cfg.g1_params.max_vel)
        root_vel[:, 2] = torch.clamp(root_vel[:, 2], -self.cfg.g1_params.max_vel, self.cfg.g1_params.max_vel)

        self._curr_demo_root_pos[:] = root_pos.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
        self._curr_demo_quat[:] = root_rot.view(self.num_envs, self._motion_num_future_steps, 4)[:, 0, :]
        self._curr_demo_root_vel[:] = root_vel.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]

        self._curr_demo_keybody[:] = local_key_body_pos[:, self._key_body_ids_sim_subset].view(self.num_envs, self._motion_num_future_steps, self._num_key_bodies, 3)[:, 0, :, :]
        demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos[:, self._dof_ids_subset], dof_vel, key_pos, local_key_body_pos[:, self._key_body_ids_sim_subset, :], self._dof_offsets)
        self._demo_obs_buf[:] = demo_obs.view(self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo)[:]

        left_feet_height = self._curr_demo_keybody[:, 2, 2] + self._curr_demo_root_pos[:, 2]
        right_feet_height = self._curr_demo_keybody[:, 5, 2] + self._curr_demo_root_pos[:, 2]

        curr_left_ground_offset = self._motion_left_ground_offsets[:]
        curr_right_ground_offset = self._motion_right_ground_offsets[:]

        # set a tolerace offset for the ground
        tolerance = 0.02
        self.left_foot_contact_buf[:] = left_feet_height < curr_left_ground_offset + tolerance
        self.right_foot_contact_buf[:] = right_feet_height < curr_right_ground_offset + tolerance

    def compute_obs_buf(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        dof_vel = self.dof_vel.clone()
        dof_vel[:, [4, 5, 10, 11, 13, 14]] = 0.0
        return torch.cat((#motion_id_one_hot,
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            imu_obs,    #[1,2]
                            torch.sin(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            torch.cos(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            self.reindex_feet(self.contact_filt.float()*0-0.5),
                            ),dim=-1)
    
    def compute_obs_demo(self):
        obs_demo = self._next_demo_obs_buf.clone()
        obs_demo[self._in_place_flag, self._n_demo_dof:self._n_demo_dof+3] = 0
        return obs_demo
    
    def compute_teacher_priv(self):
        demo_key_body_pos_local = self._curr_demo_keybody.view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)

        demo_global_body_pos = local_to_global(self._curr_demo_quat, demo_key_body_pos_local, curr_demo_xyz).view(self.num_envs, -1)
        cur_global_body_pos = self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3].view(self.num_envs, -1)
        key_body_diff = demo_global_body_pos - cur_global_body_pos

        cur_local_key_body_pos = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1)

        return torch.cat((key_body_diff, 
                        cur_local_key_body_pos,
                        self.base_lin_vel), dim=-1)       
    
    def compute_observations(self):
        
        obs_demo = self.compute_obs_demo()
        obs_buf = self.compute_obs_buf()
        obs_teacher_priv = self.compute_teacher_priv()

        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale
        
        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)
        priv_explicit = torch.cat((0*self.base_lin_vel * self.obs_scales.lin_vel,), dim=-1)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_teacher_priv, obs_demo, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_teacher_priv ,obs_demo, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.obs_extra_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.extra_history_len, dim=1),
            torch.cat([
                self.obs_extra_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, self._motion_times)
        
        root_pos[:, :2] = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        height_cutoff = (self.root_states[:, 2] < 0.2)

        '''
        Use this line to control motions resampling frequency. 
        Use time_out_buf can accelerate training, but may lead to unstable training.
        '''
        # self.time_out_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= height_cutoff
    
    ######### utils #########
    
    def reindex_dof_pos_vel(self, dof_pos, dof_vel):
        dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        return dof_pos, dof_vel

    def draw_rigid_bodies_demo(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 1, 0))
        local_body_pos = self._curr_demo_keybody.clone().view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)

        global_body_pos = local_to_global(self._curr_demo_quat, local_body_pos, curr_demo_xyz)
        for i in range(global_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, i, 0], global_body_pos[self.lookat_id, i, 1], global_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

        # use contact info to draw feet
        geom_feet = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 0, 1))
        if self.left_foot_contact_buf[self.lookat_id]:
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, 2, 0], global_body_pos[self.lookat_id, 2, 1], global_body_pos[self.lookat_id, 2, 2]), r=None)
            gymutil.draw_lines(geom_feet, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        if self.right_foot_contact_buf[self.lookat_id]:
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, 5, 0], global_body_pos[self.lookat_id, 5, 1], global_body_pos[self.lookat_id, 5, 2]), r=None)
            gymutil.draw_lines(geom_feet, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_goals(self, ):
        demo_geom = gymutil.WireframeSphereGeometry(0.2, 32, 32, None, color=(1, 0, 0))
        
        pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            norm = torch.norm(self._curr_demo_root_vel[:, :2], dim=-1, keepdim=True)
            target_vec_norm = self._curr_demo_root_vel[:, :2] / (norm + 1e-5)
            for i in range(5):
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    
    ######### Rewards #########
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    
    def _reward_tracking_ang_vel(self):
        rew = torch.minimum(self.base_ang_vel[:, 2], self.commands[:, 2]) / (self.commands[:, 2] + 1e-5)
        return rew
    
    def _reward_tracking_demo_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        return rew

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_demo_dof_pos(self):
        demo_dofs = self._curr_demo_obs_buf[:, :self._n_demo_dof]
        dof_pos = self.dof_pos[:, self._dof_ids_subset]
        rew = torch.exp(-0.7 * torch.norm((dof_pos - demo_dofs), dim=1))
        return rew
    
    def _reward_stand_still(self):
        dof_pos_error = torch.norm((self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
        dof_vel_error = torch.norm(self.dof_vel[:, :11], dim=1)
        rew = torch.exp(- 0.1*dof_vel_error) * torch.exp(- dof_pos_error) 
        rew[~self._in_place_flag] = 0
        return rew
    
    def _reward_tracking_lin_vel(self):
        demo_vel = self._curr_demo_obs_buf[:, self._n_demo_dof:self._n_demo_dof+3]
        demo_vel[self._in_place_flag] = 0
        rew = torch.exp(- 4 * torch.norm(self.base_lin_vel - demo_vel, dim=1))
        return rew


    def _reward_tracking_demo_roll_pitch(self):
        demo_roll_pitch = self._curr_demo_obs_buf[:, self._n_demo_dof+6:self._n_demo_dof+8]
        cur_roll_pitch = torch.stack((self.roll, self.pitch), dim=1)
        rew = torch.exp(-torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1))
        return rew
    
    def _reward_tracking_demo_key_body(self):
        demo_key_body_pos_local = self._curr_demo_keybody.view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
        demo_global_body_pos = local_to_global(self._curr_demo_quat, demo_key_body_pos_local, curr_demo_xyz).view(self.num_envs, -1)
        cur_global_body_pos = self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3].view(self.num_envs, -1)

        rew = torch.exp(-torch.norm(cur_global_body_pos - demo_global_body_pos, dim=1))
        return rew
    
    def _reward_energy(self):
        return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= ~contact_filt
        rew_airTime[self._in_place_flag] = 0
        return rew_airTime

    def _reward_feet_force(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < 500] = 0
        rew[rew > 500] -= 500
        rew[self._in_place_flag] = 0
        return rew

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :12], dim=1)
        return dof_error

    def _reward_ankle_action(self):
        return torch.norm(self.action_history_buf[:, -1, [4,5,10,11]], dim=1)
    
    def _reward_waist_roll_pitch_error(self):
        waist_roll_pitch_idx = torch.tensor([13, 14], device=self.device)
        waist_roll_pitch_dof_error = torch.sum(torch.square(
            self.dof_pos[:, waist_roll_pitch_idx]- self.default_dof_pos[:, waist_roll_pitch_idx]), dim=1)
        return waist_roll_pitch_dof_error
    
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            if not self.free_cam:
                self.lookat(self.lookat_id)
            # check for keyboard events
            evt_count = 0
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                
                if not self.free_cam:
                    for i in range(9):
                        if evt.action == "lookat" + str(i) and evt.value > 0:
                            self.lookat(i)
                            self.lookat_id = i
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "vx_plus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] += 0.1
                    if evt.action == "vx_minus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] -= 0.1
                    if evt.action == "left_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] += 0.2
                    if evt.action == "right_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] -= 0.2
                    if evt.action == "next_motion" and evt.value > 0:
                        self._motion_ids[self.lookat_id] = (self._motion_ids[self.lookat_id] + 1) % self._motion_lib.num_motions()
                        self.update_motion_ids([self.lookat_id])
                    if evt.action == "prev_motion" and evt.value > 0:
                        self._motion_ids[self.lookat_id] = (self._motion_ids[self.lookat_id] - 1) % self._motion_lib.num_motions()
                        self.update_motion_ids([self.lookat_id])
                    if evt.action == "reset_motion" and evt.value > 0:
                        self._motion_times[self.lookat_id] = 0
                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
                
                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    while self.pause:
                        time.sleep(0.1)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        for evt in self.gym.query_viewer_action_events(self.viewer):
                            if evt.action == "pause" and evt.value > 0:
                                self.pause = False
                        if self.gym.query_viewer_has_closed(self.viewer):
                            sys.exit()
                if evt.value > 0:
                    evt_count += 1
            self.button_pressed = True if evt_count > 0 else False

                        
                
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.poll_viewer_events(self.viewer)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
            
            if not self.free_cam:
                p = self.gym.get_viewer_camera_transform(self.viewer, None).p
                cam_trans = torch.tensor([p.x, p.y, p.z], requires_grad=False, device=self.device)
                look_at_pos = self.root_states[self.lookat_id, :3].clone()
                self.lookat_vec = cam_trans - look_at_pos
            

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_key_body_pos, dof_offsets):
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    local_root_vel = quat_rotate_inverse(root_rot, root_vel)
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)

@torch.jit.script
def reindex_motion_dof(dof, indices_sim, indices_motion, valid_dof_body_ids):
    dof = dof.clone()
    dof[:, indices_sim] = dof[:, indices_motion]
    return dof[:, valid_dof_body_ids]

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]
    return global_body_pos

@torch.jit.script
def global_to_local(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = (rigid_body_pos - root_pos[:, None, :3]).view(total_bodies, 3)
    local_end_pos = quat_rotate_inverse(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3)
    return local_end_pos

@torch.jit.script
def global_to_local_xy(yaw, global_pos_delta):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rotation_matrices = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=2).view(-1, 2, 2)
    local_pos_delta = torch.bmm(rotation_matrices, global_pos_delta.unsqueeze(-1))
    return local_pos_delta.squeeze(-1)




