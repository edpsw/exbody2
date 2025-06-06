# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1MimicPrivCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096

        n_demo_steps = 2
        n_demo = 19 + 3 + 3 + 3 + 12*3  #observe height
        interval_demo_steps = 0.1

        n_scan = 0#132
        n_priv = 3
        n_priv_latent = 4 + 1 + 23*2
        n_proprio = 3 + 2 + 2 + 23*3 + 2 # one hot
        history_len = 10

        extra_history_len = 25

        prop_hist_len = 4
        n_feature = prop_hist_len * n_proprio

        n_teacher_priv = 75

        num_observations = n_feature + n_proprio + n_teacher_priv + n_demo + history_len*n_proprio + n_priv_latent + n_priv

        episode_length_s = 50 # episode length in seconds
        
        num_actions = 23
        
        num_policy_actions = 23
    
    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        motion_name = "motions_autogen_all_no_run_jump.yaml"

        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        no_keybody = False
        regen_pkl = False

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10


    class terrain( LeggedRobotCfg.terrain ):
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        height = [0., 0.04]
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_pitch_joint' : -0.4,    
           'left_ankle_roll_joint' : 0, 
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_pitch_joint' : -0.4,
           'right_ankle_roll_joint' : 0,                                     
           'waist_yaw_joint' : 0., 
           'waist_pitch_joint' : 0.,
           'waist_roll_joint' : 0.,
           'left_shoulder_pitch_joint' : 0.5, 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.2,
           'left_elbow_joint'  : 0.3,
           'right_shoulder_pitch_joint' : 0.5,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : -0.2,
           'right_elbow_joint' : 0.3,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 150,
                     'hip_pitch': 150,
                     'hip_roll': 150,
                     'knee': 300,
                     'ankle_pitch': 40,
                     'ankle_roll': 20,
                     'waist_yaw': 400,
                     'waist_pitch': 400,
                     'waist_roll': 400,
                     'shoulder_pitch': 60,
                     'shoulder_roll': 60,
                     'shoulder_yaw': 30,
                     "elbow": 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_pitch': 2,
                     'hip_roll': 2,
                     'knee': 4,
                     'ankle_pitch': 2,
                     'ankle_roll': 0.15,
                     'waist_yaw': 15,
                     'waist_pitch': 15,
                     'waist_roll': 15,
                     'shoulder_pitch': 2,
                     'shoulder_roll': 2,
                     'shoulder_yaw': 1,
                     "elbow": 1,
                     }  # [N*m/rad]  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 10 # 4

    class normalization( LeggedRobotCfg.normalization):
        clip_actions = 10

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_loco.urdf'
        torso_name = "waist_yaw_link"
        foot_name = "ankle_roll"
        hip_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
        penalize_contacts_on = ["shoulder", "elbow", "hip"]
        terminate_after_contacts_on = ["waist_yaw_link", "torso_link", "hip"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        armature = 5e-5  # stablize semi-euler integration for end effectors
    
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            alive = 3
            tracking_lin_vel = 10

            tracking_demo_yaw = 1
            tracking_demo_roll_pitch = 1
            orientation = -4
            tracking_demo_dof_pos = 6
            tracking_demo_key_body = 10

            dof_acc = -3e-7
            action_rate = -0.1
            dof_error = -0.1
            feet_stumble = -2
            dof_pos_limits = -10.0
            feet_air_time = 10
            feet_force = -3e-3
            ankle_action = -0.1
            waist_roll_pitch_error = -0.5

        only_positive_rewards = False
        clip_rewards = True
        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]
    
    class noise():
        add_noise = True
        noise_scale = 0.5 # scales other values
        class noise_scales():
            dof_pos = 0.01
            dof_vel = 0.15
            ang_vel = 0.3
            imu = 0.2
            
    class g1_params:
        height_factor = 1.3/1.8
        # height_factor = 1.0
        max_init_height = 0.6
        min_init_height = 0.5
        max_vel = 5.0

class G1MimicPrivCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimicPriv"
        policy_class_name = 'ActorCriticMimicPriv'
        algorithm_class_name = 'PPOMimicPriv'
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False
        text_feat_input_dim = G1MimicPrivCfg.env.n_feature
        text_feat_output_dim = 16
        feat_hist_len = G1MimicPrivCfg.env.prop_hist_len
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005
        grad_penalty_coef_schedule = [0.0001, 0.0001, 700, 1000]

    class estimator:
        train_with_estimated_states = False
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = G1MimicPrivCfg.env.n_priv
        priv_start = G1MimicPrivCfg.env.n_feature + G1MimicPrivCfg.env.n_proprio + G1MimicPrivCfg.env.n_teacher_priv + G1MimicPrivCfg.env.n_demo + G1MimicPrivCfg.env.n_scan
        
        prop_start = G1MimicPrivCfg.env.n_feature
        prop_dim = G1MimicPrivCfg.env.n_proprio

class G1MimicPrivDistillCfgPPO( G1MimicPrivCfgPPO ):
    class distill:
        num_demo = G1MimicPrivCfg.env.n_demo
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]

        num_mini_batches = 4

        num_student_history = G1MimicPrivCfg.env.extra_history_len