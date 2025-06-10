# ExBody2

## Environment Setup

```bash
conda create -n humanoid python=3.8
conda activate humanoid
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone git@github.com:jimazeyu/exbody2.git
cd exbody2
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd ~/exbody2/rsl_rl && pip install -e .
cd ~/exbody2/legged_gym && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown
```

## Dataset Preparation
I put the motions_amass_CMU.yaml file in ASE/ase/poselib/data/configs as an example. The .npy files can be gotten from method2. If you want to use other training motions, you should edit your own yaml configs. The dataset curation is really important, and the key is the sampling of data. We are trying to develop a good curation method, but sometimes manually curate your data is a better choice. You may skip the data preparation and test the code with the example motions in the motions_dance_release.pkl.

### Method 1: Using CMU FBX

1. Install FBX SDK (see: https://github.com/nv-tlabs/ASE/issues/61).
2. Follow [Expressive Humanoid](https://github.com/chengxuxin/expressive-humanoid) to retarget .fbx files to .npy motion format.

### Method 2: Using AMASS

1. Use [amass_g1_retargeting](https://github.com/jimazeyu/amass_g1_retargeting.git) to convert AMASS data into .npy motions.
2. Place the output .npy files into ASE/ase/poselib/data/g1_retarget_npy/.
3. Follow [Expressive Humanoid](https://github.com/chengxuxin/expressive-humanoid) to generate key points.

## Usage

### Train Teacher Model

```bash
python train.py --task g1_mimic_priv 000-00-some_description --motion_name motions_dance_release.yaml --device cuda:0 --entity WANDB_ENTITY
```

### Train Student Model

```bash
python train.py --task g1_mimic_priv_distill 000-01-student_description --motion_name motions_dance_release.yaml --device cuda:0 --entity WANDB_ENTITY --resume --resumeid 000-00
```

### Play Teacher Policy

```bash
python play_priv.py --task g1_mimic_priv 000-00 --motion_name motions_dance_release.yaml --device cuda:0
```

### Play Student Policy

```bash
python play_priv.py --task g1_mimic_priv_distill 000-01 --motion_name motions_dance_release.yaml --device cuda:0
```

`motions_dance_release.yaml` corresponds to `motions_dance_release.pkl`, which contains 8 example dance motions. These are preprocessed and included in the repo for direct training and testing. 
Warning: the example dancing motions are not easy, you may need really good g1 to try. I suggest you start with simple motions!!!

For more viewer operations, debugging options, and experiment configurations, please refer to the [Expressive Humanoid](https://github.com/chengxuxin/expressive-humanoid) instructions — most Isaac Gym controls are consistent.

## Release

- [ ] Release the example policy.
- [ ] Release the complete training pipeline.
- [ ] Release the deployment code.

## Acknowledgements

- Retargeting and motion processing code is adapted from [ASE](https://github.com/nv-tlabs/ASE) and [PHC](https://github.com/ZhengyiLuo/PHC).
- The codebase adapted from [Expressive Humanoid](https://github.com/chengxuxin/expressive-humanoid).
