import os
import math
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import zarr
import copy
import stable_baselines3
from stable_baselines3 import PPO
import cv2
import random
from tqdm.auto import tqdm
from typing import Union
from diffusers import DDPMScheduler
from transformers import get_scheduler


class EMAModel:
    def __init__(self, parameters, power=0.75):
        self.power = power
        self.shadow = [p.detach().clone() for p in parameters]

    def step(self, parameters):
        # Update shadow parameters at each step
        for s, p in zip(self.shadow, parameters):
            s.data = s.data * self.power + (1.0 - self.power) * p.data

    def copy_to(self, parameters):
        # Copy the shadow parameter to the real model parameter for eval
        for s, p in zip(self.shadow, parameters):
            p.data = s.data.clone()

#-----------------------------------------------
# Generating datasets in the CarRacing environment using stable-baselines3 PPO-trained models
#-----------------------------------------------
model_path = "ppo_carracing_model.zip"
assert os.path.exists(model_path), "Make sure you have placed the PPO model file (ppo_carracing_model.zip) in the current directory"

env = gym.make('CarRacing-v2')
model = PPO.load(model_path, env=env)

obs_dim = env.observation_space.shape  # CarRacing obs: (96,96,3)
action_dim = env.action_space.shape[0] # CarRacing action: 3维连续动作[steering, gas, brake]

num_episodes = 10
max_steps_per_episode = 1000

all_obs = []
all_actions = []
episode_ends = []
step_count = 0

for eps_id in range(num_episodes):
    obs, _ = env.reset(return_info=True)
    done = False
    step_idx = 0
    while not done and step_idx < max_steps_per_episode:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, _, info = env.step(action)
        all_obs.append(obs)
        all_actions.append(action)
        obs = next_obs
        step_idx += 1
        step_count += 1
    episode_ends.append(step_count)

all_obs = np.array(all_obs)       # shape: [N,96,96,3]
all_actions = np.array(all_actions) # shape: [N,3]


dataset_root = {
    'data': {
        'state': all_obs,    # (N,96,96,3)
        'action': all_actions # (N,3)
    },
    'meta': {
        'episode_ends': np.array(episode_ends)
    }
}

#-----------------------------------------------
# Parameter
#-----------------------------------------------
pred_horizon = 16    # Predicting the Future in 16 Steps
obs_horizon = 2       # Conditional observation length
action_horizon = 8    # Length of implementation of actions

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    # For action data as (N, action_dim)
    # For obs data (after encoding) is (N, obs_dim)
    data_reshaped = data.reshape(-1, data.shape[-1]) if data.ndim > 2 else data
    stats = {
        'min': np.min(data_reshaped, axis=0),
        'max': np.max(data_reshaped, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [-1,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min'] + 1e-8) + stats['min']
    return data

class CarRacingDataset(Dataset):
    def __init__(self, dataset_root,
                 pred_horizon, obs_horizon, action_horizon):
        train_data = {
            'action': dataset_root['data']['action'],
            'obs': self.encode_obs(dataset_root['data']['state']) # 转换obs到低维
        }
        episode_ends = dataset_root['meta']['episode_ends']

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        self.stats = {}
        self.normalized_train_data = {}
        for key, data in train_data.items():
            self.stats[key] = get_data_stats(data)
            self.normalized_train_data[key] = normalize_data(data, self.stats[key])

        self.indices = indices
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def encode_obs(self, obs_array):
        # 将96x96x3 -> 降采样 -> 投影到低维（例如64维）
        N = obs_array.shape[0]
        obs_small = obs_array[:, ::8, ::8, :].reshape(N, -1).astype(np.float32) # (N,12*12*3=432)
        np.random.seed(42)
        W = np.random.randn(obs_small.shape[1], 64).astype(np.float32)
        obs_feat = obs_small @ W # (N,64)
        return obs_feat

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # 只取前obs_horizon步的obs作为条件
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample


dataset = CarRacingDataset(
    dataset_root=dataset_root,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)

stats = dataset.stats

dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
    persistent_workers=False
)

batch = next(iter(dataloader))
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape:", batch['action'].shape)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels)
        scale = embed[:,0,...].unsqueeze(-1)
        bias = embed[:,1,...].unsqueeze(-1)
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        sample = sample.moveaxis(-1,-2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for (resnet, resnet2, downsample) in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for (resnet, resnet2, upsample) in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1,-2)
        return x

obs_dim = 64  
action_dim = 3

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_pred_net.to(device)

# 定义扩散调度器
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6
)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader)*100
)

num_epochs = 100
for epoch_idx in range(num_epochs):
    epoch_loss = []
    for nbatch in tqdm(dataloader, desc=f'Epoch {epoch_idx}', leave=False):
        nobs = nbatch['obs'].to(device, dtype=torch.float32)   # (B,obs_horizon,obs_dim)
        naction = nbatch['action'].to(device, dtype=torch.float32) # (B,pred_horizon,action_dim)
        B = nobs.shape[0]

        obs_cond = nobs.reshape(B, -1)
        noise = torch.randn_like(naction, device=device)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        noisy_actions = noise_scheduler.add_noise(
            original_samples=naction, noise=noise, timesteps=timesteps)

        noise_pred = noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond)

        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(noise_pred_net.parameters())

        epoch_loss.append(loss.item())
    print(f"Epoch {epoch_idx}, Loss: {np.mean(epoch_loss)}")


ema.copy_to(noise_pred_net.parameters())

max_steps = 200
env.seed(12345)
obs, _ = env.reset(return_info=True)

obs_deque = collections.deque([obs]*obs_horizon, maxlen=obs_horizon)
rewards = []
done = False
step_idx = 0

noise_pred_net.eval()
with torch.no_grad():
    while not done and step_idx < max_steps:
        obs_seq = np.stack(obs_deque)  # (obs_horizon,96,96,3)

        obs_small = obs_seq[:, ::8, ::8, :].reshape(obs_horizon, -1).astype(np.float32)
        np.random.seed(42)
        W = np.random.randn(obs_small.shape[1], 64).astype(np.float32)
        obs_feat = obs_small @ W # (obs_horizon,64)

        nobs = normalize_data(obs_feat, stats['obs'])
        nobs = torch.from_numpy(nobs).unsqueeze(0).to(device)

        obs_cond = nobs.reshape(1,-1)
        noisy_action = torch.randn((1, pred_horizon, action_dim), device=device)

        noise_scheduler.set_timesteps(num_diffusion_iters)
        naction = noisy_action
        for k in noise_scheduler.timesteps:
            noise_pred = noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        naction = naction.cpu().numpy()[0]
        action_pred = unnormalize_data(naction, stats['action'])

        # 取action_horizon步动作执行
        start = obs_horizon - 1
        end = start + action_horizon
        action_seq = action_pred[start:end,:]

        for a in action_seq:
            obs, reward, done, _, info = env.step(a)
            obs_deque.append(obs)
            rewards.append(reward)
            step_idx += 1
            if done or step_idx >= max_steps:
                break

print("Max reward encountered: ", max(rewards) if len(rewards)>0 else 0)
print("Rollout finished.")

