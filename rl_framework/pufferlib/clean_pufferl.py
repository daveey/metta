import os
import random
import time
from collections import defaultdict

import imageio
import numpy as np
import pufferlib
import pufferlib.pytorch
import pufferlib.utils
import torch
from omegaconf import OmegaConf

from . import checkpoint, dashboard
from .experience import Experience
from .profile import Profile
from .utilization import Utilization

torch.set_float32_matmul_precision('high')

# Fast Cython GAE implementation
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from .c_gae import compute_gae


def create(config, vecenv, policy, optimizer=None, wandb=None):
    seed_everything(config.seed, config.torch_deterministic)
    profile = Profile()
    losses = make_losses()

    utilization = Utilization()
    msg = f'Model Size: {dashboard.abbreviate(count_params(policy))} parameters'
    dashboard.print(config, utilization, 0, 0, profile, losses, {}, msg, clear=True)

    vecenv.async_reset(config.seed)
    obs_shape = vecenv.single_observation_space.shape
    obs_dtype = vecenv.single_observation_space.dtype
    atn_shape = vecenv.single_action_space.shape
    total_agents = vecenv.num_agents

    lstm = policy.lstm if hasattr(policy, 'lstm') else None
    experience = Experience(config.batch_size, config.bptt_horizon,
        config.minibatch_size, obs_shape, obs_dtype, atn_shape, config.cpu_offload, config.device, lstm, total_agents)

    uncompiled_policy = policy

    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode)

    optimizer = torch.optim.Adam(policy.parameters(),
        lr=config.learning_rate, eps=1e-5)

    return pufferlib.namespace(
        config=config,
        vecenv=vecenv,
        policy=policy,
        uncompiled_policy=uncompiled_policy,
        optimizer=optimizer,
        experience=experience,
        profile=profile,
        losses=losses,
        wandb=wandb,
        global_step=0,
        epoch=0,
        stats={},
        msg=msg,
        last_log_time=0,
        utilization=utilization,
    )

@pufferlib.utils.profile
def evaluate(data):
    config, profile, experience = data.config, data.profile, data.experience

    with profile.eval_misc:
        policy = data.policy
        infos = defaultdict(list)
        lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

    while not experience.full:
        with profile.env:
            o, r, d, t, info, env_id, mask = data.vecenv.recv()
            env_id = env_id.tolist()

        with profile.eval_misc:
            data.global_step += sum(mask)

            o = torch.as_tensor(o)
            o_device = o.to(config.device)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)

        with profile.eval_forward, torch.no_grad():
            # TODO: In place-update should be faster. Leaking 7% speed max
            # Also should be using a cuda tensor to index
            if lstm_h is not None:
                h = lstm_h[:, env_id]
                c = lstm_c[:, env_id]
                actions, logprob, _, value, (h, c) = policy(o_device, (h, c))
                lstm_h[:, env_id] = h
                lstm_c[:, env_id] = c
            else:
                actions, logprob, _, value = policy(o_device)

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.eval_misc:
            value = value.flatten()
            actions = actions.cpu().numpy()
            mask = torch.as_tensor(mask)# * policy.mask)
            o = o if config.cpu_offload else o_device
            experience.store(o, value, actions, logprob, r, d, env_id, mask)

            for i in info:
                for k, v in pufferlib.utils.unroll_nested_dict(i):
                    infos[k].append(v)

        with profile.env:
            data.vecenv.send(actions)

    with profile.eval_misc:
        data.stats = {}

        for k, v in infos.items():
            if '_map' in k and data.wandb is not None:
                data.stats[f'Media/{k}'] = data.wandb.Image(v[0])
                continue

            try: # TODO: Better checks on log data types
                data.stats[k] = np.mean(v)
            except:
                continue

    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    data.losses = make_losses()
    losses = data.losses

    with profile.train_misc:
        idxs = experience.sort_training_data()
        dones_np = experience.dones_np[idxs]
        values_np = experience.values_np[idxs]
        rewards_np = experience.rewards_np[idxs]
        # TODO: bootstrap between segment bounds
        advantages_np = compute_gae(dones_np, values_np,
            rewards_np, config.gamma, config.gae_lambda)
        experience.flatten_batch(advantages_np)

    # Optimizing the policy and value network
    mean_pg_loss, mean_v_loss, mean_entropy_loss = 0, 0, 0
    mean_old_kl, mean_kl, mean_clipfrac = 0, 0, 0
    for epoch in range(config.update_epochs):
        lstm_state = None
        for mb in range(experience.num_minibatches):
            with profile.train_misc:
                obs = experience.b_obs[mb]
                obs = obs.to(config.device)
                atn = experience.b_actions[mb]
                log_probs = experience.b_logprobs[mb]
                val = experience.b_values[mb]
                adv = experience.b_advantages[mb]
                ret = experience.b_returns[mb]

            with profile.train_forward:
                if experience.lstm_h is not None:
                    _, newlogprob, entropy, newvalue, lstm_state = data.policy(
                        obs, state=lstm_state, action=atn)
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, newvalue = data.policy(
                        obs.reshape(-1, *data.vecenv.single_observation_space.shape),
                        action=atn,
                    )

                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                logratio = newlogprob - log_probs.reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

                adv = adv.reshape(-1)
                if config.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - ret) ** 2
                    v_clipped = val + torch.clamp(
                        newvalue - val,
                        -config.vf_clip_coef,
                        config.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

            with profile.learn:
                data.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(data.policy.parameters(), config.max_grad_norm)
                data.optimizer.step()
                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                losses.policy_loss += pg_loss.item() / experience.num_minibatches
                losses.value_loss += v_loss.item() / experience.num_minibatches
                losses.entropy += entropy_loss.item() / experience.num_minibatches
                losses.old_approx_kl += old_approx_kl.item() / experience.num_minibatches
                losses.approx_kl += approx_kl.item() / experience.num_minibatches
                losses.clipfrac += clipfrac.item() / experience.num_minibatches

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    with profile.train_misc:
        if config.anneal_lr:
            frac = 1.0 - data.global_step / config.total_timesteps
            lrnow = frac * config.learning_rate
            data.optimizer.param_groups[0]["lr"] = lrnow

        y_pred = experience.values_np
        y_true = experience.returns_np
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        losses.explained_variance = explained_var
        data.epoch += 1

        done_training = data.global_step >= config.total_timesteps
        if profile.update(data) or (
                'episode_return' in data.stats or done_training):
            dashboard.print(config, data.utilization, data.global_step, data.epoch,
                profile, data.losses, data.stats, data.msg)

            if data.wandb is not None and data.global_step > 0 and time.time() - data.last_log_time > 3.0:
                data.last_log_time = time.time()
                data.wandb.log({
                    '0verview/SPS': profile.SPS,
                    '0verview/agent_steps': data.global_step,
                    '0verview/epoch': data.epoch,
                    '0verview/learning_rate': data.optimizer.param_groups[0]["lr"],
                    'train/actual_lr': data.optimizer.param_groups[0]["lr"],
                    'train/lr': data.optimizer.param_groups[0]["lr"],
                    'perf/_sample_throughput': profile.SPS,
                    'len/len': data.stats.get('episode_length', 0),
                    'policy_stats/avg_true_objective': data.stats.get('episode_return', 0),
                    **{f'policy_stats/{k}': v for k, v in data.stats.items()},
                    **{f'losses/{k}': v for k, v in data.losses.items()},
                    'train/policy_loss': data.losses.policy_loss,
                    'train/value_loss': data.losses.value_loss,
                    'train/entropy_loss': data.losses.entropy,
                    **{f'performance/{k}': v for k, v in data.profile},
                    "global_step": data.global_step,
                })

        if data.epoch % config.checkpoint_interval == 0 or done_training:
            checkpoint.save_checkpoint(data)
            data.msg = f'Checkpoint saved at update {data.epoch}'

def close(data):
    data.vecenv.close()
    data.utilization.stop()
    config = data.config
    if data.wandb is not None:
        artifact_name = f"{config.exp_id}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = checkpoint.save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()



def make_losses():
    return pufferlib.namespace(
        policy_loss=0,
        value_loss=0,
        entropy=0,
        old_approx_kl=0,
        approx_kl=0,
        clipfrac=0,
        explained_variance=0,
    )

def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)

def rollout(cfg: OmegaConf, env_creator, env_kwargs, agent_creator, agent_kwargs,
        model_path=None, render_mode='auto', device='cuda', verbose=True):
    # We are just using Serial vecenv to give a consistent
    # single-agent/multi-agent API for evaluation
    if render_mode != 'auto':
        env_kwargs["render_mode"] = render_mode

    env = pufferlib.vector.make(env_creator, env_kwargs=env_kwargs)

    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    ob, info = env.reset()
    driver = env.driver_env
    os.system('clear')
    state = None

    frames = []
    tick = 0
    while tick <= cfg.eval.max_steps:
        if tick % 1 == 0:
            render = driver.render()
            if driver.render_mode == 'ansi':
                print('\033[0;0H' + render + '\n')
                time.sleep(0.6)
            elif driver.render_mode == 'rgb_array':
                frames.append(render)
                import cv2
                render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', render)
                cv2.waitKey(1)
                #time.sleep(1/24)
            elif driver.render_mode == 'human' and render is not None:
                frames.append(render)

        with torch.no_grad():
            ob = torch.from_numpy(ob).to(device)
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent(ob, state)
            else:
                action, _, _, _ = agent(ob)

            action = action.cpu().numpy().reshape(env.action_space.shape)

        ob, reward = env.step(action)[:2]
        reward = reward.mean()
        if tick % 100 == 0:
            print(f'Reward: {reward:.4f}, Tick: {tick}')
        tick += 1

    return {'reward': reward, 'frames': frames}

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
