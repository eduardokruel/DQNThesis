# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import importlib
import time
from dataclasses import dataclass

import gymnasium as gym
import pettingzoo 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import supersuit as ss
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "wizard_of_wor_v3"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    kill_reward: int = 0
    """the reward for killing another agent"""
    negative_reward: bool = False
    """whether to turn kill reward negative"""


# def make_env(env_id, seed, idx, capture_video, run_name):
#     if capture_video and idx == 0:
#         env = importlib.import_module(f"pettingzoo.atari.{env_id}").parallel_env(render_mode="rgb_array")
#         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#     else:
#         env = importlib.import_module(f"pettingzoo.atari.{env_id}").parallel_env()
#     env = gym.wrappers.RecordEpisodeStatistics(env)

#     env = NoopResetEnv(env, noop_max=30)
#     env = MaxAndSkipEnv(env, skip=4)
#     env = EpisodicLifeEnv(env)
#     if "FIRE" in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)
#     env = ClipRewardEnv(env)
#     env = gym.wrappers.ResizeObservation(env, (84, 84))
#     env = gym.wrappers.GrayScaleObservation(env)
#     env = gym.wrappers.FrameStack(env, 4)

#     env.action_space.seed(seed)
#     return env



# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    action_space0 = gym.spaces.Discrete(9,seed=args.seed)
    action_space1 = gym.spaces.Discrete(9,seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    # print(envs)
    # print(type(envs))
    # envs.single_observation_space = envs.observation_space
    # envs.single_action_space = envs.action_space
    # envs.is_vector_env = True
    # assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env(render_mode="rgb_array")
    env = ss.max_observation_v0(env, 2)
    # env = ss.frame_skip_v0(env, 4)
    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.black_death_v3(env)
    env = ss.frame_stack_v1(env, 4)
    # env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # envs = ss.concat_vec_envs_v1(
    #     env, args.num_envs, num_cpus=0, base_class="gymnasium"
    # )
    envs = env
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"


    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    
    start_time = time.time()
    episode_length = 0
    episode_reward = np.zeros(2)
    adjusted_episode_reward = np.zeros(2)
    episode_kills = np.zeros(2)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # print(np.moveaxis(obs[0:1],3,1).shape)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            actions = np.array([action_space0.sample(),action_space1.sample()])
        else:
            q_values = q_network(torch.Tensor(obs[0:1]).to(device).permute((0, 3, 1, 2)))
            actions = np.array([torch.argmax(q_values).cpu().numpy(),action_space1.sample()])
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # episode_reward = np.add(episode_reward,rewards)
        for i in range(2):
            if rewards[i] % 10 == 1:
                episode_kills[i] += 1
                rewards[i] -= 1
                episode_reward[i] += rewards[i]
                if args.negative_reward:
                    rewards[i] -= args.kill_reward
                else:
                    rewards[i] += args.kill_reward
                adjusted_episode_reward[i] += rewards[i]
            else:
                episode_reward[i] += rewards[i]
                adjusted_episode_reward[i] += rewards[i]
            if rewards[i] > 1:
                rewards[i] = 1
            if rewards[i] < -1:
                rewards[i] = -1
        # adjusted_episode_reward = np.add(episode_reward,rewards)
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if infos != [{},{}]:
            print(episode_reward)
            print(global_step - episode_length)
            print(f"global_step={global_step}, episodic_return={episode_reward}")
            writer.add_scalar("charts/episodic_return0", episode_reward[0], global_step)
            writer.add_scalar("charts/episodic_return1", episode_reward[1], global_step)
            writer.add_scalar("charts/adjusted_episodic_return0", adjusted_episode_reward[0], global_step)
            writer.add_scalar("charts/adjusted_episodic_return1", adjusted_episode_reward[1], global_step)
            writer.add_scalar("charts/episodic_kills0", episode_kills[0], global_step)
            writer.add_scalar("charts/episodic_kills1", episode_kills[1], global_step)
            writer.add_scalar("charts/episodic_length", global_step - episode_length, global_step)
            episode_length = global_step
            episode_reward = np.zeros(2)
            adjusted_episode_reward = np.zeros(2)
            episode_kills = np.zeros(2)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos[0]["terminal_observation"][idx]
        rb.add(obs[0:1], real_next_obs[0:1], actions[0], rewards[0], terminations[0], infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations.permute((0, 3, 1, 2))).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations.permute((0, 3, 1, 2))).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # from cleanrl_utils.evals.dqn_eval import evaluate

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=QNetwork,
        #     device=device,
        #     epsilon=0.05,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
            # writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
