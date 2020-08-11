import argparse
from collections import OrderedDict
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import ConfigDict
class DDPGLearner():
    def __init__(self,args: argparse.Namespace,env_info: ConfigDict,hyper_params: ConfigDict,log_cfg: ConfigDict,head: ConfigDict,optim_cfg: ConfigDict,noise_cfg: ConfigDict,device: torch.device):
        self.args = args
        if not self.args.test:
            self.ckpt_path = (f"./checkpoint/{env_info.name}/{log_cfg.agent}/{log_cfg.curr_time}/")
            os.makedirs(self.ckpt_path, exist_ok=True)
            shutil.copy(self.args.cfg_path, os.path.join(self.ckpt_path, "config.py"))
        self.env_info = env_info
        self.hyper_params = hyper_params
        self.log_cfg = log_cfg
        self.device = device
        self.head_cfg = head
        self.head_cfg.critic.configs.state_size = (self.env_info.observation_space.shape[0]+ self.env_info.action_space.shape[0],)
        self.head_cfg.actor.configs.state_size = self.env_info.observation_space.shape
        self.head_cfg.actor.configs.output_size = self.env_info.action_space.shape[0]
        self.optim_cfg = optim_cfg
        self.noise_cfg = noise_cfg
        self._init_network()

    def _init_network(self):
        """Initialize networks and optimizers."""
        # create actor
        self.actor = Brain(self.head_cfg.actor).to(self.device)
        self.actor_target = Brain(self.head_cfg.actor).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # create critic
        self.critic = Brain(self.head_cfg.critic).to(self.device)
        self.critic_target = Brain(self.head_cfg.critic).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizer
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=self.optim_cfg.lr_critic,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """Update actor and critic networks."""
        states, actions, rewards, next_states, dones = experience

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + self.hyper_params.gamma * next_values * masks
        curr_returns = curr_returns.to(self.device)

        # train critic
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        # train actor
        actions = self.actor(states)
        actor_loss = -self.critic(torch.cat((states, actions), dim=-1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        # update target networks
        common_utils.soft_update(self.actor, self.actor_target, self.hyper_params.tau)
        common_utils.soft_update(self.critic, self.critic_target, self.hyper_params.tau)

        return actor_loss.item(), critic_loss.item()

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        print(f"[INFO] Saved the model and optimizer to {path} \n")
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
        }
        os.makedirs(self.ckpt_path, exist_ok=True)
        path = os.path.join(self.ckpt_path+ "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def get_state_dict(self) -> Tuple[OrderedDict]:
        """Return state dicts, mainly for distributed worker."""
        return (self.critic_target.state_dict(), self.actor.state_dict())

    def get_policy(self) -> nn.Module:
        """Return model (policy) used for action selection."""
        return self.actor
