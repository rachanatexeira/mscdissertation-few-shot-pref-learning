import itertools
from typing import Any, Dict, Type, Union

import gym
import numpy as np
import torch

from research.networks.base import ActorCriticPolicy

from .base import Algorithm


class SAC(Algorithm):
    def __init__(
        self,
        env: gym.Env,
        network_class: Type[torch.nn.Module],
        dataset_class: Union[Type[torch.utils.data.IterableDataset], Type[torch.utils.data.Dataset]],
        tau: float = 0.005,
        init_temperature: float = 0.1,
        env_freq: int = 1,
        critic_freq: int = 1,
        actor_freq: int = 1,
        target_freq: int = 2,
        init_steps: int = 1000,
        **kwargs,
    ):
        # Save values needed for optim setup.
        self.init_temperature = init_temperature
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)

        # Save extra parameters
        self.tau = tau
        self.env_freq = env_freq
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.init_steps = init_steps
        self.action_range = [float(self.action_space.low.min()), float(self.action_space.high.max())]

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self, optim_class: Type[torch.optim.Optimizer], optim_kwargs: Dict) -> None:
        # Default optimizer initialization
        self.optim["actor"] = optim_class(self.network.actor.parameters(), **optim_kwargs)
        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        self.optim["critic"] = optim_class(critic_params, **optim_kwargs)

        # Setup the learned entropy coefficients. This has to be done first so its present in the setup_optim call.
        self.log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(self.action_space.low.shape)

        self.optim["log_alpha"] = optim_class([self.log_alpha], **optim_kwargs)

    def _update_critic(self, batch: Dict) -> Dict:
        with torch.no_grad():
            dist = self.network.actor(batch["next_obs"])
            next_action = dist.rsample()

            # changes
            # only clamp for log_prob to keep it strictly inside (-1, 1)
            eps = 1e-6
            safe_next_action = next_action.clamp(-1.0 + eps, 1.0 - eps)



            # log_prob = dist.log_prob(action).sum(dim=-1)
            # log_prob = dist.log_prob(safe_action).sum(dim=-1)
            next_log_prob = dist.log_prob(safe_next_action).sum(dim=-1)

            # qs = self.network.critic(obs, action)
            # qs = self.network.critic(obs, safe_action)
            
            # log_prob = dist.log_prob(next_action).sum(dim=-1)
            # target_qs = self.target_network.critic(batch["next_obs"], next_action)
            

            # target_qs = self.target_network.critic(batch["next_obs"], safe_action)

            target_qs = self.target_network.critic(batch["next_obs"], safe_next_action)




            # target_v = torch.min(target_qs, dim=0)[0] - self.alpha.detach() * log_prob
            

            target_v = torch.min(target_qs, dim=0)[0] - self.alpha.detach() * next_log_prob
            target_q = batch["reward"] + batch["discount"] * target_v

        qs = self.network.critic(batch["obs"], batch["action"])
        q_loss = (
            torch.nn.functional.mse_loss(qs, target_q.expand(qs.shape[0], -1), reduction="none").mean(dim=-1).sum()
        )  # averages over the ensemble. No for loop!

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optim["critic"].step()

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())

    def _update_actor_and_alpha(self, batch: Dict) -> Dict:
        obs = batch["obs"].detach()  # Detach the encoder so it isn't updated.
        dist = self.network.actor(obs)
        action = dist.rsample()





        # changes
        # only clamp for log_prob to keep it strictly inside (-1, 1)
        eps = 1e-6
        safe_action = action.clamp(-1.0 + eps, 1.0 - eps)


        # log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob = dist.log_prob(safe_action).sum(dim=-1)

        # qs = self.network.critic(obs, action)
        qs = self.network.critic(obs, safe_action)




        q = torch.min(qs, dim=0)[0]
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()
        entropy = -log_prob.mean()

        # Update the learned temperature
        self.optim["log_alpha"].zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optim["log_alpha"].step()

        return dict(
            actor_loss=actor_loss.item(),
            entropy=entropy.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.detach().item(),
        )

    def _step_env(self) -> Dict:
        # Step the environment and store the transition data.
        metrics = dict()
        if self._env_steps < self.init_steps:
            action = self.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                action = self.predict(dict(obs=self._current_obs), sample=True)
            self.train_mode()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # next_obs, reward, done, info = self.env.step(action)

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated


        if hasattr(self.env, "render"):
            self.env.render()
        
        self._episode_length += 1
        self._episode_reward += reward

        if "discount" in info:
            discount = info["discount"]
        elif hasattr(self.env, "_max_episode_steps") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences.
        self.dataset.add(next_obs, action, reward, done, discount)

        if done:
            self._num_ep += 1
            # update metrics
            metrics["reward"] = self._episode_reward
            metrics["length"] = self._episode_length
            metrics["num_ep"] = self._num_ep

            # ✅ Print accumulated episode reward
            print(f"[INFO] Episode {self._num_ep} finished | Length: {self._episode_length}, Total Reward: {self._episode_reward}")
            # Reset the environment
            self._current_obs = self.env.reset()
            self.dataset.add(self._current_obs)  # Add the first timestep

            self.dataset.add(self._current_obs)
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs

        self._env_steps += 1
        metrics["env_steps"] = self._env_steps



        return metrics

    def _setup_train(self) -> None:

        reset_out = self.env.reset()
        self._current_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.dataset.add(self._current_obs)   # <-- add initial obs


        # self._current_obs = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0
        self._env_steps = 0

    def _train_step(self, batch: Dict) -> Dict:
        all_metrics = {}

        if self.steps % self.env_freq == 0 or self._env_steps < self.init_steps:
            # step the environment with freq env_freq or if we are before learning starts
            metrics = self._step_env()
            all_metrics.update(metrics)
            if self._env_steps < self.init_steps:
                return all_metrics  # return here.

        if "obs" not in batch:
            return all_metrics

        updating_critic = self.steps % self.critic_freq == 0
        updating_actor = self.steps % self.actor_freq == 0

        if updating_actor or updating_critic:
            batch["obs"] = self.network.encoder(batch["obs"])
            with torch.no_grad():
                batch["next_obs"] = self.target_network.encoder(batch["next_obs"])

        if updating_critic:
            metrics = self._update_critic(batch)
            all_metrics.update(metrics)

        if updating_actor:
            metrics = self._update_actor_and_alpha(batch)
            all_metrics.update(metrics)

        if self.steps % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(
                    self.network.encoder.parameters(), self.target_network.encoder.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    # def _predict(self, batch: Any, sample: bool = False) -> torch.Tensor:
    #     obs = batch["obs"]

    #     # Convert list to numpy
    #     if isinstance(obs, list):
    #         obs = np.array(obs, dtype=np.float32)

    #     # Convert numpy to torch tensor
    #     if isinstance(obs, np.ndarray):
    #         obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

    #      # Ensure tensor is on correct device and dtype
    #     if isinstance(obs, torch.Tensor):
    #         obs = obs.to(self.device).float()  


    #     with torch.no_grad():
    #         # z = self.network.encoder(batch["obs"])
    #         z = self.network.encoder(obs)
    #         dist = self.network.actor(z)
    #         if sample:
    #             action = dist.sample()
    #         else:
    #             action = dist.loc
    #         action = action.clamp(*self.action_range)
    #         # return action
    #         return action.detach().cpu().numpy()



    
    # 04 Aug
    # def _predict(self, batch: Any, sample: bool = False) -> np.ndarray:
    #     obs = batch["obs"]

    #     if isinstance(obs, list):
    #         obs_tensors = [o for o in obs if isinstance(o, torch.Tensor)]
    #         if len(obs_tensors) == 0:
    #             raise ValueError("No valid tensor found in obs list!")
    #         obs = obs_tensors[0]  # use the real observation tensor
    #         obs = obs.detach().cpu().numpy().astype(np.float32)

    #     if isinstance(obs, np.ndarray):
    #         obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

    #     if isinstance(obs, torch.Tensor):
    #         obs = obs.to(self.device).float()
    #         obs = torch.clamp(obs, -10.0, 10.0) 

    #     with torch.no_grad():
    #         z = self.network.encoder(obs)


    #         # Access actor's MLP directly to see the raw outputs
    #         mu, log_std = self.network.actor.mlp(z).chunk(2, dim=-1)

    #         # Debugging NaNs
    #         if torch.isnan(mu).any():
    #             print(f"[DEBUG] NaN in mu at step {getattr(self, '_steps', 'unknown')}")
    #         if torch.isnan(log_std).any():
    #             print(f"[DEBUG] NaN in log_std at step {getattr(self, '_steps', 'unknown')}")

    #         dist = self.network.actor(z)
    #         action = dist.sample() if sample else dist.loc

    #         action = action.clamp(*self.action_range)
    #         action_np = action.detach().cpu()
    #         return action_np


    def _predict(self, batch: Any, sample: bool = False) -> torch.Tensor:
        obs = batch["obs"]

        # Handle lists
        if isinstance(obs, list):
            obs_tensors = [o for o in obs if isinstance(o, torch.Tensor)]
            if not obs_tensors:
                raise ValueError("No valid tensor found in obs list!")
            obs = obs_tensors[0]


        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        elif isinstance(obs, torch.Tensor):
            obs = obs.to(self.device).float()
        else:
            raise ValueError(f"Unsupported obs type: {type(obs)}")

        with torch.no_grad():
            z = self.network.encoder(obs)
            
            # 🔹 Inspect mu and log_std for NaN debugging (but don't break shapes)
            dist = self.network.actor(z)

            # Extract mu and log_std for debugging (safe)
            if hasattr(self.network.actor, "mlp"):
                mu, log_std = self.network.actor.mlp(z).chunk(2, dim=-1)
                if torch.isnan(mu).any():
                    print(f"[DEBUG] NaN in mu at step {getattr(self, '_steps', 'unknown')}")
                if torch.isnan(log_std).any():
                    print(f"[DEBUG] NaN in log_std at step {getattr(self, '_steps', 'unknown')}")

            # 🔹 Sample or take mean
            action = dist.sample() if sample else dist.loc

            # # 🔹 Clamp to action range
            # action = action.clamp(*self.action_range)

            # 🔹 Return as numpy
            action_np = action.detach().cpu().numpy().astype(np.float32)
            return action_np


    def _validation_step(self, batch: Any):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")

    def _save_extras(self) -> Dict:
        return {"log_alpha": self.log_alpha}

    def _load_extras(self, checkpoint) -> Dict:
        self.log_alpha.data = checkpoint["log_alpha"].data
