import itertools

import numpy as np
import torch

from research.utils import utils

from .base import Algorithm


def collate(x):
    return x


class PreferenceMAML(Algorithm):
    def __init__(
        self,
        env,
        network_class,
        dataset_class,
        num_support=10,
        num_query=10,
        num_inner_steps=1,
        inner_lr=0.1,
        learn_inner_lr=True,
        **kwargs,
    ):
        # Save variables needed for optim init
        self.inner_lr = inner_lr
        self.learn_inner_lr = learn_inner_lr
        super().__init__(env, network_class, dataset_class, **kwargs)
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.num_support = num_support
        self.num_query = num_query
        self.num_inner_steps = num_inner_steps
        # Re-assign network to support training ActorCriticPolicy with reward networks.
        if hasattr(self.network, "reward"):
            # self.network.reward = self.network.reward
            # First, extract network_kwargs from the config directly if present
            network_kwargs = kwargs.pop("network_kwargs", {})

            # Instantiate the network properly
            self.network = network_class(
                env.observation_space,
                env.action_space,
                **network_kwargs
            )
            self.network.reward = self.network.reward
        else:
            self.network.reward = self.network
        # assert hasattr(self.network.reward, "params"), "Network class not setup for meta algs"
        
        

        if not hasattr(self.network.reward, "params"):
            print("[Warning] reward_network has no .params attribute — skipping reward meta-learning")
            self.network.reward = None
        # assert isinstance(self.network.reward.params, torch.nn.ParameterDict)
        if self.network.reward is not None:
            assert isinstance(self.network.reward.params, torch.nn.ParameterDict)

    def setup_optimizers(self, optim_class, optim_kwargs):
        # Handle case for ActorCriticPolicy.
        print("Reward head exists:", hasattr(self.network, "reward"))
        print("Networkd:", self.network)
        print("Reward network:", self.network.reward)
        if hasattr(self.network, "reward"):
            # network_params = self.network.reward.params
            if self.network.reward is None:
                network_params = list(self.network.actor.parameters()) + list(self.network.critic.parameters())
            else:
                # network_params = self.network.reward.params
                network_params = list(self.network.reward.params.values()) 
        else:
            network_params = list(self.network.parameters())
        # self._inner_lrs = torch.nn.ParameterDict(
        #     {
        #         k: torch.nn.Parameter(torch.tensor(self.inner_lr), requires_grad=self.learn_inner_lr)
        #         for k, v in network_params.items()
        #     }
        # )
        self._inner_lrs = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(self.inner_lr), requires_grad=self.learn_inner_lr)
            for _ in network_params
        ])
        # self.optim["reward"] = optim_class(
        #     # itertools.chain(network_params.values(), self._inner_lrs.values()), **optim_kwargs
        #     itertools.chain(network_params, self._inner_lrs.parameters())
        # )
        self.optim["reward"] = optim_class(
            itertools.chain(network_params, self._inner_lrs.parameters()),
            **optim_kwargs  
        )

    def _compute_loss_and_accuracy(self, batch, parameters):
        

        # Remove the singleton dimension: [32, 1, 25, 16] → [32, 25, 16]
        if batch["obs_1"].shape[1] == 1:
            batch["obs_1"] = batch["obs_1"].squeeze(1)
            batch["obs_2"] = batch["obs_2"].squeeze(1)
            batch["action_1"] = batch["action_1"].squeeze(1)
            batch["action_2"] = batch["action_2"].squeeze(1)

        B, S = batch["obs_1"].shape[:2]  # Get the batch size and the segment length
        assert B > 0 and S > 0, "Got Empty batch"
        flat_obs_shape = (B * S,) + batch["obs_1"].shape[2:]
        flat_action_shape = (B * S,) + batch["action_1"].shape[2:]

        # Get the device from model
        device = next(self.network.parameters()).device

        # Move all batch tensors to the same device
        obs_1 = batch["obs_1"].view(*flat_obs_shape).to(device)
        obs_2 = batch["obs_2"].view(*flat_obs_shape).to(device)
        action_1 = batch["action_1"].view(flat_action_shape).to(device)
        action_2 = batch["action_2"].view(flat_action_shape).to(device)
        labels = batch["label"].float().to(device)


        # Move each parameter tensor to the correct device
        for k, v in parameters.items():
            parameters[k] = v.to(device)

        # Forward pass through the reward network
        r_hat1 = self.network.reward.forward(obs_1, action_1, parameters)
        r_hat2 = self.network.reward.forward(obs_2, action_2, parameters)

        labels = labels.squeeze()  # Remove extra dimensions → shape becomes [32]
        # Ensure labels are in [B] shape
        if labels.dim() > 1:
            labels = labels.view(-1)


        # Handle the ensemble case
        if len(r_hat1.shape) > 1:
            E, B_times_S = r_hat1.shape
            out_shape = (E, B, S)
            labels = labels.unsqueeze(0).expand(E, -1)
        else:
            E, B_times_S = 0, r_hat1.shape[0]
            out_shape = (B, S)
        assert B_times_S == B * S, "shapes incorrect"

        r_hat1 = r_hat1.view(*out_shape).sum(dim=-1)  # Shape (E, B) or (B,)
        r_hat2 = r_hat2.view(*out_shape).sum(dim=-1)  # Shape (E, B) or (B,)
        logits = r_hat2 - r_hat1

        loss = self.reward_criterion(logits, labels).mean(dim=-1)
        if E > 0:
            loss = loss.sum(dim=0)

        # Compute the accuracy
        with torch.no_grad():
            pred = (logits > 0).to(dtype=labels.dtype)
            accuracy = (torch.round(pred) == torch.round(labels)).float().mean().item()
        return loss, accuracy

    def _inner_step(self, batch, train=True):
        accuracies = []
        parameters = {k: torch.clone(v) for k, v in self.network.reward.params.items()}
        for i in range(self.num_inner_steps):
            loss, accuracy = self._compute_loss_and_accuracy(batch, parameters)
            accuracies.append(accuracy)
            grads = torch.autograd.grad(loss, parameters.values(), create_graph=train)
            # for j, k in enumerate(parameters.keys()):
            #     parameters[k] = parameters[k] - self._inner_lrs[k] * grads[j]
            for j, k in enumerate(parameters.keys()):
                parameters[k] = parameters[k] - self._inner_lrs[j] * grads[j]

        with torch.no_grad():
            loss, accuracy = self._compute_loss_and_accuracy(batch, parameters)
            accuracies.append(accuracy)

        return parameters, accuracies

    def _outer_step(self, batch, train=True):
        
        outer_losses = []
        support_accuracies = []
        query_accuracies = []
        # for task_id, task in batch:
        # for task_id, task in enumerate(batch):
        for task in batch:
            # Split into support and query datasets
            # 🔍 Check if task is (index, dict)
            if isinstance(task, (tuple, list)) and isinstance(task[1], dict):
                task = task[1]

            if not isinstance(task, dict):
                print("Skipping invalid task:", type(task))
                continue

            batch_size = task["obs_1"].shape[0]
            if batch_size < self.num_support + self.num_query:
                continue
            
            batch_support = utils.get_from_batch(task, 0, end=self.num_support)
            batch_query = utils.get_from_batch(task, self.num_support, end=self.num_support + self.num_query)
            
            parameters, support_accuracy = self._inner_step(batch_support, train=train)
            loss, query_accuracy = self._compute_loss_and_accuracy(batch_query, parameters)
            outer_losses.append(loss)
            support_accuracies.append(support_accuracy)
            query_accuracies.append(query_accuracy)
        if len(outer_losses) == 0:
            return None, None, None  # Skip the last batch if it isn't full.
        outer_loss = torch.mean(torch.stack(outer_losses))
        support_accuracy = np.mean(support_accuracies, axis=0)
        query_accuracy = np.mean(query_accuracies)
        return outer_loss, support_accuracy, query_accuracy

    def _train_step(self, batch):
        self.optim["reward"].zero_grad()
        loss, support_accuracy, query_accuracy = self._outer_step(batch, train=True)
        if loss is None:
            print("Loss is None. Skipping this batch.")
            return {}

        # print(f"Loss: {loss.item()}")
        loss.backward()
        self.optim["reward"].step()
        # print(batch)
        try:
            inner = batch[0][1]
            obs_1 = inner["obs_1"]
            if isinstance(obs_1, torch.Tensor):
                obs_1 = obs_1.squeeze(1).detach().cpu().numpy()  # → (64, 25, 16)

            sample = obs_1[0]  # shape (25, 16)
            last_step = sample[-1]  # shape (16,)

            
            gripper_pos = last_step[:3]
            goal_pos = last_step[3:6]



            distance = np.linalg.norm(np.array(gripper_pos) - np.array(goal_pos))

        except Exception as e:
            print("Could not compute gripper-goal distance - train:", e)
            distance = -1.0


        metrics = {
            "outer_loss": loss.item(),
            "pre_adapt_support_accuracy": support_accuracy[0],
            "post_adapt_support_accuracy": support_accuracy[-1],
            "post_adapt_query_accuracy": query_accuracy,
            "gripper_goal_distance": distance,
        }
        # print(f"Train step [{self._steps}] metrics: {metrics}")
        return metrics

    def _validation_step(self, batch):
        loss, support_accuracy, query_accuracy = self._outer_step(batch, train=False)
        if loss is None:
            return {}

        try:
            inner = batch[0][1]
            obs_1 = inner["obs_1"]
            if isinstance(obs_1, torch.Tensor):
                obs_1 = obs_1.squeeze(1).detach().cpu().numpy()  # → (64, 25, 16)

            sample = obs_1[0]  # shape (25, 16)
            last_step = sample[-1]  # shape (16,)
            
            gripper_pos = last_step[:3]
            goal_pos = last_step[3:6]

            distance = np.linalg.norm(np.array(gripper_pos) - np.array(goal_pos))

        except Exception as e:
            print("Could not compute gripper-goal distance - valid:", e)
            distance = -1.0

        metrics = {
            "outer_loss": loss.item(),
            "pre_adapt_support_accuracy": support_accuracy[0],
            "post_adapt_support_accuracy": support_accuracy[-1],
            "post_adapt_query_accuracy": query_accuracy,
            "gripper_goal_distance": distance,
        }
        # print(f"Train step [{self._steps}] metrics: {metrics}")
        return metrics

    def _save_extras(self):
        return {"lrs": self._inner_lrs.state_dict()}

    def _load_extras(self, checkpoint, strict=True):
        self._inner_lrs.load_state_dict(checkpoint["lrs"], strict=strict)
