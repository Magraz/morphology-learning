import torch
import torch.nn as nn

from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        n_agents_train: int,
        n_agents_eval: int,
        d_state: int,
        d_action: int,
        device: str,
    ):
        super(ActorCritic, self).__init__()

        self.log_action_std = nn.Parameter(
            torch.zeros(
                d_action * n_agents_train,
                requires_grad=True,
                device=device,
            )
            * -0.5
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(d_state, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d_action * n_agents_train),
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(d_state, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return value, action_mean

    def get_value(self, state: torch.Tensor):
        with torch.no_grad():
            return self.critic(state)

    def get_action_dist(self, action_mean):
        action_std = torch.exp(self.log_action_std)
        return Normal(action_mean, action_std)

    def act(self, state, deterministic=False):

        value, action_mean = self.forward(state)

        if deterministic:
            return action_mean.detach()

        dist = self.get_action_dist(action_mean)
        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        return (
            action.detach(),
            action_logprob.detach(),
            value.detach(),
        )

    def evaluate(self, state, action):

        value, action_mean = self.forward(state)

        dist = self.get_action_dist(action_mean)
        action_logprobs = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

        return action_logprobs, value, dist_entropy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActorCritic(
        n_agents_train=4,
        n_agents_eval=4,
        d_state=4 * 18,
        d_action=2 * 4,
        device=device,
    ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
