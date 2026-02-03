import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt


class ActorCritic(nn.Module):

    # Constructor
    def __init__(
        self,
        n_agents_train: int,
        n_agents_eval: int,
        d_state: int,
        d_action: int,
        device: str,
        # Model specific
        d_model: int = 64,
        n_heads: int = 2,
        n_encoder_layers: int = 2,
    ):
        super(ActorCritic, self).__init__()

        # INFO
        self.d_model = d_model
        self.n_agents_eval = n_agents_eval
        self.n_agents_train = n_agents_train
        self.device = device
        self.d_action = d_action

        # LAYERS
        self.log_action_std = nn.Parameter(
            torch.ones(d_action * n_agents_train, requires_grad=True, device=device)
            * -0.5
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.state_embedding = nn.Sequential(
            nn.LayerNorm(d_state), nn.Linear(d_state, d_model), nn.GELU()
        )

        # Encoder Params
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, batch_first=True, norm_first=True, dim_feedforward=1024
        )
        self.enc = nn.TransformerEncoder(
            encoder_layer, n_encoder_layers, enable_nested_tensor=False
        )

        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, d_action),
        )

        # Critic
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

        # Add attention pooling for value function
        self.value_pool = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))

    def forward(self, state: torch.Tensor):
        embedded_state = self.state_embedding(state)

        encoder_out = self.enc(embedded_state)

        # Forward actor
        action_mean = self.actor_head(encoder_out)

        # Attention-pooled value function
        attn_weights = self.value_pool(encoder_out)
        value = self.value_head(torch.sum(encoder_out * attn_weights, dim=1))

        return action_mean, value

    def get_value(self, state: torch.Tensor):
        with torch.no_grad():
            _, value = self.forward(state)
            return value

    def act(self, state, deterministic=False):

        action_mean, value = self.forward(state)

        if deterministic:
            return action_mean.flatten(start_dim=1).detach()

        action_std = torch.exp(
            self.log_action_std[: self.n_agents_train * self.d_action]
        )

        dist = Normal(action_mean.flatten(start_dim=1), action_std)

        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        return (
            action.detach(),
            action_logprob.detach(),
            value.detach(),
        )

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):

        action_mean, value = self.forward(state)

        action_std = torch.exp(self.log_action_std[: state.shape[1] * self.d_action])

        dist = Normal(action_mean.flatten(start_dim=1), action_std)

        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        return action_logprob, value, dist_entropy

    def build_attention_hooks(self):
        # --- monkey-patch all attention modules to return weights ----------
        for layer in self.enc.layers:
            old_fwd = layer.self_attn.forward

            def make_enc_fwd(old_fwd):
                def new_fwd(*args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    return old_fwd(*args, **kwargs)

                return new_fwd

            layer.self_attn.forward = make_enc_fwd(old_fwd)

        # --- hook to grab the weights --------------------------------------------------
        attn_scores = {}

        def make_hook(name):
            def hook(_, __, out):
                # MultiheadAttention returns (attn_output, attn_weights)
                if isinstance(out, tuple) and len(out) == 2:
                    attn_scores[name] = out[1].detach()  # Just store the weights
                else:
                    print(f"Warning: Unexpected output format for {name}: {type(out)}")
                    attn_scores[name] = out

            return hook

        # Add encoder self-attention hooks
        for i, layer in enumerate(self.enc.layers):
            layer.self_attn.register_forward_hook(make_hook(f"Enc_L{i}"))

        return attn_scores


if __name__ == "__main__":
    from learning.plotting.utils import (
        plot_all_attention_heads,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActorCritic(
        n_agents_train=8,
        n_agents_eval=8,
        d_state=18,
        d_action=2,
        device=device,
    ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    attn_scores = model.build_attention_hooks()

    # -----------------------------------------------------------------------
    x = torch.randn(1, 8, 18).to(device)  # (batch, sequence, d_model)
    x = model.state_embedding(x)
    enc_out = model.enc(x)

    print(attn_scores["Enc_L0"].shape)

    # Encoder self-attention
    fig_enc = plot_all_attention_heads(
        attn_scores["Enc_L0"], layer_name="Encoder Layer 0"
    )
    plt.savefig("transformer_encoder_attention.png", dpi=300, bbox_inches="tight")
