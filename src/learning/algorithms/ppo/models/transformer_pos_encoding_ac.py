import math
import torch
import torch.nn as nn
from enum import Enum
from torch.distributions.normal import Normal


class SpecialTokens(Enum):
    PADDING = 0
    SOS = 2
    START_OF_STATE = 3
    START_OF_ACTION = 4
    EOS = 5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(p=dropout)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(1e5)) / d_model
        )  # 1000^(2i/d_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/d_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/d_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]
        )


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
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        use_autoregress: bool = False,
        use_pos_encoding: bool = False,
    ):
        super(ActorCritic, self).__init__()

        # INFO
        self.d_model = d_model
        self.n_agents_eval = n_agents_eval
        self.device = device
        self.d_action = d_action
        self.use_autoregress = use_autoregress
        self.use_pos_encoding = use_pos_encoding

        # LAYERS
        self.log_action_std = nn.Parameter(
            torch.ones(d_action * n_agents_train, requires_grad=True, device=device)
            * -0.5
        )

        self.positional_encoder = PositionalEncoding(
            d_model=d_model, dropout=0.1, max_len=500
        )

        self.special_token_embedding = nn.Embedding(
            len(SpecialTokens), d_model, padding_idx=0
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.state_embedding = nn.Sequential(
            nn.LayerNorm(d_state), nn.Linear(d_state, d_model), nn.GELU()
        )

        self.action_embedding = nn.Sequential(
            nn.LayerNorm(d_action), nn.Linear(d_action, d_model), nn.GELU()
        )

        # Encoder Params
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, batch_first=True, norm_first=True, dim_feedforward=1024
        )
        self.enc = nn.TransformerEncoder(
            encoder_layer, n_encoder_layers, enable_nested_tensor=False
        )

        # Decoder Params
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, batch_first=True, norm_first=True, dim_feedforward=1024
        )
        self.dec = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

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

    def forward(self):
        raise NotImplementedError

    def get_value(self, state: torch.Tensor):
        with torch.no_grad():
            embedded_state = self.state_embedding(state)

            if self.use_pos_encoding:
                embedded_state = self.positional_encoder(embedded_state)

            encoder_out = self.enc(embedded_state)

            return self.value_head(encoder_out[:, 0])

    def auto_regress(self, encoder_out):
        batch_dim = encoder_out.shape[0]
        # Pre-allocate action tensor
        action_means = torch.zeros(
            batch_dim, self.n_agents_eval, self.d_action, device=self.device
        )
        # Start with SOS token
        tgt = (
            self.special_token_embedding(
                torch.tensor(SpecialTokens.SOS.value, device=self.device)
            )
            .view(1, 1, self.d_model)
            .repeat(batch_dim, 1, 1)
        )

        if self.use_pos_encoding:
            # Apply positional encoding to initial token
            tgt = self.positional_encoder(tgt)

        for idx in range(self.n_agents_eval):
            # Create mask for current target length
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.shape[1], device=self.device
            )
            decoder_out = self.dec(
                tgt,
                memory=encoder_out,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
            )

            # Generate action for current position
            action_means[:, idx] = self.actor_head(decoder_out[:, -1]).squeeze(1)

            # If not the last iteration, prepare next token
            if idx < self.n_agents_eval - 1:
                # Extract newly generated embedding
                next_token_embedding = decoder_out[:, -1:, :]

                # Concatenate with existing sequence
                tgt = torch.cat([tgt, next_token_embedding], dim=1)

                tgt = self.layer_norm(tgt)

                if self.use_pos_encoding:
                    # Reapply positional encoding to the entire sequence
                    tgt = self.positional_encoder(tgt)

        return action_means

    def act(self, state, deterministic=False):

        embedded_state = self.state_embedding(state)

        if self.use_pos_encoding:
            embedded_state = self.positional_encoder(embedded_state)

        encoder_out = self.enc(embedded_state)

        if self.use_autoregress:
            action_mean = self.auto_regress(encoder_out)

        else:
            decoder_input = encoder_out.clone().detach()

            if self.use_pos_encoding:
                decoder_input = self.positional_encoder(decoder_input)

            decoder_out = self.dec(
                tgt=decoder_input,
                memory=encoder_out,
            )

            action_mean = self.actor_head(decoder_out)

        if deterministic:
            return action_mean.flatten(start_dim=1).detach()

        action_std = torch.exp(
            self.log_action_std[: encoder_out.shape[1] * self.d_action]
        )

        dist = Normal(action_mean.flatten(start_dim=1), action_std)

        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        value = self.value_head(encoder_out[:, 0])

        return (
            action.detach(),
            action_logprob.detach(),
            value.detach(),
        )

    def evaluate(self, state, action):

        embedded_state = self.state_embedding(state)

        if self.use_pos_encoding:
            embedded_state = self.positional_encoder(embedded_state)

        encoder_out = self.enc(embedded_state)

        if self.use_autoregress:
            embedded_action = self.action_embedding(
                action.reshape(
                    action.shape[0],
                    self.n_agents_eval,
                    self.d_action,
                )
            )

            if self.use_pos_encoding:
                embedded_action = self.positional_encoder(embedded_action)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                embedded_action.shape[1], device=self.device
            )
            decoder_out = self.dec(
                tgt=embedded_action,
                memory=encoder_out,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
            )
        else:

            if self.use_pos_encoding:
                decoder_input = self.positional_encoder(encoder_out)

            decoder_out = self.dec(tgt=decoder_input, memory=encoder_out)

        action_mean = self.actor_head(decoder_out)

        action_std = torch.exp(
            self.log_action_std[: encoder_out.shape[1] * self.d_action]
        )

        dist = Normal(action_mean.flatten(start_dim=1), action_std)

        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        value = self.value_head(encoder_out[:, 0])

        return action_logprob, value, dist_entropy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActorCritic(
        n_agents=4,
        d_state=18,
        d_action=2,
        device=device,
    ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
