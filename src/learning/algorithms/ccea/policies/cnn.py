import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class CNN_Policy(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
    ):
        super(CNN_Policy, self).__init__()

        dim1 = 24
        dim2 = 48

        self.cnn = nn.Conv1d(in_channels=3, out_channels=dim1, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv1d(
            in_channels=dim1, out_channels=dim2, kernel_size=3, stride=1
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.linear = nn.Linear(dim2, 1)
        self.num_params = nn.utils.parameters_to_vector(self.parameters()).size()[0]

        # Disable gradient calcs
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        out = F.leaky_relu(self.cnn(x.unsqueeze(0)))
        out = F.leaky_relu(self.cnn2(out))
        out = self.global_avg_pool(out)
        out = self.linear(out.flatten())
        return F.sigmoid(out).unsqueeze(-1)

    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, params: torch.Tensor):
        nn.utils.vector_to_parameters(params, self.parameters())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN_Policy().to(device)
    model_copy = deepcopy(model)
    print(model_copy.num_params)

    input = torch.ones((1, 160), dtype=torch.float).to(device)
    print(model_copy.forward(input))

    rand_params = torch.rand(model_copy.get_params().size()).to(device)
    mutated_params = torch.add(model_copy.get_params(), rand_params).to(device)

    model_copy.set_params(mutated_params)

    print(model_copy.forward(input))
