import torch
import neural_ode


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        out = self.layer1(x):
        out = F.Gelu(out)
        out = self.layer2(out)

        return out

class Classifier(nn.Module):
    def __init__():
        #self.encoder = LatentEncoder()
        self.real = NeuralODE()
        self.fake = NeuralODE()
        self.mlp = MLP()

    def forward(self, x):
        
        #neuralODE has to inv_sample
        real = self.real.inv_sample(out)
        fake = self.fake.inv_sample(out)
        out = torch.cat([real, fake])
        out = self.mlp(out)

        return out

