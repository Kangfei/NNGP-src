import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
import gpytorch

class FC(Module):
    def __init__(self, in_ch, out_ch):
        super(FC, self).__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class MLP(Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(MLP, self).__init__()
        self.fc1 = FC(in_ch, hid_ch)
        self.fc2 = FC(hid_ch, out_ch)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MultiTaskMLP(Module):
    def __init__(self, in_ch, hid_ch, reg_out_ch, cla_out_ch):
        super(MultiTaskMLP, self).__init__()
        self.fc1 = FC(in_ch, hid_ch)
        self.reg_layer = FC(hid_ch, reg_out_ch)
        self.cla_layer = FC(hid_ch, cla_out_ch)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.reg_layer(x), F.log_softmax(self.cla_layer(x), dim = 1)

class MLPDensityRegressor(Module):
    def __init__(self, in_ch, hid_ch):
        super(MLPDensityRegressor, self).__init__()
        self.fc1 = FC(in_ch, hid_ch)
        #self.fc2 = FC(hid_ch, 2)
        self.mu_layer = FC(hid_ch, 1)
        self.sigma_layer = FC(hid_ch, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        #mu, sigma = torch.split(x, 1, dim=1)

        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)
        #sigma_pos = torch.log(torch.exp(sigma) + 1) + 1e-6
        sigma_pos = F.softplus(sigma) + 1e-6
        return mu, sigma_pos

    def loss(self, y, mu, sigma_pos, constant= 10.0):
        # Negative Log likelihood
        #loss = tf.reduce_mean(0.5 * tf.log(output_sig_pos) + 0.5 * tf.div(tf.square(y - output_mu), output_sig_pos)) + 10
        #return torch.mean(0.5 * torch.log(sigma_pos) + 0.5 * torch.div(torch.square(y - mu), sigma_pos)) + constant
        return (0.5 * (torch.log(sigma_pos) + ((y - mu).pow(2))/sigma_pos)).mean()


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),num_dims=2, grid_size=100)
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        #self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MCDropoutModel(Module):
    def __init__(self, input_dim, output_dim, hid_dim, dropout = 0.5, num_samples = 100):
        super(MCDropoutModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_samples = num_samples

        self.fc1 = FC(input_dim, hid_dim)
        self.mu_layer = FC(hid_dim, output_dim)
        self.sigma_layer = FC(hid_dim, output_dim)
        self.log_noise = torch.nn.Parameter(torch.FloatTensor([0]))
        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=True) # always dropout
        #print(x)
        mu, sigma = self.mu_layer(x), self.sigma_layer(x)
        #sigma_pos = F.softplus(sigma) + 1e-6
        #sigma = sigma.exp()
        #print(mu, sigma)
        sigma = torch.exp(self.log_noise)
        return mu, sigma

    def loss(self, mu, y, sigma):
        #return (0.5 * (torch.log(sigma) + ((y - mu).pow(2)) / sigma)).mean()
        return (torch.log(sigma) + 0.5 * (mu - y).pow(2) / sigma.pow(2)).mean()


    def predict(self, x):
        means, stds = [], []
        for _ in range(self.num_samples):
            mu, sigma = self.forward(x)
            means.append(mu)
            stds.append(sigma)
        #means, stds = torch.cat(means, dim=1), torch.cat(stds, dim=1)
        means = torch.cat(means, dim=1)
        mean = means.mean(dim=-1)
        std = (means.var(dim=-1)).sqrt()
        #std = (means.var(dim= -1) + stds.mean(dim=-1).pow(2)).sqrt()
        return mean, std




