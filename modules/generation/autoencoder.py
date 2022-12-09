import torch
import torch.nn as nn
import torch.nn.functional as F

class AccidentSeverityVariationalAutoencoder(nn.Module):
    def __init__(self, num_features=25, num_dim=74):
        super().__init__()

        self.num_features = num_features
        self.num_dim = num_dim 

        self.encoder_layer_1 = nn.Linear(in_features=self.num_dim, out_features=100)
        self.encoder_layer_2 = nn.Linear(in_features=100, out_features=50)
        self.encoder_layer_3 = nn.Linear(in_features=50, out_features=(self.num_features * 2))

        self.decoder_layer_1 = nn.Linear(in_features=self.num_features, out_features=50)
        self.decoder_layer_2 = nn.Linear(in_features=50, out_features=100)
        self.decoder_layer_3 = nn.Linear(in_features=100, out_features=self.num_dim)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)    # sampling as if coming from the input space
        
        return sample

    def encode(self, x):
        # encoding
        x = F.relu(self.encoder_layer_1(x))
        x = F.relu(self.encoder_layer_2(x))
        x = self.encoder_layer_3(x).view(-1, 2, self.num_features)
        
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        
        return z, mu, log_var

    def decode(self, z, mu, log_var):
        # decoding
        x = F.relu(self.decoder_layer_1(z))
        x = F.relu(self.decoder_layer_2(x))
        reconstruction = torch.sigmoid(self.decoder_layer_3(x))
        
        return reconstruction, mu, log_var

    def sample(self, mu, log_var):
        z = self.reparameterize(mu, log_var)
        reconstruction, mu, log_var = self.decode(z, mu, log_var)
        
        return reconstruction

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        reconstruction, mu, log_var = self.decode(z, mu, log_var)
        
        return reconstruction, mu, log_var

