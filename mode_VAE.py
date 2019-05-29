import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):
    def __init__(self,input_size,hidden_size,G_size):
        super(VAE, self).__init__()

        self.input_size = input_size[0] * input_size[1] * input_size[2]
        self.hidden_size = hidden_size
        self.G_size = G_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2_mu = nn.Linear(self.hidden_size, self.G_size)
        self.fc2_sigma = nn.Linear(self.hidden_size, self.G_size)
        self.fc3 = nn.Linear(self.G_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_size)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_sigma(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        error = torch.randn_like(std)
        return mu + error*std

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = self.fc4(h3)
        return torch.sigmoid(h4)

    def forward(self, x):
        #mu: mean value logvar:get log(sigma^2) directly
        mu, logvar = self.encoder(x.view(-1, self.input_size))
        p = self.reparameterize(mu, logvar)
        out = self.decoder(p)
        return out, mu, logvar

# pixel loss and KL-divergence
def loss_function(new_x, x, mu, logvar,input_size):
      BCE = F.binary_cross_entropy(new_x, x.view(-1, input_size), reduction='sum')
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      return BCE + KLD

def train(epoch,train_loader,input_size):
      model.train()
      sum_loss = 0
      one_d_input = input_size[0] * input_size[1] * input_size[2]
      for batch_idx, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            new_batch, mu, logvar = model(batch)
            loss = loss_function(new_batch, batch, mu, logvar,one_d_input)
            loss.backward()

            sum_loss += loss.item()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                  print('Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * batch_idx / len(train_loader),
                  loss.item() / len(batch)))

def test(epoch,test_loader,input_size):
      model.eval()
      sum_loss = 0
      one_d_input = input_size[0] * input_size[1] * input_size[2]
      with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(test_loader):
                  batch = batch.to(device)
                  new_batch, mu, logvar = model(batch)
                  sum_loss += loss_function(new_batch, batch, mu, logvar, one_d_input).item()
                  if batch_idx == 0:
                        n = min(len(batch), 8)
                        img = torch.cat([batch[:n],new_batch.view(len(batch), input_size[0], input_size[1], input_size[2])[:n]])
                        save_image(img.cpu(),'VAEresults/compare_' + str(epoch) + '.png', nrow=n)

      print('Test loss: {:.6f}'.format(sum_loss / len(test_loader.dataset)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model_VAE')
    parser.add_argument('--input-size', type=list, default=[1,28,28],help='input size(MNIST as default:[1,28,28])')
    parser.add_argument('--hidden-size', type=int, default=400,help='out_dim of the first fc layer(400 for MNIST)')
    parser.add_argument('--G-size', type=int, default=20,help='dim of mu and sigma = num of Gaussian')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False,help=' cuda or not ')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='interval of logging training status')
    args = parser.parse_args()

    #cuda or not
    args.cuda = args.cuda and torch.cuda.is_available()
    #random seed
    torch.manual_seed(args.seed)
    #device
    if args.cuda :
        device = torch.device("cuda")
        dataloader_args = {'num_workers': 1, 'pin_memory': True}
    else:
        device = torch.device("cpu")
        dataloader_args = {}

    #load data and tranform to tensor
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **dataloader_args)

    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **dataloader_args)

    #model and optimizer
    model = VAE(args.input_size,args.hidden_size,args.G_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        train(epoch,train_loader,args.input_size)
        test(epoch,test_loader,args.input_size)
        with torch.no_grad():
                fake = torch.randn(64, args.G_size).to(device)
                fake = model.decoder(fake).cpu()
                save_image(fake.view(64, args.input_size[0], args.input_size[1], args.input_size[2]),
                    'VAEresults/fake_' + str(epoch) + '.png')