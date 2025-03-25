import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
batch_size = 64
epochs = 50

# Pré-processamento dos dados
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregar dataset MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Definir o Gerador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


# Definir o Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        validity = self.model(flattened)
        return validity


# Inicializar modelos
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Otimizadores
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Função de perda
adversarial_loss = nn.BCELoss()

# Treinamento
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # Dados reais e falsos
        real_imgs = imgs.to(device)
        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # ---------------------
        #  Treinar Discriminador
        # ---------------------

        optimizer_D.zero_grad()

        # Perda com dados reais
        real_loss = adversarial_loss(discriminator(real_imgs), valid)

        # Gerar dados falsos
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)

        # Perda com dados falsos
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        # Perda total do discriminador
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Treinar Gerador
        # -----------------

        optimizer_G.zero_grad()

        # Gerar dados falsos
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)

        # Perda do gerador (tentar enganar o discriminador)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

    # Mostrar progresso
    print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Salvar algumas imagens geradas
    if epoch % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            gen_imgs = generator(z)

            fig, axs = plt.subplots(4, 4, figsize=(4, 4))
            cnt = 0
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(gen_imgs[cnt, 0, :, :].cpu().numpy(), cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            plt.show()