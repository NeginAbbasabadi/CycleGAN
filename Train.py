import os
import random
import torch
import torch.nn as nn
import itertools
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from your other files
from Dataset import UnpairedArtDataset, art_transforms
from Models import GeneratorResNet, Discriminator

# Initialize data
dataset = UnpairedArtDataset(root_dir='/Users/macbook/CycleGAN/monet2photo', transform=art_transforms, mode='train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize networks
G_A2B = GeneratorResNet()
G_B2A = GeneratorResNet()
D_A = Discriminator()
D_B = Discriminator()

# Loss criteria and Optimizers
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
G_A2B.to(device)
G_B2A.to(device)
D_A.to(device)
D_B.to(device)
criterion_GAN.to(device)
criterion_cycle.to(device)
criterion_identity.to(device)

epochs = 5 
lambda_cyc = 10.0
lambda_id = 5.0   

save_dir = './CycleGAN_Checkpoints'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        valid = torch.ones((real_A.size(0), 1, real_A.size(2) // 16, real_A.size(3) // 16), device=device, requires_grad=False)
        fake = torch.zeros((real_A.size(0), 1, real_A.size(2) // 16, real_A.size(3) // 16), device=device, requires_grad=False)

        # Train Generators
        optimizer_G.zero_grad()
        loss_id_A = criterion_identity(G_B2A(real_A), real_A)
        loss_id_B = criterion_identity(G_A2B(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        fake_B = G_A2B(real_A)
        loss_GAN_A2B = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_B2A(real_B)
        loss_GAN_B2A = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2

        recov_A = G_B2A(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_A2B(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        loss_G = loss_GAN + (lambda_cyc * loss_cycle) + (lambda_id * loss_identity)
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator A
        optimizer_D_A.zero_grad()
        loss_real_A = criterion_GAN(D_A(real_A), valid)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake_A = criterion_GAN(D_A(fake_A_.detach()), fake)
        loss_D_A = (loss_real_A + loss_fake_A) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # Train Discriminator B
        optimizer_D_B.zero_grad()
        loss_real_B = criterion_GAN(D_B(real_B), valid)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake_B = criterion_GAN(D_B(fake_B_.detach()), fake)
        loss_D_B = (loss_real_B + loss_fake_B) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D_A.item() + loss_D_B.item():.4f}] [G loss: {loss_G.item():.4f}]")
            
    torch.save(G_A2B.state_dict(), f"{save_dir}/G_A2B_epoch_{epoch}.pth")
    torch.save(G_B2A.state_dict(), f"{save_dir}/G_B2A_epoch_{epoch}.pth")
    torch.save(D_A.state_dict(), f"{save_dir}/D_A_epoch_{epoch}.pth")
    torch.save(D_B.state_dict(), f"{save_dir}/D_B_epoch_{epoch}.pth")
    print(f"Epoch {epoch} checkpoints saved locally.")
