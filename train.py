import torch
from math import log2
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models import Generator,Discriminator
from utils import save_checkpoint,load_checkpoint,save_on_tensorboard,gradient_penalty


#### Configurations ####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

START_TRAIN_IMG_SIZE = 16
DATASET = "celebA-hq/celeba_hq"                 ## Dataset Directory

CHECKPOINT_GEN = "./Checkpoint/generator.pth"   ## Checkpoints Directory
CHECKPOINT_CRITIC = "./Checkpoint/critic.pth"
SAVE_MODEL = True
LOAD_MODEL = False

LR = 1e-3
BATCH_SIZES = [32,32,32,32,16,16,16,16,8,4]     ## modifiable/ Batch_sizes for each step
IMAGE_SIZE = 1024                               ## 1024 in paper
IMG_CHANNELS = 3
Z_DIM = 512                                     ## 512 in paper
IN_CHANNELS = 512                               ## 512 in paper
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE/4)) + 1

PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8,Z_DIM,1,1).to(DEVICE)
# NUM_WORKERS = 4
NUM_WORKERS = 0

#### Train ####
torch.backends.cudnn.benchmarks = True

def get_loader(img_size):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
        ])

    batch_size = BATCH_SIZES[int(log2(img_size / 4))]
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    return loader, dataset


def train_fn(gen, critic, loader, dataset, step, alpha, opt_gen, opt_critic, tensorboard_step, writer, scaler_gen,
             scaler_critic):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)

        ## Train Critic
        ## Wasserstein Loss : Maximize "E[Critic(real)] - E[Critic(fake)]"   ==   Minimize "-(E[Critic(real)] - E[Critic(fake)])"
        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step).to(DEVICE)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=DEVICE)
            loss_critic = -1 * (
                        torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp + 0.001 * torch.mean(
                critic_real ** 2)

        critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        ## Train Generator
        ## Maximize "E[Critic(fake)]"   ==   Minimize "- E[Critic(fake)]"
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -1 * torch.mean(gen_fake)

        gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += (cur_batch_size / len(dataset)) * (1 / PROGRESSIVE_EPOCHS[step]) * 2
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(FIXED_NOISE, alpha, step) * 0.5 + 0.5
                save_on_tensorboard(writer, loss_critic.item(), loss_gen.item(), real.detach(), fixed_fakes.detach(),
                                    tensorboard_step)
                tensorboard_step += 1

    return tensorboard_step, alpha

def generate_examples(gen, current_epoch, steps, n=16):
    gen.eval()
    aplha = 1.0

    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
            generated_img = gen(noise, alpha=alpha, steps=steps)
            #             save_image(generated_img*0.5+0.5,f"generated_images/epoch{current_epoch}_{i}.png")
            save_image(generated_img * 0.5 + 0.5, f"./generated_images/step:{steps}_epoch{current_epoch}_{i}.png")

    gen.train()


## build model
gen = Generator(Z_DIM, IN_CHANNELS, IMG_CHANNELS).to(DEVICE)
critic = Discriminator(IN_CHANNELS, IMG_CHANNELS).to(DEVICE)

## initialize optimizer,scalers (for FP16 training)
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.99))
scaler_gen = torch.cuda.amp.GradScaler()
scaler_critic = torch.cuda.amp.GradScaler()

## tensorboard writer
writer = SummaryWriter(f"runs/ProGAN")
tensorboard_step = 0

## if checkpoint files exist, load model
if LOAD_MODEL:
    load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LR)
    load_checkpoint(CHECKPOINT_CRITIC, critic, opt_critic, LR)

gen.train()
critic.train()

step = int(log2(START_TRAIN_IMG_SIZE / 4))  ## starts from 0

global_epoch = 0
generate_examples_at = [ 10*i for i in range(1,11)]

for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-4
    loader, dataset = get_loader(4 * 2 ** step)
    print(f"Image size:{4 * 2 ** step} | Current step:{step}")

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}] Global Epoch:{global_epoch}")
        tensorboard_step, alpha = train_fn(gen, critic, loader, dataset, step, alpha, opt_gen, opt_critic,
                                           tensorboard_step, writer, scaler_gen, scaler_critic)
        global_epoch += 1
        if global_epoch in generate_examples_at:
            generate_examples(gen, global_epoch, step, n=6)

        if SAVE_MODEL and (epoch + 1) % 8 == 0:
            save_checkpoint(gen, opt_gen, filename="CHECKPOINT_GEN")
            save_checkpoint(critic, opt_critic, filename="CHECKPOINT_CRITIC")

    step += 1  ## Progressive Growing

print("Training finished")