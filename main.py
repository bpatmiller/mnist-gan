import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from plot import save_images
from tqdm import tqdm
import pickle
from random import randint


class DiscriminatorNet(torch.nn.Module):
    """three hidden layers"""

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        num_features = 28 * 28

        self.layer0 = nn.Sequential(
            nn.Linear(
                num_features,
                1024),
            nn.LeakyReLU(.2),
            nn.Dropout(.3))
        self.layer1 = nn.Sequential(
            nn.Linear(
                1024,
                512),
            nn.LeakyReLU(.2),
            nn.Dropout(.3))
        self.layer2 = nn.Sequential(
            nn.Linear(
                512,
                256),
            nn.LeakyReLU(.2),
            nn.Dropout(.3))
        self.out = nn.Sequential(torch.nn.Linear(256, 1), torch.nn.Sigmoid())

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out(x)
        return x


class GeneratorNet(torch.nn.Module):
    """three hidden layers"""

    def __init__(self):
        super(GeneratorNet, self).__init__()
        num_features = 100
        num_out = 28 * 28

        self.epoch = 0
        self.layer0 = nn.Sequential(
            nn.Linear(
                num_features,
                256),
            nn.LeakyReLU(.2))
        self.layer1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(.2))
        self.layer2 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(.2))
        self.out = nn.Sequential(nn.Linear(1024, num_out), nn.Tanh())

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out(x)
        return x


def train_discriminator(discriminator, optimizer, loss, real_data, fake_data):
    size = real_data.size(0)
    optimizer.zero_grad()

    pred_real = discriminator(real_data)
    error_real = loss(pred_real, ones(size))
    error_real.backward()

    pred_fake = discriminator(fake_data)
    error_fake = loss(pred_fake, zeros(size))
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake, pred_real, pred_fake


def train_generator(discriminator, generator, optimizer, loss, fake_data):
    size = fake_data.size(0)
    optimizer.zero_grad()

    pred = discriminator(fake_data)
    error = loss(pred, ones(size))
    error.backward()

    optimizer.step()

    return error


def get_mnist_data():
    """download and normalize nmist dataset"""
    compose = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([.5], [.5])])
    return datasets.MNIST(
        root='./data',
        train=True,
        transform=compose,
        download=True)


def imgs_to_vecs(imgs):
    return imgs.view(imgs.size(0), 28 * 28)


def vecs_to_imgs(vecs):
    return vecs.view(vecs.size(0), 1, 28, 28)


def noise(size):
    """returns a vector of gaussian random numbers"""
    return Variable(torch.randn(size, 100))

def ones(size):
    return Variable(torch.ones(size, 1))


def zeros(size):
    return Variable(torch.zeros(size, 1))


def main():
    start_epoch = 0
    num_epochs = 200
    num_example_samples = 16
    sample_noise = noise(num_example_samples)

    dataset = get_mnist_data()
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True)
    num_batches = len(dataset_loader)

    try:
        with open('discriminator.obj', 'rb') as f:
            discriminator = pickle.load(f)
        with open('generator.obj', 'rb') as f:
            generator = pickle.load(f)
        with open('noise.obj', 'rb') as f:
            sample_noise = pickle.load(f)
        start_epoch = generator.epoch
        print('GAN loaded from file, starting at epoch {}'.format(start_epoch))
    except BaseException:
        discriminator = DiscriminatorNet()
        generator = GeneratorNet()

    discrim_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    loss = nn.BCELoss()

    # save the sample noise for consistency when pausing/resuming
    with open('noise.obj', 'wb') as f:
        pickle.dump(sample_noise, f)
    # train the model and output generated images
    for epoch in range(start_epoch, num_epochs):
        print("training epoch {:d}".format(epoch))
        with tqdm(total=num_batches) as t:
            for batch_number, (real_batch, _) in enumerate(dataset_loader):
                size = real_batch.size(0)

                # train discriminator
                real_data = Variable(imgs_to_vecs(real_batch))
                fake_data = generator(noise(size)).detach()
                discrim_error, discrim_pred_real, discrim_pred_fake = train_discriminator(
                    discriminator, discrim_optimizer, loss, real_data, fake_data)

                # train generator
                fake_data = generator(noise(size))
                gen_error = train_generator(
                    discriminator, generator, gen_optimizer, loss, fake_data)

                t.update(1)
                # output samples
                if batch_number % 100 == 0:
                    sample_images = vecs_to_imgs(generator(sample_noise)).data
                    save_images(
                        sample_images,
                        epoch,
                        batch_number,
                        num_example_samples)

        generator.epoch += 1
        with open('discriminator.obj', 'wb') as f:
            pickle.dump(discriminator, f)
        with open('generator.obj', 'wb') as f:
            pickle.dump(generator, f)

    print('{0}done training\n{0}'.format(10*'-'+'\n'))

    def valid_noise():
        x = noise(1)
        print('finding a valid point in latent space... ', end='')
        while discriminator(generator(x))[0][0].item() < 0.9:
            x = noise(1)
        print('done')
        return x

    # now show a sweep through the 100-dimension latent space
    num_steps = 30
    num_points = 10

    x2 = valid_noise()
    for point in range(num_points):
        x1 = x2
        x2 = valid_noise()
        delta = (x2 - x1) / float(num_steps)
        for step in range(num_steps):
            sweep_image = vecs_to_imgs(generator(x1)).data
            save_images(
                sweep_image,
                point,
                step,
                1,
                out_dir='./sweep_images'
            )
            x1 += delta


if __name__ == '__main__':
    main()
