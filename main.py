import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from plot import save_images
from tqdm import tqdm


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
        root='./out_dir',
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
    num_epochs = 250
    num_example_samples = 16
    sample_noise = noise(num_example_samples)

    dataset = get_mnist_data()
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True)
    num_batches = len(dataset_loader)

    discriminator = DiscriminatorNet()
    generator = GeneratorNet()

    discrim_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    loss = nn.BCELoss()

    # train the model and output generated images
    for epoch in range(num_epochs):
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
                if epoch % 10 == 0:
                    sample_images = vecs_to_imgs(generator(sample_noise)).data
                    save_images(sample_images)


if __name__ == '__main__':
    main()
