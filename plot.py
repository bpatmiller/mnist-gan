from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch


def save_images(images, epoch, batch_number, num_images):
    out_dir = './images'
    img_name = 'sample_{}-{}'.format(epoch, batch_number)

    if type(images) == np.ndarray:
        images = torch.from_numpy(images)
    
    nrows = int(np.sqrt(num_images))
    grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)
    
    fig = plt.figure()
    plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
    plt.axis('off')

    fig.tight_layout()
    fig.savefig('{}/{}.png'.format(out_dir, img_name), bbox_inches='tight')

    plt.close()