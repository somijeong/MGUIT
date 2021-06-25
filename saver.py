import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
import torch

# tensor to PIL Image
def tensor2img(img):
    img = img[0].cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)

# save a set of images
def save_imgs(img, path):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(path)

def write_memory(fpath, m_items):
    item = []
    for it in range(0, 4):
        items = m_items[it]
        m_item = []
        m_item.append(items[0].cpu().detach())
        m_item.append(items[1].cpu().detach())
        m_item.append(items[2].cpu().detach())
        item.append(m_item)
    torch.save(item, fpath)

class Saver():
    def __init__(self, opts):
        self.model_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.memory_dir = os.path.join(self.model_dir, 'memory')
        self.ckpt_dir = os.path.join(self.model_dir, 'ckpt')
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        # make directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    # save result images
    def write_img(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.jpg' % (self.image_dir)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model, m_items):
        if (ep + 1) % self.model_save_freq == 0:
            print('--- save the model @ ep %d ---' % (ep))
            model.save('%s/%05d.pth' % (self.ckpt_dir, ep), ep, total_it)
            write_memory('%s/%05d_memory.pt' % (self.memory_dir, ep), m_items)
        elif ep == -1:
            model.save('%s/last.pth' % self.ckpt_dir, ep, total_it)
            write_memory('%s/last_memory.pt' % (self.memory_dir), m_items)

    def write_mid(self, ep, total_it, model, m_items):
        model.save('%s/latest.pth' % (self.model_dir), ep, total_it)
        write_memory('%s/latest_memory.pt' % (self.model_dir), m_items)

        assembled_images = model.assemble_outputs()
        img_filename = '%s/gen_%05d_%05d.jpg' % (self.image_dir, ep, total_it)
        torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
