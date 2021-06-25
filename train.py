import torch
from options import TrainOptions
from dataset import dataset_unpair
from model import MGUIT
from saver import Saver
import torch.nn.functional as F
import os

def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()
    torch.cuda.set_device(opts.gpu)

    # data loader
    print('\n--- load dataset ---')
    dataset = dataset_unpair(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = MGUIT(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(os.path.join(opts.result_dir, opts.name, opts.resume), opts.gpu)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opts)

    # memory initialize
    if opts.resume is None:
        print('\n---initialize memory items---')
        m_items = []
        for i in range(0, 4):
            m_item_ = []
            m_item_.append(F.normalize(torch.randn((opts.msize[i], opts.kdim), dtype=torch.float), dim=1).cuda(opts.gpu))    # key
            m_item_.append(torch.randn((opts.msize[i], opts.vdim), dtype=torch.float).cuda(opts.gpu)) # value A
            m_item_.append(torch.randn((opts.msize[i], opts.vdim), dtype=torch.float).cuda(opts.gpu)) # value B
            m_items.append(m_item_)
        del m_item_
    else:
        print('\n---load memory items---')
        m_item = torch.load(os.path.join(opts.result_dir, opts.name, opts.resume.replace('.pth','_memory.pt')))
        m_items = []
        for i in range(0, 4):
            tmp = m_item[i]
            m_item_ = []
            m_item_.append(tmp[0].cuda(opts.gpu))   # key
            m_item_.append(tmp[1].cuda(opts.gpu))   # value A
            m_item_.append(tmp[2].cuda(opts.gpu))   # value B
            m_items.append(m_item_)
        del m_item, tmp, m_item_

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        for it, (images_a, images_b, masks_a, masks_b) in enumerate(train_loader):
            if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
                continue

            # input data
            images_a = images_a.cuda(opts.gpu).detach()
            images_b = images_b.cuda(opts.gpu).detach()
            masks_a = masks_a.cuda(opts.gpu).detach()
            masks_b = masks_b.cuda(opts.gpu).detach()

            # update model
            if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                model.update_D_content(images_a, images_b)
                continue
            else:
                m_items = model.update_D(images_a, images_b, masks_a, masks_b, m_items)
                loss_G, loss_M = model.update_EG()

            if (total_it) % 500 == 0:
                saver.write_mid(ep, total_it, model, m_items)

            print('total_it: %d (ep %d, it %d), loss_G %04f loss_M %04f' % (total_it, ep, it, loss_G, loss_M))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, model, m_items)
                break

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        # save result image
        saver.write_img(ep, model)

        # Save network weights and memory
        saver.write_model(ep, total_it, model, m_items)
    return

if __name__ == '__main__':
    main()
