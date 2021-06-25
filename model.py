import networks
import torch
import torch.nn as nn
import memory

class MGUIT(nn.Module):
    def __init__(self, opts):
        super(MGUIT, self).__init__()

        # parameters
        lr = 0.0001
        lr_dcontent = lr / 2.5

        # discriminators
        self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.disContent = networks.Dis_content()

        # encoders
        self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b, opts.kdim)
        self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, norm_layer=nn.InstanceNorm2d, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
        
        # memory
        self.memory = memory.Memory(opts.msize, opts.kdim, opts.vdim)
        
        # generator
        self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, opts.vdim)

        # optimizers
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()

    def initialize(self):
        self.disA.apply(networks.gaussian_weights_init)
        self.disB.apply(networks.gaussian_weights_init)
        self.disContent.apply(networks.gaussian_weights_init)
        self.gen.apply(networks.gaussian_weights_init)
        self.enc_c.apply(networks.gaussian_weights_init)
        self.enc_a.apply(networks.gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
        self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
        self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
        self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
        self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)
        self.disContent.cuda(self.gpu)
        self.enc_c.cuda(self.gpu)
        self.enc_a.cuda(self.gpu)
        self.gen.cuda(self.gpu)

    def forward(self):
        # get value, key
        self.cont_a, self.cont_b = self.enc_c.forward(self.input_A, self.input_B)   # content
        self.sty_a, self.sty_b = self.enc_a.forward(self.input_A, self.input_B)     # attribute

        # memory read & update
        updated_memory, sty_aa, sty_ab, sty_ba, sty_bb, rand_aa, rand_ab, rand_ba, rand_bb, key_loss, value_loss \
            = self.memory.forward(self.cont_a, self.sty_a, self.cont_b, self.sty_b, self.masks_A, self.masks_B, self.memory_current)

        # first cross translation
        input_content_forA = torch.cat((self.cont_b, self.cont_a, self.cont_b, self.cont_a), 0)
        input_content_forB = torch.cat((self.cont_a, self.cont_b, self.cont_a, self.cont_b), 0)
        input_attr_forA = torch.cat((sty_ba, sty_aa, rand_ba.detach(), rand_aa.detach()), 0)
        input_attr_forB = torch.cat((sty_ab, sty_bb, rand_ab.detach(), rand_bb.detach()), 0)

        output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
        output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)

        self.fake_BA, self.fake_AA, self.rand_BA, self.rand_AA = torch.split(output_fakeA, self.cont_a.size(0), dim=0)
        self.fake_AB, self.fake_BB, self.rand_AB, self.rand_BB = torch.split(output_fakeB, self.cont_a.size(0), dim=0)

        return updated_memory, key_loss, value_loss

    def forward_content(self):
        self.cont_a, self.cont_b = self.enc_c.forward(self.input_A, self.input_B)         # content

    def update_D_content(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward_content()
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.cont_a, self.cont_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def update_D(self, image_a, image_b, masks_a, masks_b, memory):
        self.input_A = image_a
        self.input_B = image_b
        self.masks_A = masks_a
        self.masks_B = masks_b
        self.memory_current = memory

        self.updated_memory, self.key_loss, self.value_loss = self.forward()

        # update disA
        self.disA_opt.zero_grad()
        real = torch.cat((self.input_A, self.input_A), 0)
        fake = torch.cat((self.fake_BA, self.rand_BA), 0)
        loss_D1_A = self.backward_D(self.disA, real, fake)
        self.disA_loss = loss_D1_A.item()
        self.disA_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        real = torch.cat((self.input_B, self.input_B), 0)
        fake = torch.cat((self.fake_AB, self.rand_AB), 0)
        loss_D1_B = self.backward_D(self.disB, real, fake)
        self.disB_loss = loss_D1_B.item()
        self.disB_opt.step()

        # update disContent
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.cont_a, self.cont_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()
        return self.updated_memory

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def backward_contentD(self, imageA, imageB):
        pred_fake = self.disContent.forward(imageA.detach())
        pred_real = self.disContent.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def update_EG(self):
        # update G, Ec, Ea
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG()
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()
        return self.G_loss, self.M_loss

    def backward_EG(self):
        cont_ba, cont_ab = self.enc_c.forward(self.fake_BA, self.fake_AB)

        # second cross translation
        sty_aba, sty_bab = self.memory.forward_second(cont_ab, cont_ba, self.masks_A, self.masks_B, self.updated_memory)
        self.fake_ABA = self.gen.forward_a(cont_ab, sty_aba)
        self.fake_BAB = self.gen.forward_b(cont_ba, sty_bab)

        # content Ladv for generator
        loss_G_GAN_Acontent = self.backward_G_GAN_content(self.cont_a)
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.cont_b)

        # Ladv for generator
        loss_G_GAN_A = self.backward_G_GAN(torch.cat((self.fake_BA, self.rand_BA), 0), self.disA)
        loss_G_GAN_B = self.backward_G_GAN(torch.cat((self.fake_AB, self.rand_AB), 0), self.disB)

        # KL loss - z_a
        loss_kl_za_a = self._l2_regularize(self.sty_a) * 0.01
        loss_kl_za_b = self._l2_regularize(self.sty_b) * 0.01

        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(self.cont_a) * 0.01
        loss_kl_zc_b = self._l2_regularize(self.cont_b) * 0.01

        # key consistency loss
        loss_cont_A = self.criterionL1(self.cont_a, cont_ab)
        loss_cont_B = self.criterionL1(self.cont_b, cont_ba)

        # cross cycle consistency loss
        loss_G_L1_A = self.criterionL1(self.fake_ABA, self.input_A) * 50
        loss_G_L1_B = self.criterionL1(self.fake_BAB, self.input_B) * 50
        loss_G_L1_AA = self.criterionL1(self.fake_AA, self.input_A) * 10
        loss_G_L1_BB = self.criterionL1(self.fake_BB, self.input_B) * 10

        loss_G = loss_G_GAN_A + loss_G_GAN_B + \
                 loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
                 loss_G_L1_AA + loss_G_L1_BB + \
                 loss_G_L1_A + loss_G_L1_B + \
                 loss_cont_A + loss_cont_B + \
                 loss_kl_zc_a + loss_kl_zc_b + \
                 loss_kl_za_a + loss_kl_za_b
        loss_M = self.key_loss + self.value_loss

        loss_MG = loss_G + loss_M
        loss_MG.backward()

        self.gan_loss_a = loss_G_GAN_A.item()
        self.gan_loss_b = loss_G_GAN_B.item()
        self.gan_loss_acontent = loss_G_GAN_Acontent.item()
        self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
        self.kl_loss_za_a = loss_kl_za_a.item()
        self.kl_loss_za_b = loss_kl_za_b.item()
        self.kl_loss_zc_a = loss_kl_zc_a.item()
        self.kl_loss_zc_b = loss_kl_zc_b.item()
        self.l1_recon_A_loss = loss_G_L1_A.item()
        self.l1_recon_B_loss = loss_G_L1_B.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()
        self.l1_cont_A_loss = loss_cont_A.item()
        self.l1_cont_B_loss = loss_cont_B.item()
        self.G_loss = loss_G.item()

        if loss_M == 0:
            self.M_loss = loss_M
        else:
            self.M_loss = loss_M.item()

    def backward_G_GAN_content(self, data):
        outs = self.disContent.forward(data)
        for out in outs:
            outputs_fake = nn.functional.sigmoid(out)
            all_half = 0.5*torch.ones((outputs_fake.size(0))).cuda(self.gpu)
            ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
        return ad_loss

    def backward_G_GAN(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = nn.functional.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        return loss_G

    def backward_G_alone(self):
        # Ladv for generator
        loss_G_GAN2_A = self.backward_G_GAN(self.rand_BA, self.disA2)
        loss_G_GAN2_B = self.backward_G_GAN(self.rand_AB, self.disB2)

        loss_z_L1 = loss_G_GAN2_A + loss_G_GAN2_B
        loss_z_L1.backward(retain_graph=True)
        self.gan2_loss_a = loss_G_GAN2_A.item()
        self.gan2_loss_b = loss_G_GAN2_B.item()

    def update_lr(self):
        self.disA_sch.step()
        self.disB_sch.step()
        self.disContent_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def resume(self, model_dir, gpu, train=True):
        checkpoint = torch.load(model_dir, map_location='cuda:%d' % gpu)
        # weight
        if train:
            self.disA.load_state_dict(checkpoint['disA'])
            self.disB.load_state_dict(checkpoint['disB'])
            self.disContent.load_state_dict(checkpoint['disContent'])
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_a.load_state_dict(checkpoint['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint['disA_opt'])
            self.disB_opt.load_state_dict(checkpoint['disB_opt'])
            self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
            self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'disA': self.disA.state_dict(),
            'disB': self.disB.state_dict(),
            'disContent': self.disContent.state_dict(),
            'enc_c': self.enc_c.state_dict(),
            'enc_a': self.enc_a.state_dict(),
            'gen': self.gen.state_dict(),
            'disA_opt': self.disA_opt.state_dict(),
            'disB_opt': self.disB_opt.state_dict(),
            'disContent_opt': self.disContent_opt.state_dict(),
            'enc_c_opt': self.enc_c_opt.state_dict(),
            'enc_a_opt': self.enc_a_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_a = self.normalize_image(self.input_A).detach()
        images_b = self.normalize_image(self.input_B).detach()
        images_a1 = self.normalize_image(self.fake_AB).detach()
        images_a2 = self.normalize_image(self.rand_AB).detach()
        images_a3 = self.normalize_image(self.fake_AA).detach()
        images_a4 = self.normalize_image(self.rand_AA).detach()
        images_a5 = self.normalize_image(self.fake_ABA).detach()
        images_b1 = self.normalize_image(self.fake_BA).detach()
        images_b2 = self.normalize_image(self.rand_BA).detach()
        images_b3 = self.normalize_image(self.fake_BB).detach()
        images_b4 = self.normalize_image(self.rand_BB).detach()
        images_b5 = self.normalize_image(self.fake_BAB).detach()
        row1 = torch.cat((images_a[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_a3[0:1, ::], images_a4[0:1, ::], images_a5[0:1, ::]),3)
        row2 = torch.cat((images_b[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_b3[0:1, ::], images_b4[0:1, ::], images_b5[0:1, ::]),3)
        return torch.cat((row1,row2),2)

    def normalize_image(self, x):
        return x[:,0:3,:,:]
