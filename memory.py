import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class Memory(nn.Module):
    def __init__(self, memory_size, kdim, vdim):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.kdim = kdim
        self.vdim = vdim
        self.norm = nn.InstanceNorm1d(vdim)

    def forward(self, conts_a, stys_a, conts_b, stys_b, masks_a, masks_b, memorys):
        batch_size, _, h, w = conts_a.size()
        sty_ab = torch.empty(batch_size * h * w, self.vdim).to(stys_b.device)
        sty_ba = torch.empty(batch_size * h * w, self.vdim).to(stys_b.device)
        sty_aa = torch.empty(batch_size * h * w, self.vdim).to(stys_b.device)
        sty_bb = torch.empty(batch_size * h * w, self.vdim).to(stys_b.device)
        rand_ab = torch.empty(batch_size * h * w, self.vdim).to(stys_b.device)
        rand_ba = torch.empty(batch_size * h * w, self.vdim).to(stys_b.device)
        rand_aa = torch.empty(batch_size * h * w, self.vdim).to(stys_b.device)
        rand_bb = torch.empty(batch_size * h * w, self.vdim).to(stys_b.device)

        gathers_a = torch.empty(batch_size * h * w, 1).to(stys_b.device)
        gathers_b = torch.empty(batch_size * h * w, 1).to(stys_b.device)

        key_loss = 0
        value_loss = 0
        updated_memorys = []
        for i in range(0, len(memorys)):
            memory = memorys[i]
            msize = memory[0].size()

            rand1 = torch.randn((msize[0], self.vdim), dtype=torch.float).to(stys_b.device)
            rand2 = torch.randn((msize[0], self.vdim), dtype=torch.float).to(stys_b.device)

            mask_a = masks_a[:,i,:,:].unsqueeze(1)
            mask_b = masks_b[:,i,:,:].unsqueeze(1)
            cont_a = torch.mul(conts_a, mask_a)
            cont_a = cont_a.permute(0, 2, 3, 1)  # b X h X w X d
            sty_a = torch.mul(stys_a, mask_a)
            sty_a = sty_a.permute(0, 2, 3, 1)  # b X h X w X d
            cont_b = torch.mul(conts_b, mask_b)
            cont_b = cont_b.permute(0, 2, 3, 1)  # b X h X w X d
            sty_b = torch.mul(stys_b, mask_b)
            sty_b = sty_b.permute(0, 2, 3, 1)  # b X h X w X d

            idx_a = mask_a.reshape(batch_size * h * w).nonzero()
            idx_b = mask_b.reshape(batch_size * h * w).nonzero()

            if torch.sum(mask_b) * torch.sum(mask_a) == 0:
                updated_memorys.append(memory)
                # read only
                if torch.sum(mask_a) != 0:
                    sty_aa_, sty_ab_, rand_aa_, rand_ab_, score_cont_a = self.read(cont_a, idx_a, memory, rand1, True)
                    sty_aa[idx_a[:, 0], :] = sty_aa_
                    sty_ab[idx_a[:, 0], :] = sty_ab_
                    rand_aa[idx_a[:, 0], :] = rand_aa_
                    rand_ab[idx_a[:, 0], :] = rand_ab_
                    _, gather_a = torch.topk(score_cont_a, 2, dim=1)
                    gathers_a[idx_a[:, 0], :] = torch.unsqueeze(gather_a[:, 0].detach() + sum(self.memory_size[:i]), dim=1).to(torch.float)
                if torch.sum(mask_b) != 0:
                    sty_ba_, sty_bb_, rand_ba_, rand_bb_, score_cont_b = self.read(cont_b, idx_b, memory, rand2, True)
                    sty_ba[idx_b[:, 0], :] = sty_ba_
                    sty_bb[idx_b[:, 0], :] = sty_bb_
                    rand_ba[idx_b[:, 0], :] = rand_ba_
                    rand_bb[idx_b[:, 0], :] = rand_bb_
                    _, gather_b = torch.topk(score_cont_b, 2, dim=1)
                    gathers_b[idx_b[:, 0], :] = torch.unsqueeze(gather_b[:, 0].detach() + sum(self.memory_size[:i]), dim=1).to(torch.float)
                continue

            # update
            updated_memory = self.update(cont_a, sty_a, cont_b, sty_b, idx_a, idx_b, memory)
            updated_memorys.append(updated_memory)

            # read
            sty_aa_, sty_ab_, rand_aa_, rand_ab_, _ = self.read(cont_a, idx_a, updated_memory, rand1, True)
            sty_ba_, sty_bb_, rand_ba_, rand_bb_, _ = self.read(cont_b, idx_b, updated_memory, rand2, True)

            sty_aa[idx_a[:, 0], :] = sty_aa_
            sty_ab[idx_a[:, 0], :] = sty_ab_
            sty_ba[idx_b[:, 0], :] = sty_ba_
            sty_bb[idx_b[:, 0], :] = sty_bb_
            rand_aa[idx_a[:, 0], :] = rand_aa_
            rand_ab[idx_a[:, 0], :] = rand_ab_
            rand_ba[idx_b[:, 0], :] = rand_ba_
            rand_bb[idx_b[:, 0], :] = rand_bb_

            # loss
            k_loss, v_loss, gather_a, gather_b = self.gather_loss(cont_a, cont_b, sty_a, sty_b, idx_a, idx_b, memory)
            gathers_a[idx_a[:, 0], :] = torch.unsqueeze(gather_a + sum(self.memory_size[:i]), dim=1).to(torch.float)
            gathers_b[idx_b[:, 0], :] = torch.unsqueeze(gather_b + sum(self.memory_size[:i]), dim=1).to(torch.float)
            key_loss += k_loss
            value_loss += v_loss

        k_loss = self.gather_total_loss(conts_a.permute(0, 2, 3, 1), conts_b.permute(0, 2, 3, 1), gathers_a, gathers_b, memorys)
        key_loss += k_loss

        sty_aa = sty_aa.view(batch_size, h, w, self.vdim)
        sty_aa = sty_aa.permute(0, 3, 1, 2)
        sty_ab = sty_ab.view(batch_size, h, w, self.vdim)
        sty_ab = sty_ab.permute(0, 3, 1, 2)
        sty_ba = sty_ba.view(batch_size, h, w, self.vdim)
        sty_ba = sty_ba.permute(0, 3, 1, 2)
        sty_bb = sty_bb.view(batch_size, h, w, self.vdim)
        sty_bb = sty_bb.permute(0, 3, 1, 2)

        rand_aa = rand_aa.view(batch_size, h, w, self.vdim)
        rand_aa = rand_aa.permute(0, 3, 1, 2)
        rand_ab = rand_ab.view(batch_size, h, w, self.vdim)
        rand_ab = rand_ab.permute(0, 3, 1, 2)
        rand_ba = rand_ba.view(batch_size, h, w, self.vdim)
        rand_ba = rand_ba.permute(0, 3, 1, 2)
        rand_bb = rand_bb.view(batch_size, h, w, self.vdim)
        rand_bb = rand_bb.permute(0, 3, 1, 2)

        return updated_memorys, sty_aa, sty_ab, sty_ba, sty_bb, rand_aa, rand_ab, rand_ba, rand_bb, key_loss, value_loss

    def forward_second(self, conts_a, conts_b, masks_a, masks_b, memorys):
        batch_size, _, h, w = conts_a.size()

        sty_aa = torch.empty(batch_size * h * w, self.vdim).to(conts_a.device)
        sty_bb = torch.empty(batch_size * h * w, self.vdim).to(conts_a.device)

        for i in range(0, len(memorys)):
            memory = memorys[i]
            mask_a = masks_a[:,i,:,:].unsqueeze(1)
            mask_b = masks_b[:,i,:,:].unsqueeze(1)
            cont_a = torch.mul(conts_a, mask_a)
            cont_b = torch.mul(conts_b, mask_b)
            cont_a = cont_a.permute(0, 2, 3, 1)  # b X h X w X d
            cont_b = cont_b.permute(0, 2, 3, 1)  # b X h X w X d
            idx_a = mask_a.reshape(batch_size * h * w).nonzero()
            idx_b = mask_b.reshape(batch_size * h * w).nonzero()

            if torch.sum(mask_a) != 0:
                sty_aa_, _ = self.read(cont_a, idx_a, memory)
                sty_aa[idx_a[:, 0], :] = sty_aa_
            if torch.sum(mask_b) != 0:
                _, sty_bb_ = self.read(cont_b, idx_b, memory)
                sty_bb[idx_b[:, 0], :] = sty_bb_

        sty_aa = sty_aa.view(batch_size, h, w, self.vdim)
        sty_aa = sty_aa.permute(0, 3, 1, 2)
        sty_bb = sty_bb.view(batch_size, h, w, self.vdim)
        sty_bb = sty_bb.permute(0, 3, 1, 2)
        return sty_aa, sty_bb

    def read(self, query, idx, mem, value=[], rand=False):
        k, vdim = mem[1].size()
        val1 = mem[1].contiguous().view(k, vdim)
        val2 = mem[2].contiguous().view(k, vdim)
        score_cont, score_mem = self.get_score(mem[0], query, idx)

        updated_value1 = torch.matmul(score_mem.detach(), val1)
        updated_value2 = torch.matmul(score_mem.detach(), val2)

        if rand==False:
            return updated_value1.detach(), updated_value2.detach()
        else:
            rand_value1 = torch.matmul(score_mem.detach(), value.contiguous())
            rand_value2 = torch.matmul(score_mem.detach(), value.contiguous())
            return updated_value1.detach(), updated_value2.detach(), rand_value1.detach(), rand_value2.detach(), score_cont

    def read_single(self, query, idx, key, value):
        k, vdim = value.size()
        value = value.contiguous().view(k, vdim)
        softmax_score_key, softmax_score_memory = self.get_score(key, query, idx)

        updated_value = torch.matmul(softmax_score_memory.detach(), value)  # (b X h X w) X d
        return updated_value.detach()

    def update(self, cont_a, sty_a, cont_b, sty_b, idx_a, idx_b, memory):
        batch_size, h, w, dims = cont_a.size()  # b X h X w X d
        _, h1, w1, dims1 = sty_a.size()  # b X h X w X d

        score_cont_a, score_memory_a = self.get_score(memory[0], cont_a, idx_a)
        score_cont_b, score_memory_b = self.get_score(memory[0], cont_b, idx_b)

        cont_a_reshape = cont_a.contiguous().view(batch_size * h * w, dims)
        sty_a_reshape = sty_a.contiguous().view(batch_size * h1 * w1, dims1)
        cont_b_reshape = cont_b.contiguous().view(batch_size * h * w, dims)
        sty_b_reshape = sty_b.contiguous().view(batch_size * h1 * w1, dims1)

        cont_a_reshape = cont_a_reshape[idx_a, :].squeeze(1)
        cont_b_reshape = cont_b_reshape[idx_b, :].squeeze(1)
        sty_a_reshape = sty_a_reshape[idx_a, :].squeeze(1)
        sty_b_reshape = sty_b_reshape[idx_b, :].squeeze(1)

        _, gathering_indices_a = torch.topk(score_memory_a, 1, dim=1)
        _, gathering_indices_b = torch.topk(score_memory_b, 1, dim=1)

        key_update_a = self.get_update_key(memory[0], gathering_indices_a, score_cont_a, cont_a_reshape)
        value_update_a = self.get_update_value(memory[1], gathering_indices_a, score_cont_a, sty_a_reshape)
        key_update_b = self.get_update_key(memory[0], gathering_indices_b, score_cont_b, cont_b_reshape)
        value_update_b = self.get_update_value(memory[2], gathering_indices_b, score_cont_b, sty_b_reshape)

        updated_key = F.normalize(0.25 * key_update_a + 0.25 * key_update_b + 0.5 * memory[0], dim=1)
        updated_sty_a = 0.5 * value_update_a.squeeze(1) + 0.5 * memory[1]
        updated_sty_b = 0.5 * value_update_b.squeeze(1) + 0.5 * memory[2]
        return updated_key.detach(), updated_sty_a.detach(), updated_sty_b.detach()

    def get_update_key(self, mem, max_indices, score, query):
        m, d = mem.size()
        query_update = torch.zeros((m, d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                q = score[idx, i] * query[idx].squeeze(1)
                query_update[i] = F.normalize(torch.sum(q, dim=0).unsqueeze(0), dim=1)
            else:
                query_update[i] = 0
        return query_update

    def get_update_value(self, mem, max_indices, score, query):
        m, d = mem.size()
        query_update = torch.zeros((m, d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                q = score[idx, i] * query[idx].squeeze(1)
                query_update[i] = torch.sum(q, dim=0).unsqueeze(0)
            else:
                query_update[i] = 0
        return self.norm(query_update.unsqueeze(1))

    def get_score(self, mem, query, mask):
        bs, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))  # b X h X w X m
        score = score.view(bs * h * w, m)  # (b X h X w) X m

        score_mask = score[mask, :].squeeze(1)
        score_query = F.softmax(score_mask, dim=0)
        score_memory = F.softmax(score_mask, dim=1)

        return score_query, score_memory

    def get_score_total(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))  # b X h X w X m
        score = score.view(bs * h * w, m)  # (b X h X w) X m
        score_query = F.softmax(score, dim=0)
        return score_query

    def gather_loss(self, cont_a, cont_b, sty_a, sty_b, idx_a, idx_b, mem):
        score_cont_a, _ = self.get_score(mem[0], cont_a, idx_a)
        score_cont_b, _ = self.get_score(mem[0], cont_b, idx_b)

        score_sty_a, _ = self.get_score(mem[1], sty_a, idx_a)
        score_sty_b, _ = self.get_score(mem[2], sty_b, idx_b)

        _, gathering_indices_a = torch.topk(score_cont_a, 2, dim=1)
        _, gathering_indices_b = torch.topk(score_cont_b, 2, dim=1)

        key_loss = 0.1 * torch.exp(F.nll_loss(score_cont_a, gathering_indices_a[:, 0].detach())) + 0.1 * torch.exp(F.nll_loss(score_cont_b, gathering_indices_b[:, 0].detach()))
        value_loss = 0.1 * torch.exp(F.nll_loss(score_sty_a, gathering_indices_a[:, 0].detach())) + 0.1 * torch.exp(F.nll_loss(score_sty_b, gathering_indices_b[:, 0].detach()))
        return key_loss, value_loss, gathering_indices_a[:, 0].detach(), gathering_indices_b[:, 0].detach()

    def gather_total_loss(self, cont_a, cont_b, gathers_a, gathers_b, memorys):
        mem0 = memorys[0][0]
        for i in range(1, len(memorys)):
            mem0 = torch.cat((mem0, memorys[i][0]), dim=0)

        score_cont_a = self.get_score_total(mem0, cont_a)
        score_cont_b = self.get_score_total(mem0, cont_b)

        total_loss = 0.1 * torch.exp(F.nll_loss(score_cont_a, gathers_a[:,0].to(torch.long))) + 0.1 * torch.exp(F.nll_loss(score_cont_b, gathers_b[:,0].to(torch.long)))
        return total_loss
