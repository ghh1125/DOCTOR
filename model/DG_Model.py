import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import copy
import tqdm
import numpy as np

import torch.autograd as autograd
from typing import List

class DG_Model(nn.Module):
    def __init__(self, video_feature_dim=512, audio_feature_dim=768, num_classes=9,
                 hidden_dim=2048, out_dim=128, trans_hidden_num=2048,
                 use_video=True, use_audio=True,
                 CM_mixup=True, mix_alpha=0.1, contrast=True, temp=0.1,
                 distill=True, distill_coef=1.0, SMA=True, sma_start_step=10):
        super(DG_Model, self).__init__()

        self.distill_check = True

        self.v_dim = video_feature_dim // 2
        self.a_dim = audio_feature_dim // 2
        self.use_video = use_video
        self.use_audio = use_audio
        self.CM_mixup = CM_mixup
        self.contrast = contrast
        self.distill = distill
        self.distill_coef = distill_coef
        self.SMA = SMA
        self.sma_start_step = sma_start_step
        self.alpha_contrast = 3.0
        self.alpha_trans = 0.1

        self.explore_loss_coeff = 0.7

        if self.use_video:
            self.v_proj = ProjectHead(input_dim=self.v_dim, hidden_dim=hidden_dim, out_dim=out_dim)
            if self.CM_mixup:
                self.v_proj_m = ProjectHead(input_dim=video_feature_dim, hidden_dim=hidden_dim, out_dim=out_dim)
                self.v_proj_m_cls = nn.Linear(128, num_classes)

        if self.use_audio:
            self.a_proj = ProjectHead(input_dim=self.a_dim, hidden_dim=hidden_dim, out_dim=out_dim)
            if self.CM_mixup:
                self.a_proj_m = ProjectHead(input_dim=audio_feature_dim, hidden_dim=hidden_dim, out_dim=out_dim)
                self.a_proj_m_cls = nn.Linear(128, num_classes)

        input_dim = 0
        if self.use_video:
            input_dim += video_feature_dim
        if self.use_audio:
            input_dim += audio_feature_dim

        self.mlp_cls = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.LayerNorm(512),
            nn.Linear(512, 2)
        )

        if self.use_video and self.use_audio:
            self.mlp_v2a = EncoderTrans(input_dim=video_feature_dim, hidden=trans_hidden_num, out_dim=audio_feature_dim)
            self.mlp_a2v = EncoderTrans(input_dim=audio_feature_dim, hidden=trans_hidden_num, out_dim=video_feature_dim)

        if self.contrast:
            self.criterion_contrast = SupConLoss(temperature=temp)

        self.criterion = nn.CrossEntropyLoss()
        self.mix_alpha = mix_alpha

        if self.SMA:
            self.mlp_cls_sma = copy.deepcopy(self.mlp_cls)
            if self.use_video:
                self.v_proj_sma = copy.deepcopy(self.v_proj_m)
                self.v_proj_m_cls_sma = copy.deepcopy(self.v_proj_m_cls)
            if self.use_audio:
                self.a_proj_sma = copy.deepcopy(self.a_proj_m)
                self.a_proj_m_cls_sma = copy.deepcopy(self.a_proj_m_cls)

    def forward(self, global_step=0, sma_count=0, **kwargs):
        v_emd = kwargs['raw_visual_frames']
        a_emd = kwargs['raw_audio_emo']
        labels = kwargs['label']
        loss = 0
        v_emd = torch.mean(v_emd, dim=1)


        if self.CM_mixup and self.SMA:
            if self.use_video and self.use_audio:
                with torch.no_grad():
                    # 修改原本获取特征的逻辑，直接使用输入的特征
                    v_emd_sma = copy.deepcopy(v_emd)
                    a_emd_sma = copy.deepcopy(a_emd)

                    v_mix_proj_sma = self.v_proj_sma(v_emd_sma)
                    a_mix_proj_sma = self.a_proj_sma(a_emd_sma)

                    v_emd_mix_sma, a_emd_mix_sma, _ = mix_feature(v_mix_proj_sma, a_mix_proj_sma, self.mix_alpha,
                                                                      self.mix_alpha)  # teacher features

                    # v_emd_mix_sma = v_mix_proj_sma
                    # audio_emd_mix_sma = a_mix_proj_sma
            v_mix_proj = self.v_proj_m(v_emd)
            a_mix_proj = self.a_proj_m(a_emd)
            if self.contrast:
                emd_proj = torch.stack([v_mix_proj, a_mix_proj], dim=1)
                # sup contrast
                loss_mix_contrast = self.criterion_contrast(emd_proj, labels)
                loss += self.alpha_contrast * loss_mix_contrast

            if self.distill and global_step > self.sma_start_step:  # 現在是SMA的模型向online模型蒸餾，能否將online模型變得比SMA模型更好，如果不能能否改為SMA的模型自蒸餾
                if self.distill_check:
                    print(f"global_step is {global_step}, distill start!")
                    self.distill_check = False
                v_emd_tea = v_emd_mix_sma / torch.norm(v_emd_mix_sma, dim=1, keepdim=True)
                v_emd_stu = v_mix_proj / torch.norm(v_mix_proj, dim=1, keepdim=True)
                a_emd_tea = a_emd_mix_sma / torch.norm(a_emd_mix_sma, dim=1, keepdim=True)
                a_emd_stu = a_mix_proj / torch.norm(a_mix_proj, dim=1, keepdim=True)
                loss += self.distill_coef * (torch.mean(torch.norm(v_emd_tea.detach() - v_emd_stu, dim=1)) + 1/2 *torch.mean(
                    torch.norm(a_emd_tea.detach() - a_emd_stu, dim=1)))

            # loss += self.mix_coef * (criterion(predict1, labels) + 1./2 * criterion(a_predict, labels))

        feat = torch.cat((v_emd, a_emd), dim=1)

        predict = self.mlp_cls(feat)
        loss += self.criterion(predict, labels)

        if self.use_video and self.use_audio:
            a_emd_t = self.mlp_v2a(v_emd)
            v_emd_t = self.mlp_a2v(a_emd)
            a_emd_t = a_emd_t / torch.norm(a_emd_t, dim=1, keepdim=True)
            v_emd_t = v_emd_t / torch.norm(v_emd_t, dim=1, keepdim=True)

            v2a_loss = torch.mean(torch.norm(a_emd_t - a_emd / torch.norm(a_emd, dim=1, keepdim=True), dim=1))
            a2v_loss = torch.mean(torch.norm(v_emd_t - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
            loss = loss + self.alpha_trans * (v2a_loss + a2v_loss) / 2

        # Supervised Contrastive Learning
        if self.use_video:
            v_emd_proj = self.v_proj(v_emd[:, :self.v_dim])
        if self.use_audio:
            a_emd_proj = self.a_proj(a_emd[:, :self.a_dim])
        if self.use_video and self.use_audio:
            emd_proj = torch.stack([v_emd_proj, a_emd_proj], dim=1)

        loss_contrast = self.criterion_contrast(emd_proj, labels)
        loss = loss + self.alpha_contrast * loss_contrast

        # Feature Splitting with Distance
        loss_e = 0
        num_loss = 0
        if self.use_video:
            loss_e = loss_e - F.mse_loss(v_emd[:, :self.v_dim], v_emd[:, self.v_dim:])
            num_loss = num_loss + 1
        if self.use_audio:
            loss_e = loss_e - F.mse_loss(a_emd[:, :self.a_dim], a_emd[:, self.a_dim:])
            num_loss = num_loss + 1

        loss = loss + self.explore_loss_coeff * loss_e / num_loss

        if self.training:
            new_v_proj_m_dict = {}
            new_a_proj_m_dict = {}
            new_v_proj_m_cls_dict = {}
            new_a_proj_m_cls_dict = {}
            new_cls_dict = {}

            if global_step > self.sma_start_step:
                sma_count += 1

                if self.use_video:
                    for (name, param_q), (_, param_k) in zip(self.v_proj_m.state_dict().items(),
                                                             self.v_proj_sma.state_dict().items()):
                        new_v_proj_m_dict[name] = (
                                (param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                1. + sma_count))
                    for (name, param_q), (_, param_k) in zip(self.v_proj_m_cls.state_dict().items(),
                                                             self.v_proj_m_cls_sma.state_dict().items()):
                        new_v_proj_m_cls_dict[name] = (
                                (param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                1. + sma_count))
                if self.use_audio:
                    for (name, param_q), (_, param_k) in zip(self.a_proj_m.state_dict().items(),
                                                             self.a_proj_sma.state_dict().items()):
                        new_a_proj_m_dict[name] = (
                                (param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                1. + sma_count))
                    for (name, param_q), (_, param_k) in zip(self.a_proj_m_cls.state_dict().items(),
                                                             self.a_proj_m_cls_sma.state_dict().items()):
                        new_a_proj_m_cls_dict[name] = (
                                (param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                                1. + sma_count))
                for (name, param_q), (_, param_k) in zip(self.mlp_cls.state_dict().items(),
                                                         self.mlp_cls_sma.state_dict().items()):
                    new_cls_dict[name] = ((param_k.data.detach().clone() * sma_count + param_q.data.detach().clone()) / (
                            1. + sma_count))

                if self.use_video:
                    self.v_proj_sma.load_state_dict(new_v_proj_m_dict)
                    self.v_proj_m_cls_sma.load_state_dict(new_v_proj_m_cls_dict)
                if self.use_audio:
                    self.a_proj_sma.load_state_dict(new_a_proj_m_dict)
                    self.a_proj_m_cls_sma.load_state_dict(new_a_proj_m_cls_dict)
                self.mlp_cls_sma.load_state_dict(new_cls_dict)

        return predict, loss


def norm(tensor_list: List[torch.tensor], p=2):
    """Compute p-norm for tensor list"""
    return torch.cat([x.flatten() for x in tensor_list]).norm(p)


def mix_feature(m1_f, m2_f, alpha1=None, alpha2=None):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    bsz = m1_f.shape[0]
    lam1 = np.zeros(bsz)
    lam2 = np.zeros(bsz)
    if alpha1 > 0:
        i = 0
        while i < bsz:
            lam = np.random.beta(alpha1, alpha1)
            if lam > 0.5:
                lam1[i] = lam
                i += 1
            else:
                continue
        lam1 = torch.tensor(lam1).cuda().float().unsqueeze(-1)
    else:
        lam1 = 1.
    if alpha2 > 0:
        i = 0
        while i < bsz:
            lam = np.random.beta(alpha2, alpha2)
            if lam < 0.5:
                lam2[i] = lam
                i += 1
            else:
                continue
        lam2 = torch.tensor(lam2).cuda().float().unsqueeze(-1)
    else:
        lam2 = 0.

    f1 = copy.deepcopy(m1_f)
    f2 = copy.deepcopy(m2_f)
    for i in range(lam1.shape[0]):
        f1[i] = lam1[i] * m1_f[i] + (1. - lam1[i]) * m2_f[i]
        f2[i] = lam2[i] * m1_f[i] + (1. - lam2[i]) * m2_f[i]

    return f1, f2, [lam1, lam2]


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        return self.enc_net(feat)


class EncoderTrans(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat


class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
