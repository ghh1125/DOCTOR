import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAdapter(nn.Module):
    def __init__(self, video_dim=512, audio_dim=768, latent_dim=512):
        super().__init__()
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, video_feat, audio_feat):
        video_proj = self.video_proj(video_feat)
        audio_proj = self.audio_proj(audio_feat).unsqueeze(1)
        return video_proj, audio_proj


class AdaptiveCapsule(nn.Module):
    """胶囊网络结构"""
    def __init__(self, in_caps, out_caps=16, in_dim=512, out_dim=32, num_routing=3):
        super().__init__()
        self.num_routing = num_routing
        self.W = nn.Parameter(torch.randn(1, in_caps, out_caps, out_dim, in_dim))
        self.route_bias = nn.Parameter(torch.zeros(1, 1, out_caps, 1, 1))

    def forward(self, x):
        B = x.size(0)
        x = x.unsqueeze(2).unsqueeze(-1)
        prior = (self.W + self.route_bias).repeat(B, 1, 1, 1, 1)
        u_hat = torch.matmul(prior, x).squeeze(-1)

        logits = torch.zeros(B, u_hat.size(1), u_hat.size(2), device=x.device)
        for i in range(self.num_routing):
            attn = F.softmax(logits, dim=2)
            s = (attn.unsqueeze(-1) * u_hat).sum(dim=1)
            v = self.squash(s)
            if i < self.num_routing - 1:
                delta = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                logits = logits + delta
        return v

    def squash(self, s):
        norm = torch.norm(s, dim=-1, keepdim=True)
        return (norm / (1 + norm ** 2)) * s


class LocalAwareDiffusion(nn.Module):
    """双向扩散模块"""
    def __init__(self, latent_dim=512, timesteps=500):
        super().__init__()
        # 时间嵌入
        self.temb = nn.Embedding(timesteps, latent_dim)

        # 双向噪声预测器
        self.v2a_predictor = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),  # 输入通道修正为64
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1)
        )
        self.a2v_predictor = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),  # 输入通道修正为64
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1)
        )

        # 局部门控
        self.local_gate = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Sigmoid()
        )

        # 噪声调度
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x0, t, noise):
        sqrt_alpha = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def forward(self, v_caps, a_caps):
        B, num_caps, feat_dim = v_caps.size()  # [B,16,32]
        device = v_caps.device

        # 时间条件
        t = torch.randint(0, len(self.betas), (B,), device=device)
        t_emb = self.temb(t)  # [B,512]

        # 维度转换适配卷积
        v_trans = v_caps.transpose(1, 2)  # [B,32,16]
        a_trans = a_caps.transpose(1, 2)  # [B,32,16]

        # 局部门控噪声
        gate = self.local_gate(t_emb).view(B, 32, 1)  # [B,32,1]
        noise_v = torch.randn_like(v_trans) * gate
        noise_a = torch.randn_like(a_trans) * gate

        # 前向扩散
        noisy_v = self.q_sample(v_trans, t, noise_v)
        noisy_a = self.q_sample(a_trans, t, noise_a)

        # 双向交互
        v2a_feat = torch.cat([noisy_v, a_trans], dim=1)  # [B,64,16]
        a2v_feat = torch.cat([noisy_a, v_trans], dim=1)

        # 噪声预测
        pred_v = self.v2a_predictor(v2a_feat).transpose(1, 2)  # [B,16,32]
        pred_a = self.a2v_predictor(a2v_feat).transpose(1, 2)

        return pred_v, pred_a


class FusionClassifier(nn.Module):
    """分类器结构"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=32, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, video, audio):
        attn_out, _ = self.cross_attn(video, audio, audio)
        pooled = torch.cat([attn_out.mean(1), attn_out.max(1).values], dim=1)
        return self.classifier(pooled)


class Diffusion(nn.Module):
    """最终整合模型"""

    def __init__(self, video_dim=512, audio_dim=768, latent_dim=512,
                 video_seq=83, audio_seq=1, timesteps=500, num_classes=2):
        super().__init__()
        # 特征适配
        self.feature_adapter = FeatureAdapter(video_dim, audio_dim, latent_dim)

        # 胶囊网络
        self.video_caps = AdaptiveCapsule(video_seq, 16, latent_dim, 32)
        self.audio_caps = AdaptiveCapsule(audio_seq, 16, latent_dim, 32)

        # 双向扩散
        self.diffusion = LocalAwareDiffusion(latent_dim, timesteps)

        # 路由协同
        self.route_adaptor = nn.Parameter(torch.eye(32), requires_grad=True)

        # 分类器
        self.classifier = FusionClassifier(num_classes)

    def forward(self, **kwargs):
        # 原始特征
        video = kwargs['raw_visual_frames']  # [B,83,512]
        audio = kwargs['raw_audio_emo']  # [B,768]

        # 特征处理
        v_feat, a_feat = self.feature_adapter(video, audio)
        v_caps = self.video_caps(v_feat)  # [B,16,32]
        a_caps = self.audio_caps(a_feat)

        # 路由协同
        v_caps = torch.einsum('bij,jk->bik', v_caps, self.route_adaptor)
        a_caps = torch.einsum('bij,jk->bik', a_caps, self.route_adaptor)

        # 双向扩散
        pred_v, pred_a = self.diffusion(v_caps, a_caps)

        # 残差连接
        v_final = v_caps + pred_v
        a_final = a_caps + pred_a

        # 分类
        return self.classifier(v_final, a_final)