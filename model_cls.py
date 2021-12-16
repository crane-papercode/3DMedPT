import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from model_util import knn_point, index_points
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils as pt


class MGR(nn.Module):
    def __init__(self, C_in, C_out, dim_k=32, heads=8):
        super().__init__()
        self.heads = heads
        self.k = dim_k

        assert (C_out % heads) == 0, 'values dimension must be integer'
        dim_v = C_out // heads

        self.conv_q = nn.Conv1d(C_in, dim_k * heads, 1, bias=False)
        self.conv_k = nn.Conv1d(C_in, dim_k, 1, bias=False)
        self.conv_v = nn.Conv1d(C_in, dim_v, 1, bias=False)

        self.norm_q = nn.BatchNorm1d(dim_k * heads)
        self.norm_v = nn.BatchNorm1d(dim_v)

        self.blocker = nn.BatchNorm1d(C_out)
        self.skip = nn.Conv1d(C_out, C_out, 1)

        # multi-dimensional adjacency matrix
        self.A = nn.Parameter(torch.randn(dim_v, dim_v, dim_k), requires_grad=True)

    def forward(self, x):
        '''
        :param x: [B, C_in, N]
        :return: out: [B, C_out, N]
        '''
        query = self.conv_q(x)  # [B, head * C_k, N]
        key = self.conv_k(x)  # [B, C_k, N]
        value = self.conv_v(x)  # [B, C_v, N]

        # normalization
        query = self.norm_q(query)
        value = self.norm_v(value)
        key = key.softmax(dim=-1)

        query = rearrange(query, 'b (h k) n -> b h k n', h=self.heads)  # [B, head, C_k, N]
        k_v_attn = einsum('b k n, b v n -> b k v', key, value)  # [B, C_k, C_v]
        Yc = einsum('b h k n, b k v -> b n h v', query, k_v_attn)  # [B, N, head, C_v]
        G = einsum('b v n, w v k -> b n k w', value, self.A).contiguous()  # A*x: [B, N, C_k, C_v]
        value = rearrange(value, 'b v n -> b n (1) v').contiguous()
        G = F.relu(G + value)  # [B, N, C_k, C_v]
        Yp = einsum('b h k n, b n k v -> b n h v', query, G)  # [B, N, head, C_v]

        out = Yc + Yp
        out = rearrange(out, 'b n h v -> b n (h v)')
        out = rearrange(out, 'b n c -> b c n')
        out = self.blocker(self.skip(out))

        return F.relu(out + x)


class transformer(nn.Module):
    def __init__(self, C_in, C_out, n_samples=None, K=20, dim_k=32, heads=8, ch_raise=64):
        super().__init__()
        self.d = dim_k
        assert (C_out % heads) == 0, 'values dimension must be integer'
        dim_v = C_out // heads

        self.n_samples = n_samples
        self.K = K
        self.heads = heads

        C_in = C_in * 2 + dim_v
        self.mlp = nn.Sequential(
            nn.Conv2d(C_in, ch_raise, 1, bias=False),
            nn.BatchNorm2d(ch_raise),
            nn.ReLU(True),
            nn.Conv2d(ch_raise, ch_raise, 1, bias=False),
            nn.BatchNorm2d(ch_raise),
            nn.ReLU(True))

        self.mlp_v = nn.Conv1d(C_in, dim_v, 1, bias=False)
        self.mlp_k = nn.Conv1d(C_in, dim_k, 1, bias=False)
        self.mlp_q = nn.Conv1d(ch_raise, heads * dim_k, 1, bias=False)
        self.mlp_h = nn.Conv2d(3, dim_v, 1, bias=False)

        self.bn_value = nn.BatchNorm1d(dim_v)
        self.bn_query = nn.BatchNorm1d(heads * dim_k)

    def forward(self, xyz, feature):
        bs = xyz.shape[0]

        fps_idx = pt.furthest_point_sample(xyz.contiguous(), self.n_samples).long()  # [B, S]
        new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
        new_feature = index_points(feature, fps_idx).transpose(2, 1).contiguous()  # [B, C, S]

        knn_idx = knn_point(self.K, xyz, new_xyz)  # [B, S, K]
        neighbor_xyz = index_points(xyz, knn_idx)  # [B, S, K, 3]
        grouped_features = index_points(feature, knn_idx)  # [B, S, K, C]
        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()  # [B, C, S, K]
        grouped_points_norm = grouped_features - new_feature.unsqueeze(-1).contiguous()  # [B, C, S, K]
        # relative spatial coordinates
        relative_pos = neighbor_xyz - new_xyz.unsqueeze(-2).repeat(1, 1, self.K, 1)  # [B, S, K, 3]
        relative_pos = relative_pos.permute(0, 3, 1, 2).contiguous()  # [B, 3, S, K]

        pos_encoder = self.mlp_h(relative_pos)
        feature = torch.cat([grouped_points_norm,
                             new_feature.unsqueeze(-1).repeat(1, 1, 1, self.K),
                             pos_encoder], dim=1)  # [B, 2C_in + d, S, K]

        feature_q = self.mlp(feature).max(-1)[0]  # [B, C, S]
        query = F.relu(self.bn_query(self.mlp_q(feature_q)))  # [B, head * d, S]
        query = rearrange(query, 'b (h d) n -> b h d n', b=bs, h=self.heads, d=self.d)  # [B, head, d, S]

        feature = feature.permute(0, 2, 1, 3).contiguous()  # [B, S, 2C, K]
        feature = feature.view(bs * self.n_samples, -1, self.K)  # [B*S, 2C, K]
        value = self.bn_value(self.mlp_v(feature))  # [B*S, v, K]
        value = value.view(bs, self.n_samples, -1, self.K)  # [B, S, v, K]
        key = self.mlp_k(feature).softmax(dim=-1)  # [B*S, d, K]
        key = key.view(bs, self.n_samples, -1, self.K)  # [B, S, d, K]
        k_v_attn = einsum('b n d k, b n v k -> b d v n', key, value)  # [bs, d, v, N]
        out = einsum('b h d n, b d v n -> b h v n', query, k_v_attn.contiguous())  # [B, S, head, v]
        out = rearrange(out.contiguous(), 'b h v n -> b (h v) n')  # [B, C_out, S]

        return new_xyz, out


class Model(nn.Module):
    def __init__(self, args, trans_block, output_channels=2):
        super().__init__()
        self.use_norm = args.use_norm

        # transformer layer
        self.tf1 = trans_block(3, 128, n_samples=512, K=args.num_K[0], dim_k=args.dim_k, heads=args.head, ch_raise=64)
        self.tf2 = trans_block(128, 256, n_samples=128, K=args.num_K[1], dim_k=args.dim_k, heads=args.head, ch_raise=256)

        # multi-graph attention
        self.attn = MGR(256, 256, dim_k=args.dim_k, heads=args.head)

        self.conv_raise = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.ReLU(True))

        self.cls = nn.Sequential(
            nn.Linear(args.emb_dims, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, output_channels))

    def forward(self, x):
        # input x: [B, N, 3+3]
        xyz = x[..., :3]
        if not self.use_norm:
            feature = xyz
        else:
            assert x.size()[-1] == 6
            feature = x[..., 3:]

        xyz1, feature1 = self.tf1(xyz, feature)
        feature1 = feature1.transpose(2, 1)
        _, feature2 = self.tf2(xyz1, feature1)

        feature3 = self.attn(feature2)
        out = self.conv_raise(feature3)
        out = self.cls(out.max(-1)[0])

        return out
