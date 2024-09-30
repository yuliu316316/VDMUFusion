from torch import nn
import numpy as np
from abc import abstractmethod

from .__init__ import time_embedding
from .__init__ import Downsample
from .__init__ import Upsample


# use GN for norm layer
def group_norm(channels):
    return nn.GroupNorm(32, channels)


#  time_embedding block
class TimeBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """

        """


class TimeSequential(nn.Sequential, TimeBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimeBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ******** Attention mudule ***********
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention_channel = self.sigmoid(avg_out + max_out)
        attention_spatial = self.conv(x)
        attention_spatial = self.sigmoid(attention_spatial)
        attention = attention_channel * attention_spatial
        return attention * x


class ResBlock(TimeBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout, add_time):
        super().__init__()
        self.add_time = add_time
        self.conv1 = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class NoisePred(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model_channels,
                 num_res_blocks,
                 dropout,
                 time_embed_dim_mult,
                 down_sample_mult,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.down_sample_mult = down_sample_mult

        # time embedding
        time_embed_dim = model_channels * time_embed_dim_mult
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        down_channels = [model_channels * i for i in down_sample_mult]
        up_channels = down_channels[::-1]

        downBlock_chanNum = [num_res_blocks + 1] * (len(down_sample_mult) - 1)
        downBlock_chanNum.append(num_res_blocks)  # [3, 3, 3, 2]
        upBlock_chanNum = downBlock_chanNum[::-1]
        self.downBlock_chanNum_cumsum = np.cumsum(downBlock_chanNum)
        self.upBlock_chanNum_cumsum = np.cumsum(upBlock_chanNum)[:-1]

        self.inBlock = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)

        # DownSample block
        self.downBlock = nn.ModuleList()
        self.attention_block = nn.ModuleList()
        self.attention_block_1 = nn.ModuleList()
        down_init_channel = model_channels
        for level, channel in enumerate(down_channels):
            # **************attention layer *******************
            attention_layer = AttentionBlock(channel)
            attention_layer_1 = AttentionBlock(channel)
            self.attention_block.append(attention_layer)
            self.attention_block_1.append(attention_layer_1)
            for _ in range(num_res_blocks):
                layer1 = ResBlock(in_channels=down_init_channel,
                                  out_channels=channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout,
                                  add_time=True)
                down_init_channel = channel
                self.downBlock.append(TimeSequential(layer1))

            if level != len(down_sample_mult) - 1:
                down_layer = Downsample(channels=channel)
                self.downBlock.append(TimeSequential(down_layer))

        # middle block
        self.middleBlock = nn.ModuleList()
        for _ in range(num_res_blocks):
            layer2 = ResBlock(in_channels=down_channels[-1],
                              out_channels=down_channels[-1],
                              time_channels=time_embed_dim,
                              dropout=dropout,
                              add_time=False)
            self.middleBlock.append(TimeSequential(layer2))

        # upsample block
        self.upBlock = nn.ModuleList()
        up_init_channel = down_channels[-1]
        for level, channel in enumerate(up_channels):
            if level == len(up_channels) - 1:
                out_channel = model_channels
            else:
                out_channel = channel // 2
            for _ in range(num_res_blocks):
                layer3 = ResBlock(in_channels=up_init_channel,
                                  out_channels=out_channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout,
                                  add_time=False)
                up_init_channel = out_channel
                self.upBlock.append(TimeSequential(layer3))
            if level > 0:
                up_layer = Upsample(channels=out_channel)
                self.upBlock.append(TimeSequential(up_layer))

        # upsample and fusion block
        self.fusionBlock = nn.ModuleList()
        up_init_channel = down_channels[-1]
        for level, channel in enumerate(up_channels):
            if level == len(up_channels) - 1:
                out_channel = model_channels
            else:
                out_channel = channel // 2
            for _ in range(num_res_blocks):
                layer4 = ResBlock(in_channels=up_init_channel,
                                  out_channels=out_channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout,
                                  add_time=False)
                up_init_channel = out_channel
                self.fusionBlock.append(TimeSequential(layer4))
            if level > 0:
                up_layer = Upsample(channels=out_channel)
                self.fusionBlock.append(TimeSequential(up_layer))

        # out block
        self.outBlock = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

        # fusion out block
        self.fusion_outBlock = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, timesteps):
        embedding = time_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(embedding)

        res_noise = []
        res_fusion = []

        # in stage
        x = self.inBlock(x)

        # down stage
        h = x
        num_down = 1
        for down_block in self.downBlock:
            h = down_block(h, time_emb)
            if num_down in self.downBlock_chanNum_cumsum:
                res_noise.append(h)
                res_fusion.append(h)
            num_down += 1

        # middle stage
        for middle_block in self.middleBlock:
            h = middle_block(h, time_emb)
        x1 = h
        x2 = h
        attention_layer = self.attention_block[3]
        attention_layer_1 = self.attention_block_1[3]
        x1 = x1 + attention_layer(res_noise.pop())
        x2 = x2 + attention_layer_1(res_fusion.pop())
        assert len(res_noise) == len(self.upBlock_chanNum_cumsum)
        assert len(res_fusion) == len(self.upBlock_chanNum_cumsum)

        # up stage
        num_up = 1
        num_attention = 2
        for up_block in self.upBlock:
            if num_up in self.upBlock_chanNum_cumsum:  # [2,5,8]
                x1 = up_block(x1, time_emb)
                x1_crop = x1[:, :, :res_noise[-1].shape[2], :res_noise[-1].shape[3]]
                attention_layer = self.attention_block[num_attention]
                num_attention = num_attention - 1
                x1 = x1_crop + attention_layer(res_noise.pop())
                # x1 = x1_crop + res_noise.pop()
            else:
                x1 = up_block(x1, time_emb)
            num_up += 1
        assert len(res_noise) == 0

        # # fusion stage
        num_up = 1
        num_attention = 2
        for fusion_block in self.fusionBlock:

            if num_up in self.upBlock_chanNum_cumsum:  # [2,5,8]
                x2 = fusion_block(x2, time_emb)
                x2_crop = x2[:, :, :res_fusion[-1].shape[2], :res_fusion[-1].shape[3]]
                attention_layer_1 = self.attention_block_1[num_attention]
                num_attention = num_attention - 1
                x2 = x2_crop + attention_layer_1(res_fusion.pop())
                # x2 = x2_crop + res_fusion.pop()
            else:
                x2 = fusion_block(x2, time_emb)
            num_up += 1
        assert len(res_fusion) == 0

        # out stage
        noise_out = self.outBlock(x1)
        fusion_out = self.fusion_outBlock(x2)


        return noise_out, fusion_out