import argparse
import copy
import glob
from pathlib import Path

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import functional as F

from phraser.modules.natblocks import NAT2DBlock, NAT1DBlock
from phraser.modules.patches import PatchMerging, PatchExpand, PatchExpand2
from phraser.modules.rope import RotaryEmbedding, Wav2Vec2ConformerRotaryPositionalEmbeddingScale
from phraser.modules.self_similarity import SelfSimilarityAttention, ModelArgs
import torch, librosa
from muq import MuQ
from phraser.utils.utils import SECTION_COLORS, SECTION_LABELS, STEPS_PER_SEC, MUQ_SR, most_common_spacing, robust_periodic_reconstruction, BEATS_HZ

from phraser.utils.utils import SECTION_LABELS, MUQ_SR

ENCODER_BATCH = 32
class DownLayer(nn.Module):

    def __init__(self, *, dim_embed, out_dim_embed, kernel_size, num_heads, blocks, apply_down = True):
        super().__init__()

        self.layer = NAT2DBlock(dim=dim_embed, num_heads=num_heads, kernel_size=kernel_size, depth=blocks)
        if apply_down:
            self.down = PatchMerging(kernel_size, dim_embed, out_dim_embed)
        else:
            self.down = None

    def forward(self, x):
        x = self.layer(x)
        if self.down is None:
            return x, x

        return self.down(x), x


class UpLayer(nn.Module):

    def __init__(self, *, dim_embed, out_dim_embed, kernel_size, num_heads, depth, apply_up = True):
        super().__init__()

        self.layer = NAT2DBlock(dim=out_dim_embed, num_heads=num_heads, kernel_size=kernel_size, depth=depth)
        if apply_up:
            self.up = PatchExpand(kernel_size, dim_embed, out_dim_embed)
        else:
            self.up = None


    def forward(self, x, x_res):
        if self.up is not None:
            x = self.up(x)
        assert x.shape == x_res.shape, "Wrong shape"
        x = x_res + x
        return self.layer(x)

class UpSuperResolutionLayer(nn.Module):

    def __init__(self, *, dim_embed, out_dim_embed, kernel_size, num_heads, depth, apply_up = True):
        super().__init__()

        self.layer = NAT1DBlock(dim=out_dim_embed, num_heads=num_heads, kernel_size=15, depth=depth)
        self.up = PatchExpand(4, dim_embed, out_dim_embed)

    def forward(self, x):
        x = self.up(x)
        return self.layer(x)


class PredictHead(nn.Module):

    def __init__(self, dim_embed):
        super().__init__()

        self.rope = RotaryEmbedding(dim = 16)

        self.layer1 = NAT2DBlock(dim=dim_embed, num_heads=16, kernel_size=5, depth=1, rope=self.rope)
        self.layer2 = NAT1DBlock(dim=dim_embed, num_heads=16, kernel_size=15, depth=1, rope=self.rope)
        self.layer3 = NAT2DBlock(dim=dim_embed, num_heads=16, kernel_size=5, depth=1, rope=self.rope)

        self.predict_split = nn.Linear(dim_embed, 1)

        self.predict_type = nn.Linear(dim_embed, len(SECTION_LABELS))
        self.predict_silent = nn.Linear(dim_embed, 1)


    def forward(self, x):

        x = self.layer1(x)
        B, S, L, D = x.shape
        x = x.reshape(B*S,L,D)
        x = self.layer2(x)
        x = x.reshape(B, S, L, D)
        x = self.layer3(x)

        silent = self.predict_silent(x[:, :4, :, :])

        split = self.predict_split(x)
        section_type = self.predict_type(x[:, -1, :, :])

        return split, silent, section_type


class PredictBeatsHead(nn.Module):

    def __init__(self, dim_embed):
        super().__init__()

        self.rope = RotaryEmbedding(dim = 8)

        self.layer = NAT2DBlock(dim=dim_embed, num_heads=8, kernel_size=5, depth=2)
        self.up = PatchExpand2(4, dim_embed, dim_embed)

        self.layer1 = NAT1DBlock(dim=dim_embed, num_heads=4, kernel_size=5, depth=1, rope=self.rope)
        self.layer2 = NAT1DBlock(dim=dim_embed, num_heads=4, kernel_size=15, depth=1, rope=self.rope)
        self.layer3 = NAT1DBlock(dim=dim_embed, num_heads=4, kernel_size=5, depth=1, rope=self.rope)

        self.predict_beats = nn.Linear(dim_embed, 1)
        self.predict_on_sets = nn.Linear(dim_embed, 1)


    def forward(self, x):

        x = self.layer(x)
        x = self.up( x[:, -1, :, :])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        bats = self.predict_beats(x)
        on_sets = self.predict_on_sets(x)


        return bats, on_sets

class PhraserModel(nn.Module):

    def __init__(self, dim_embed, num_heads):
        super().__init__()

        self.down_layers = nn.Sequential(
            DownLayer(dim_embed=dim_embed, out_dim_embed=dim_embed*2, kernel_size=5, blocks=2, num_heads=num_heads),
            DownLayer(dim_embed=dim_embed*2, out_dim_embed=dim_embed*4, kernel_size=5, blocks=2, num_heads=num_heads*2),
            DownLayer(dim_embed=dim_embed*4, out_dim_embed=dim_embed*4, kernel_size=5, blocks=2, num_heads=num_heads*2, apply_down=False),

        )

        self.up_layers = nn.Sequential(
            UpLayer(dim_embed=dim_embed*4, out_dim_embed=dim_embed*4, kernel_size=5, depth=2, num_heads=num_heads*2, apply_up=False),
            UpLayer(dim_embed=dim_embed*4, out_dim_embed=dim_embed*2, kernel_size=5, depth=2, num_heads=num_heads*2),
            UpLayer(dim_embed=dim_embed*2, out_dim_embed=dim_embed, kernel_size=5, depth=2, num_heads=num_heads),
        )

      #   self.up_resolution_layers = nn.Sequential(
      #       UpSuperResolutionLayer(dim_embed=dim_embed, out_dim_embed=dim_embed, kernel_size=5, depth=2, num_heads=num_heads),
      # #      UpSuperResolutionLayer(dim_embed=dim_embed, out_dim_embed=dim_embed, kernel_size=5, depth=2, num_heads=num_heads),
      #
      #   )

        self.stem_emb =  nn.Parameter(torch.rand((1, 5, 1, dim_embed)), requires_grad=True)

        args = ModelArgs(
            input_dim=dim_embed*4, dim=dim_embed*2, n_layers=3, n_heads=num_heads,
        )

        self.self_similarity_transformer = SelfSimilarityAttention(args)
        self.predict_layer = PredictHead(dim_embed*4)
        self.predict_beats = PredictBeatsHead(dim_embed)

        self.project = nn.Linear(1024, dim_embed)
        # This will automatically fetch the checkpoint from huggingface
        # muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        #
        # config = copy.deepcopy(muq.model.conformer.config)
        # config['rope_scaling_factor'] = 10 #scale from 30 s to 5min
        # muq.model.conformer.embed_positions = Wav2Vec2ConformerRotaryPositionalEmbeddingScale(config)
        #
        # muq.eval()
        self.muq = nn.Linear(1,1)

    def parameters_grad(self):
        for name, param in self.named_parameters():
            if 'muq' not in name: #skip encoder
                yield param


    def forward_encoded(self, x):

        x = F.pad(x, (0, 0, 0, 0, 0, 1, 0, 0))

        residual_x = []
        x = self.project(x)
        x = x + self.stem_emb
        for down_layer in self.down_layers:
            x, res_x = down_layer(x)
            residual_x.append(res_x)

        B, S, L, D = x.shape
        x = x.reshape(B*S,L,D)

        e_ssm = self.self_similarity_transformer(x)
        e_ssm = e_ssm.reshape(B, S, L, D)

        x = e_ssm

        y = self.predict_layer(e_ssm)

        for up_layer, res_x in list(zip(self.up_layers, residual_x[::-1])):
            x = up_layer(x, res_x)

      #  x = self.up_resolution_layers(x[:, -1, :, :])

        y_beats = self.predict_beats(x)

        return y, e_ssm, y_beats

    def forward_encoder(self, x):
        B, S, F = x.shape
        x = x.reshape(B*S, F)

        self.muq.cuda()
        encoded = []


     #   duration = x.shape[-1] / MUQ_SR
      #  int(duration // 30)
        # start = librosa.time_to_samples(0, sr=MUQ_SR)
        # end = librosa.time_to_samples(30, sr=MUQ_SR)
        x = x.reshape(-1, F//10)

        with torch.no_grad():
            for start_idx in range(0, math.ceil(x.shape[0] / ENCODER_BATCH) * ENCODER_BATCH, ENCODER_BATCH):

                x_e = self.muq(x[start_idx:start_idx + ENCODER_BATCH]).last_hidden_state
                encoded.append(x_e)

        x = torch.concatenate(encoded)

        self.muq.cpu()
        _, N, D = x.shape
        return x.reshape(B, S, -1, D).contiguous()


    def forward(self, x):
        x = self.forward_encoder(x)
        x = x.detach()
        # Pad the second dimension from 4 to 5

        return self.forward_encoded(x)


    @classmethod
    def load(cls, log_dir, name, device="cpu"):
        path = Path(f"{log_dir}/checkpoints/")
        checkpoint = torch.load(f"{str(path)}/{name}.ckpt", map_location=device)

        model = cls(64, 8) #config
        state_dict = {k.replace("module.", ""):v for k,v in checkpoint['state_dict'].items()} #Remove DDP

        result = model.load_state_dict(state_dict, strict=False)
        missing = [x for x in result[0] if not x.startswith("muq.")]
        if missing:
            raise Exception("MISSING KEYS", missing)
        return model

    def save(self, log_dir, config, name, **kwargs):

        path = Path(f"{log_dir}/checkpoints/")
        path.mkdir(parents=True, exist_ok=True)

        state_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('muq')} #ignore encoded

        checkpoint_dict = {
            "state_dict": state_dict,
            "config": config.__dict__,
        }

        checkpoint_dict.update(kwargs)

        torch.save(checkpoint_dict, f"{str(path)}/{name}.ckpt")
       # return self.forward_encoded(x)
#torch.Size([3, 4, 10000, 1024 ])