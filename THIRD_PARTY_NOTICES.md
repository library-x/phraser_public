# Third-party code

These files are adapted from other projects and remain under their original licenses.

| file | source | license |
|---|---|---|
| `phraser/modules/natblocks.py` | [SHI-Labs/Neighborhood-Attention-Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) | MIT |
| `phraser/modules/drop.py` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (timm) | Apache-2.0 |
| `phraser/modules/patches.py` | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) | see upstream |
| `phraser/modules/rope.py` | [lucidrains/rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch) | MIT |
| `phraser/modules/self_similarity.py` | [meta-llama/llama](https://github.com/meta-llama/llama) `model.py` | Llama 2 Community License |
| `phraser/scnet/` | [starrytong/SCNet](https://github.com/starrytong/SCNet), parts from [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | see upstream / MIT |

Runtime dependencies that are not vendored: MuQ (`OpenMuQ/MuQ-large-msd-iter`, weights are
CC BY-NC 4.0), SCNet-large weights, NATTEN, and mir_eval.

`self_similarity.py` carries Meta's header: "This software may be used and distributed
according to the terms of the Llama 2 Community License Agreement."
