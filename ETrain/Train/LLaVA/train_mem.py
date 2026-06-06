# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import sys, os as _os
# Ensure the repo-local ETrain/ takes precedence over the editable src/etrain install.
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Need to call this before importing transformers.
from ETrain.Train.LLaVA.attn_flash_train import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from ETrain.Train.LLaVA.train import train

if __name__ == "__main__":
    train()
