import argparse
import time
import tiktoken

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Any, Literal, Optional

class Expert(nn.Module): # Pa que no se queje ni na
    def __init__(self, in_dim: int, m: int, bias: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,x, *args):
        pass

class Router(nn.Module): # Pa que no se queje ni na
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args):
        pass



class LatentMoe(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.shared_expert = config.shared_expert
        self.bias = config.bias
        d,m,latent = config.d, config.m, config.latent #d es dimension de embedins, m es el latente de los expertos, latent es que tanto se reduce d
        self.W_down=nn.Linear(d,latent,bias=self.bias)
        self.W_up=nn.Linear(latent,d,bias=self.bias)

        self.experts = nn.Module(
            Expert(latent,m,bias=self.bias) for _ in range(config.n_experts)
        )

        self.router = nn.Linear(d,self.n_experts,bias=self.bias) # como pollas lo hago
        self.shared_expert = nn.Module(Expert(d,m,bias=self.bias) for _ in range(config.shared_expert))
        

