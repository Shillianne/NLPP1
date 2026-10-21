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
        self.linear = nn.Linear(in_dim, m, bias=bias)

    def forward(self,x, *args):
        return self.linear(x)

class Router(nn.Module): # Pa que no se queje ni na
    def __init__(self,d,n_experts,top_k, *args: Any, **kwargs: Any) -> None:# d: dim embedinbs, n_experts, top_k expertos que influyen en cada token
        super().__init__(*args, **kwargs)

    def forward(self, *args):
        pass


class LatentMoe(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.top_k= config.top_k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.n_shared_exp = config.n_shared_exp
        self.bias = config.bias
        d,m,latent = config.d, config.m, config.latent #d es dimension de embedins, m es el latente de los expertos, latent es que tanto se reduce d
        self.W_down=nn.Linear(d,latent,bias=self.bias)
        self.W_up=nn.Linear(latent,d,bias=self.bias)

        self.experts = nn.ModuleList(
            Expert(latent,m,bias=self.bias) for _ in range(config.n_experts)
        )

        self.router = Router(d,self.n_experts,self.top_k,bias=self.bias) # como pollas lo hago # 1ºque devuelve: pesos, 2º indices (b, n_tokens_total, n_exters)
        self.shared_experts = nn.ModuleList(Expert(d,m,bias=self.bias) for _ in range(config.n_shared_exp))



        # Meter todo en GPU
        self.W_down.to(self.device)
        self.W_up.to(self.device)
        self.router.to(self.device)
        self.shared_experts.to(self.device)
        self.experts.to(self.device)

    def forward(self, x:Tensor) -> Tensor: # meto la loss al final, primero la arquietectura
        x_shape=x.shape
        input_device=x.device
        d=self.config.d

        x_flat=x.reshape(-1,d).to(self.device)# esto lo pasa a (T,d)
        T=x_flat.shape[0]

        topk_pesos,topk_idx=self.router(x_flat) # preguntar a gabriel
        # TODO : hacer la loss, edit despues de ver el codigo de github, grabriel me pasa logits y probs?
        x_reducido = self.W_down(x) # (T,latent)

        flat_idx=topk_idx.reshape(-1) # esto pasa a (T*top_k,)
        flat_pesos=topk_idx.reshape(-1) # (T*top_k,)
        token_ids=torch.arange(T,device=self.device).repeat_interleave(self.top_k)# (T*top_k,) que es por asi decirlo una lista enumerada de la longitud de T y cada numero se repite topk veces


        # empezamos a repartir a los expertos
        # TODO: COMO MIERDA HAGO LO DE DIVIDIR ENTRE LOS EXPERTOS



        y = self.W_up(Tensor) 
        for exp in self.shared_experts:
            y=y+exp(x_flat)

        return y.reshape(x_shape).to(input_device)

