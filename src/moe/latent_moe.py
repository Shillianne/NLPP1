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


class Router(nn.Module):
    def __init__(
        self,
        d: int,
        n_experts: int,
        top_k: int,
        bias: bool = False
    ):
        super(Router, self).__init__()

        self.experts_per_tok = top_k

        self.linear = nn.Linear(
            in_features=d,
            out_features=n_experts,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        scores = self.linear(x)
        scores = torch.softmax(scores, dim=-1)
        indices = torch.topk(scores, self.experts_per_tok, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights *= 1.2
        return weights.type_as(x), indices


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
        x_reducido = self.W_down(x) # (T,latent)

        flat_idx=topk_idx.reshape(-1) # esto pasa a (T*top_k,)
        flat_pesos=topk_pesos.reshape(-1) # (T*top_k,)
        token_ids=torch.arange(T,device=self.device).repeat_interleave(self.top_k)# (T*top_k,) que es por asi decirlo una lista enumerada de la longitud de T y cada numero se repite topk veces


        # empezamos a repartir a los expertos
        # TODO: COMO MIERDA HAGO LO DE DIVIDIR ENTRE LOS EXPERTOS
        # postada, esta parte la he escrito mirando el repo de latent moe k no sabia como hacerlo

        ids_ordenadas = torch.argsort(flat_idx)
        tokens_odenados = token_ids[ids_ordenadas]
        pesos_ordenados = flat_pesos[ids_ordenadas]
        z_sorted = x_reducido[tokens_odenados]

        counts = torch.bincount(
            flat_idx,minlength=self.n_experts
        )

        offsets = torch.cumsum(counts,dim=0)
        counts_list= counts.tolist()
        ends=offsets.tolist()



        out_sorted = torch.empty_like(z_sorted)  # (S, l)
        start = 0
        for e, n_e in enumerate(counts_list):
            if n_e == 0:
                continue
            end = ends[e]
            out_sorted[start:end] = self.experts[e](
                z_sorted[start:end]
            )
            start = end

        latent_out = torch.zeros_like(x_reducido)  # (T, l)
        latent_out.index_add_(
            0, tokens_odenados, out_sorted * pesos_ordenados.unsqueeze(-1)
        )



        y = self.W_up(latent_out) 
        for exp in self.shared_experts: # estos, pos los expertos que no se meten en el latent
            y=y+exp(x_flat)

        return y.reshape(x_shape).to(input_device)

