from gpt import GPTModel, load_weights_into_gpt, Config
from multiprocessing.connection import Connection
from src.moe.latent_moe import LatentMoe
from safetensors.torch import load_file
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from typing import Literal
import torch.nn as nn
import subprocess
import tiktoken
import warnings
import torch
import time


def generate(
    model: GPTModel,
    text: str,
    n: int,
    behavior: Literal["best", "temp"] = "best",
    temp: float = 0.0,
    topk: int = 1,
    topp: float = 0.1,
    device: str = "cuda",
    show: bool = True,
):
    if show:
        print(f"Input: {text}")
        print(f"Response: ")
    encoding = tiktoken.get_encoding("gpt2")
    x = torch.tensor(encoding.encode(text)).to(device)
    for _ in range(n):
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                y = model(x.unsqueeze(0))
        if behavior == "best":
            choice = torch.argmax(y[:, -1, :].squeeze(0), dim=-1, keepdim=True)
        else:  # temp
            assert temp > 0.0, "For a temperature based decoding, please ensure the temperature is higher than 0"
            y = (y[:, -1, :] / temp).squeeze(0)
            indices = torch.topk(y, topk).indices
            keep = torch.zeros_like(y, device=y.device, dtype=torch.bool)
            keep[indices] = True
            y = torch.clone(y)
            y[torch.logical_not(y)] = float("-inf")

            sorted_logits, sorted_indices = torch.sort(y, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative_probs > topp
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            y[sorted_indices[remove]] = float("-inf")
            choice = torch.multinomial(torch.softmax(y, dim=-1), num_samples=1)

        x = torch.hstack([x, choice])
        token = encoding.decode([choice.item()])  # type: ignore
        if show:
            print(f"{token}", end="")


def vram_check(pipe: Connection, frec: float):
    while True:
        if pipe.poll():
            content = pipe.recv()
            if content == "end":
                break
        else:
            used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"]).decode().strip().split(" ")[0])
            pipe.send(used)
            time.sleep(frec)


def run_gen(model: GPTModel, n: int):
    time.sleep(3)
    generate(
        model,
        "Hello, tell me a bit about yourself.",
        n=n,
        behavior="best",
        device="cuda",
        show=False,
    )
    

def compute_perplexity_for_batch(model: GPTModel, input_texts: list[str]):
    encoding = tiktoken.get_encoding("gpt2")
    inputs = [torch.tensor(encoding.encode(s)).to(device) for s in input_texts]
    outs = []
    loss_fn = nn.CrossEntropyLoss()

    for x in inputs:
        x = x.unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        shift_logits = logits[:, :-1, :] 
        shift_labels = x[:, 1:]
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nll = loss * x.size(1)
        ppl = torch.exp(nll / x.size(1)).item()
        outs.append(ppl)

    return {
        "mean_perplexities": outs
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()
    latent_model = GPTModel(config, ff=LatentMoe)  # type: ignore
    old_model = GPTModel(config)  # type: ignore
    state_dict = load_file("./data/model.safetensors")
    load_weights_into_gpt(old_model, state_dict)
    state_dict = torch.load("./data/training/add_trained.pth")
    latent_model.load_state_dict(state_dict)
    latent_model.to(device)
    old_model.to(device)

    print(f"Number of parameters in old model: {sum(p.nelement() for p in old_model.parameters()):,}")
    print(f"Number of parameters in latent model: {sum(p.nelement() for p in latent_model.parameters()):,}")
    print()


    lengths = [10, 100, 500, 1000]
    old_model.to("cpu")
    latent_model.to("cpu")
    for ntokens in lengths:
        old_model.to(device)
        start = time.time()
        generate(
            old_model,
            "Hello, tell me a bit about yourself.",
            n=ntokens,
            behavior="best",
            device=device,
            show=False
        )
        end = time.time()
        s = end - start
        tokps_old = ntokens / s
        old_model.to("cpu")


        latent_model.to(device)
        start = time.time()
        generate(
            latent_model,
            "Hello, tell me a bit about yourself.",
            n=ntokens,
            behavior="best",
            device=device,
            show=False,
        )
        end = time.time()
        s = end - start
        tokps_new = ntokens / s
        latent_model.to("cpu")
        print(f"Generation speed at {ntokens} tokens.")
        print(f"Speed of base GPT2: {tokps_old:.2f} tok/s")
        print(f"Speed of LatentMoE GPT2: {tokps_new:.2f} tok/s")
        print()
        torch.cuda.empty_cache()


    old_model.to(device)
    latent_model.to("cpu")
    ctx = mp.get_context("spawn")
    end1, end2 = ctx.Pipe()
    frec = 0.1
    p1 = ctx.Process(target=vram_check, args=(end2, frec))
    p2 = ctx.Process(target=run_gen, args=(old_model, 500))
    p1.start()
    p2.start()
    old_usage = []
    while True:
        if p2.is_alive():
            used = end1.recv()
            old_usage.append(used)
        else:
            end1.send("end")
            break
    p1.join()
    p2.join()
    old_model.to("cpu")
    latent_model.to(device)
    end1, end2 = ctx.Pipe()
    frec = 0.1
    n = 1000
    p1 = ctx.Process(target=vram_check, args=(end2, frec))
    p2 = ctx.Process(target=run_gen, args=(latent_model, n))
    p1.start()
    p2.start()
    latent_usage = []
    while True:
        if p2.is_alive():
            used = end1.recv()
            latent_usage.append(used)
        else:
            end1.send("end")
            break
    p1.join()
    p2.join()
    old_model.to(device)

    plt.figure(figsize=(12, 8))
    old_usage = torch.tensor(old_usage)
    latent_usage = torch.tensor(latent_usage)
    x_old = torch.linspace(0, len(old_usage) * frec, len(old_usage))
    x_latent = torch.linspace(0, len(latent_usage) * frec, len(latent_usage))
    plt.plot(x_old, old_usage - torch.min(old_usage), c="royalblue", label="Base GPT2")
    plt.plot(x_latent, latent_usage - torch.min(latent_usage), c="orangered", label="LatentMoE GPT2")
    plt.xlabel("Time (s)")
    plt.ylabel("VRAM usage (MiB)")
    plt.title(f"VRAM usage while generating {n} tokens")
    plt.legend()
    plt.show()
    torch.save({
        "base_usage": old_usage,
        "latent_usage": latent_usage,
    }, "model_usages.pth")
    print("Saved values in file")
    print()

    s1 = "2132 + 5341 = 7473"
    s2 = "7412 + 4521 = 11933"
    base_results = compute_perplexity_for_batch(old_model, [s1, s2])
    latent_results = compute_perplexity_for_batch(latent_model, [s1, s2])
    print("Perplexity results")
    print(f"Base GPT2: {[round(v, 2) for v in base_results['mean_perplexities']]}")
    print(f"LatentMoE GPT2: {[round(v, 2) for v in latent_results['mean_perplexities']]}")
