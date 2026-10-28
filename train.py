import torch
from gpt import Config, GPTModel, load_weights_into_gpt
from src.moe.latent_moe import LatentMoe
from safetensors.torch import load_file
import time
import torch.nn as nn
from torchmetrics import Accuracy
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import os
import numpy as np
from typing import TypeVar, Generic, Literal, overload
from datetime import timedelta
import pickle


class DataLoader:
    def __init__(
        self,
        config: Config,
        datapath: str,
        min_idx: int,
        max_idx: int,
    ):
        self.tokens = np.load(datapath)[min_idx:max_idx]
        self.batch = config.batch_size
        self.ctx = config.context_length
        self.max_iters = len(self.tokens) // (config.batch_size * config.context_length + 1)
        assert self.max_iters >= 1, "Dataset doesn't have enough data to perform a single training pass"
        self.iter = 0
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter < self.max_iters:
            buf = self.tokens[self.pos:self.pos + (self.batch * self.ctx + 1)]
            X = torch.from_numpy(buf[:-1]).to(torch.long).view(self.batch, self.ctx)
            y = torch.from_numpy(buf[1:]).to(torch.long).view(self.batch, self.ctx)
            self.iter += 1
            self.pos += self.batch * self.ctx + 1
            return X, y
        else:
            self.iter = 0
            self.pos = 0
            raise StopIteration()

class FinewebDataLoader:
    def __init__(
        self,
        batch_size: int,
        context_size: int,
        loaded_tokens: int,
        tokens_per_epoch: int,
        split: Literal["train", "test", "val"],
        min_idx: int,
        max_idx: int,
    ):
        self.B, self.T = batch_size, context_size
        self.split = split
        self.data_root = "/workspace/Datasets/FineWeb"
        self.shards = sorted(
            [file for file in os.listdir(self.data_root) if "data" in file]
        )
        with open(os.path.join(self.data_root, "metadata.pkl"), "rb") as pf:
            metadata = pickle.load(pf)
        self.shard_sizes = metadata["lengths"]
        self.loaded_tokens = loaded_tokens
        self.tokens_per_epoch = tokens_per_epoch
        self.pos = self.B * self.T
        self.tokens = torch.empty((loaded_tokens,), dtype=torch.long)
        self.g_idx = np.where(np.cumsum(self.shard_sizes) - min_idx >= 0)[0][0].item()
        self.g_count = 0
        if self.g_idx == 0:
            self.sl_min = min_idx
        else:
            self.sl_min: int = (
                self.shard_sizes[self.g_idx]
                - (np.cumsum(self.shard_sizes) - min_idx)[self.g_idx].item()
            )
        self.sl_max = self.sl_min + loaded_tokens
        self.sl_idx = self.sl_min
        self.ll_idx = 0
        self.ll_offset = 0
        self.acc_iterations = 0
        self.max_reachable_tokens = (max_idx - min_idx) - (
            (max_idx - min_idx) % (self.B * self.T)
        )
        self.ll_limit = loaded_tokens - loaded_tokens % (
            self.B * self.T
        )
        self.acc_limit = tokens_per_epoch // (self.B * self.T)
        self.tokens = self.load_tokens(self.sl_min, self.sl_max)

    def load_tokens(self, min: int, max: int):
        npts = np.load(
            os.path.join(self.data_root, self.shards[self.g_idx]), mmap_mode="r"
        )
        return torch.tensor(npts[min:max], dtype=torch.long)

    def reset(self):
        self.pos = self.B * self.T
        self.tokens = torch.empty((self.loaded_tokens,), dtype=torch.long)
        self.g_idx = np.argmin(abs(np.cumsum(self.shard_sizes) - self.min_idx)).item()
        self.g_count = 0
        if self.g_idx == 0:
            self.sl_min = self.min_idx
        else:
            self.sl_min: int = (
                self.shard_sizes[self.g_idx]
                - (np.cumsum(self.shard_sizes) - self.min_idx)[self.g_idx].item()
            )
        self.sl_max = self.sl_min + self.loaded_tokens
        self.ll_idx = 0
        self.ll_offset = 0
        self.acc_iterations = 0
        self.max_reachable_tokens = (self.max_idx - self.min_idx) - (
            (self.max_idx - self.min_idx) % (self.B * self.T)
        )
        self.ll_limit = self.loaded_tokens - self.loaded_tokens % (
            self.B * self.T
        )
        self.acc_limit = self.tokens_per_epoch // (self.B * self.T)
        self.tokens = self.load_tokens(self.sl_min, self.sl_max)

    def resume(self, state: list):
        raise NotImplementedError("We ball")

    def __iter__(self):
        return self

    def __next__(self):
        if self.acc_iterations < self.acc_limit:
            B, T = self.B, self.T
            buf = torch.empty((B * T,), dtype=torch.long)
            buf = self.tokens[self.pos : self.pos + B * T + 1].to(buf, copy=True)
            X = buf[:-1].view(B, T)
            y = buf[1:].view(B, T)
            self.pos += B * T
            self.ll_idx += B * T
            self.sl_idx += B * T
            self.g_count += B * T
            self.acc_iterations += 1
            if self.g_count == self.max_reachable_tokens:
                # reset state variables
                self.reset()
                raise StopIteration()
            if self.ll_idx == self.ll_limit:
                # checks for the amount of (B * T * world_size) segments that fit in loaded_tokens
                ll_remainder = self.loaded_tokens - self.ll_idx
                sl_remainder = self.shard_sizes[self.g_idx] - self.sl_idx
                more_shards = self.g_idx != len(self.shards) - 1
                if self.shard_sizes[self.g_idx] - self.sl_idx >= self.loaded_tokens:
                    # npts = np.load(data[g_idx], mmap_mode="r")
                    # self.tokens[:ll_remainder] = torch.tensor(npts[sl_min + (ll_idx + ll_offset):sl_max], dtype=torch.uint16)
                    # self.tokens[ll_remainder:] = torch.tensor(npts[sl_max:sl_max + (loaded_tokens - ll_remainder)], dtype=torch.uint16)
                    self.tokens[:ll_remainder] = self.load_tokens(
                        self.sl_min + (self.ll_idx + self.ll_offset), self.sl_max
                    )
                    self.tokens[ll_remainder:] = self.load_tokens(
                        self.sl_max, self.sl_max + (self.loaded_tokens - ll_remainder)
                    )
                    self.pos = B * T
                    self.sl_min = self.sl_min + (self.ll_idx + self.ll_offset)
                    self.sl_max = self.sl_max + (self.loaded_tokens - ll_remainder)
                    self.ll_idx = 0
                    self.ll_offset = 0
                else:
                    # there aren't enough tokens in the current shard for another loaded_tokens segment
                    # npts = np.load(data[g_idx], mmap_mode="r")
                    # self.tokens[:sl_remainder] = torch.tensor(npts[sl_min + ll_idx:shard_sizes[g_idx]], dtype=torch.uint16)
                    self.tokens[:sl_remainder] = self.load_tokens(
                        self.sl_min + self.ll_idx, self.shard_sizes[self.g_idx]
                    )
                    self.ll_idx = 0
                    if more_shards:
                        self.g_idx += 1
                        self.sl_idx = -sl_remainder
                        # npts = np.load(data[g_idx], mmap_mode="r")
                        # self.tokens[sl_remainder:] = torch.tensor(npts[:(loaded_tokens - sl_remainder)], dtype=torch.uint16)
                        self.tokens[sl_remainder:] = self.load_tokens(
                            0, self.loaded_tokens - sl_remainder
                        )
                        self.ll_offset = -sl_remainder
                        self.sl_min = 0
                        self.sl_max = self.loaded_tokens - sl_remainder
                    self.pos = B * T
            return X, y
        else:
            self.acc_iterations = 0
            raise StopIteration()


Val = TypeVar("Val", Literal[True], Literal[False])

class SimpleSplitDataset(Generic[Val]):
    def __init__(
        self,
        config: Config,
        datapath: str,
        use_val: Val
    ):
        if not os.path.exists(datapath):
            raise FileNotFoundError("Data file not found")
        self.datapath = datapath
        self.config = config
        self.data = np.load(datapath)
        self.use_val = use_val

    @overload
    def split(self: "SimpleSplitDataset[Literal[False]]", train_size: float, test_size: float) -> tuple[DataLoader, DataLoader]:
        ...

    @overload
    def split(self: "SimpleSplitDataset[Literal[True]]", train_size: float, test_size: float, val_size: float) -> tuple[DataLoader, DataLoader, DataLoader]:
        ...

    def split(
        self,
        train_size: float,
        test_size: float,
        val_size: float = 0.0
    ):
        if self.use_val:
            assert train_size + test_size + val_size == 1.0, "Sizes have to add up to 1."
        else:
            assert train_size + test_size == 1.0, "Sizes have to add up to 1."

        if self.use_val:
            assert train_size + test_size + val_size == 1.0, "Sizes have to add up to 1."
            min_train_idx = 0
            max_train_idx = int(train_size * len(self.data))
            min_test_idx = max_train_idx
            max_test_idx = int((train_size + test_size) * len(self.data))
            min_val_idx = max_test_idx
            max_val_idx = len(self.data)
            return (
                DataLoader(self.config, self.datapath, min_idx=min_train_idx, max_idx=max_train_idx),
                DataLoader(self.config, self.datapath, min_idx=min_test_idx, max_idx=max_test_idx),
                DataLoader(self.config, self.datapath, min_idx=min_val_idx, max_idx=max_val_idx),
            )
        else:
            assert train_size + test_size == 1.0, "Sizes have to add up to 1."
            min_train_idx = 0
            max_train_idx = int(train_size * len(self.data))
            min_test_idx = max_train_idx
            max_test_idx = len(self.data)
            return (
                DataLoader(self.config, self.datapath, min_idx=min_train_idx, max_idx=max_train_idx),
                DataLoader(self.config, self.datapath, min_idx=min_test_idx, max_idx=max_test_idx),
            )

class SplitGenerator(Generic[Val]):
    def __init__(
        self,
        data_path: str,
        use_val: Val,
        train_size: float = 0.8,
        test_size: float = 0.2,
        val_size: float = 0.0
    ):
        """
        Class to generate cross validation splits for a data assortment.

        Arguments:
            - data_path (str): Path to the directory containing the numpy files of the dataset.
            - use_val (bool): Whether to use a validation partition.
            - train_size (float): Percentage of the dataset to use for training.
            - test_size (float): Percentage of the dataset to use for testing.
            - val_size (float): Percentage of the dataset to use for validation (ignored if use_val is False).
        """
        self.testing = int(os.environ.get("DLMAKE_TESTING", -1)) != -1
        if not self.testing:
            with open(os.path.join(data_path, "metadata.pkl"), "rb") as pf:
                metadata = pickle.load(pf)
            self.total_tokens = metadata["total_tokens"]
        else:
            self.total_tokens = None
        self.data_path = data_path
        self.use_val = use_val
        if use_val:
            assert val_size != 0 and train_size + test_size + val_size == 1, "All partitions usage must add to 1."
            self.train_size = train_size
            self.test_size = test_size
            self.val_size = val_size
        else:
            assert train_size + test_size == 1, "All partitions usage must add to 1."
            self.train_size = train_size
            self.test_size = test_size
            self.val_size = 0.0

    @overload
    def generate_splits(
        self: "SplitGenerator[Literal[True]]",
        dataloader: type[FinewebDataLoader],
        batch_size: int,
        context_size: int,
        loaded_tokens: int,
        tokens_per_epoch: int,
    ) -> tuple[FinewebDataLoader, FinewebDataLoader, FinewebDataLoader]:
        ...

    @overload
    def generate_splits(
        self: "SplitGenerator[Literal[False]]",
        dataloader: type[FinewebDataLoader],
        batch_size: int,
        context_size: int,
        loaded_tokens: int,
        tokens_per_epoch: int,
    ) -> tuple[FinewebDataLoader, FinewebDataLoader]:
        ...

    def generate_splits(
        self,
        dataloader: type[FinewebDataLoader],
        batch_size: int,
        context_size: int,
        loaded_tokens: int,
        tokens_per_epoch: int,
        config: Config,
    ):
        if self.use_val:
            if not self.testing:
                assert self.total_tokens is not None
                min_train_idx = 0
                max_train_idx = int(self.total_tokens * self.train_size)
                min_test_idx = max_train_idx
                max_test_idx = int(self.total_tokens * (self.train_size + self.test_size))
                min_val_idx = max_test_idx
                max_val_idx = self.total_tokens
            else:
                min_train_idx = 0
                max_train_idx = 0
                min_test_idx = 0
                max_test_idx = 0
                min_val_idx = 0
                max_val_idx = 0
            
            return (
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="train",
                    min_idx=min_train_idx,
                    max_idx=max_train_idx,
                ),
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="test",
                    min_idx=min_test_idx,
                    max_idx=max_test_idx,
                ),
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="val",
                    min_idx=min_val_idx,
                    max_idx=max_val_idx,
                ),
            )
        else:
            if not self.testing:
                assert self.total_tokens is not None
                min_train_idx = 0
                max_train_idx = int(self.total_tokens * self.train_size)
                min_test_idx = max_train_idx
                max_test_idx = self.total_tokens
            else:
                min_train_idx = 0
                max_train_idx = 0
                min_test_idx = 0
                max_test_idx = 0

            return (
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="train",
                    min_idx=min_train_idx,
                    max_idx=max_train_idx,
                ),
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="test",
                    min_idx=min_test_idx,
                    max_idx=max_test_idx,
                ),
            )


def train(
    model: GPTModel,
    config: Config,
    epochs: int,
    test_rate: int,
    val_rate: int,
    checkpoint_rate: int,
    use_val: bool,
    train_size: float = 0.8,
    test_size: float = 0.2,
    val_size: float = 0.0,
):
    # if use_val:
    #     train_dataloader, test_dataloader, val_dataloader = SplitDataset(config, "./data/test.npy", use_val=True).split(
    #         train_size=train_size,
    #         test_size=test_size,
    #         val_size=val_size
    #     )
    # else:
    #     train_dataloader, test_dataloader = SplitDataset(config, "./data/test.npy", use_val=False).split(
    #         train_size=train_size,
    #         test_size=test_size,
    #     )
    #     val_dataloader = None
    if use_val:
        train_dataloader, test_dataloader, val_dataloader = SplitGenerator(
            data_path="workspace/Datasets/FineWeb",
            use_val=True, 
            train_size=train_size,
            test_size=test_size,
            val_size=val_size
        ).generate_splits(
            dataloader=FinewebDataLoader,
            batch_size=config.batch_size,
            context_size=config.context_length,
            loaded_tokens=2 ** 18,
            tokens_per_epoch=2 ** 15,
        )
    else:
        val_dataloader = None
        train_dataloader, test_dataloader  = SplitGenerator(
            data_path="workspace/Datasets/FineWeb",
            use_val=False, 
            train_size=train_size,
            test_size=test_size,
            val_size=val_size
        ).generate_splits(
            dataloader=FinewebDataLoader,
            batch_size=config.batch_size,
            context_size=config.context_length,
            loaded_tokens=2 ** 18,
            tokens_per_epoch=2 ** 15,
        )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.95), eps=1e-8, fused=True)  # type: ignore
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=1e-3,
        total_steps=epochs * train_dataloader.max_iters,
    )
    accuracy = Accuracy(task="multiclass", num_classes=config.vocab_size)
    accuracy.to(device)
    loss_history = []
    acc_history = []
    test_loss_history = []
    test_acc_history = []
    val_loss_history = []
    val_acc_history = []
    norm_history = []
    times = []


    for epoch in range(epochs):
        acc_time = 0
        last_epoch = (epoch == epochs - 1)

        if (epoch % test_rate == 0 or last_epoch):
            model.eval()
            with torch.inference_mode():
                test_loss = torch.tensor(0., device=device)
                test_acc = torch.tensor(0., device=device)
                i = 0
                for i, data in enumerate(test_dataloader):
                    gpu_data = [v.to(device) for v in data]
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits: torch.Tensor = model(*gpu_data[:-1])
                        loss: torch.Tensor = loss_fn(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                    loss /= test_dataloader.max_iters
                    acc: torch.Tensor = accuracy(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                    test_loss += loss.detach()
                    test_acc += acc.detach()
                test_acc /= (i + 1)
                print(f"[Test] [{epoch:>{len(str(epochs))}}/{epochs}] Loss: {test_loss.item():.2f} | Accuracy: {(test_acc * 100).item():.2f} %")

            test_loss_history.append(test_loss)
            test_acc_history.append(test_acc)


        if (epoch % val_rate == 0 or last_epoch) and use_val:
            assert val_dataloader is not None
            model.eval()
            with torch.inference_mode():
                val_loss = torch.tensor(0., device=device)
                val_acc = torch.tensor(0., device=device)
                i = 0
                for i, data in enumerate(val_dataloader):
                    gpu_data = [v.to(device) for v in data]
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits: torch.Tensor = model(*gpu_data[:-1])
                        loss: torch.Tensor = loss_fn(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                    loss /= val_dataloader.max_iters
                    acc: torch.Tensor = accuracy(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                    val_loss += loss.detach()
                    val_acc += acc.detach()
                val_acc /= (i + 1)
                print(f"[Val] [{epoch:>{len(str(epochs))}}/{epochs}] Loss: {val_loss.item():.2f} | Accuracy: {(val_acc * 100).item():.2f} %")

        else:
            val_loss = None
            val_acc = None

        start = time.time()
        losses = torch.tensor(0., device=device)
        accs = torch.tensor(0., device=device)
        total = 0
        model.train()
        optimizer.zero_grad()

        for i, data in enumerate(train_dataloader):
            gpu_data = [v.to(device) for v in data]
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                y_preds: torch.Tensor = model(*gpu_data[:-1])
                loss: torch.Tensor = loss_fn(y_preds.view(-1, y_preds.shape[-1]), gpu_data[-1].view(-1))
            loss = loss / train_dataloader.max_iters
            losses += loss.detach()
            acc: torch.Tensor = accuracy(y_preds.view(-1, y_preds.shape[-1]), gpu_data[-1].view(-1))
            accs += acc.detach()
            loss.backward()
            total = i
        accs /= (i + 1)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        end = time.time()
        dt = end - start
        times.append(dt)
        avg = torch.median(torch.tensor(times))
        tok_per_sec = (train_dataloader.batch * train_dataloader.ctx * train_dataloader.max_iters) / dt
        acc_time += dt
        # seconds = ((epochs * acc_time) / (epoch + 1)) - acc_time  # expectation - evidence
        seconds = (epochs - (epoch + 1)) * avg
        time_format = str(timedelta(seconds=seconds))
        d, h, m, s = "", "", "", ""
        if "day" in time_format:
            d = time_format.split(", ")[0].split(" ")[0]
            h, m, s = time_format.split(", ")[1].split(":")
            m, s = m.lstrip('0'), s.lstrip('0')
        else:
            h, m, s = time_format.split(":")
        eta = f"ETA: {'' if d == '' else d + ' days '}{h} h {m} min {s} s"

        lr = scheduler.get_last_lr()[0]
        loss_history.append(losses)
        acc_history.append(accs)
        norm_history.append(norm)

        if epoch % checkpoint_rate == 0 or last_epoch:
            if not os.path.exists(os.path.join("data", "training")):
                os.mkdir(os.path.join("data", "training"))
            torch.save(model.state_dict(), os.path.join("./data", "training", "snapshot.pth"))
            print("Saved checkpoint")

        print(f"[Train] [{epoch:>{len(str(epochs))}}/{epochs}] Loss: {losses.item():.3f} | Accuracy: {(accs * 100).item():.2f} % | Norm: {norm:.2f} | lr: {lr:3e} | dt: {dt * 1000:.2f} ms | tkps: {tok_per_sec:.2f} | {eta}")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config(

    )
    model = GPTModel(config, ff=LatentMoe)
    state_dict = load_file("./data/model.safetensors")
    load_weights_into_gpt(model, state_dict, latent=True)
    model.to(device)
    train(
        model,
        config,
        epochs=200,
        test_rate=50,
        val_rate=80,
        checkpoint_rate=25,
        use_val=False
    )