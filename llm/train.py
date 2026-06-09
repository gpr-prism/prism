#!/usr/bin/env python3
"""
Causal Language Model Training - PRISM and Baselines
====================================================
Trains and evaluates GDN, EFLA, PGDN, PRISM, and PRISM ablation variants
on pre-tokenized Arrow data, with WikiText-103 as the default public fallback.

Usage (single GPU):
    python train.py --models gdn prism --config large_130m --epochs 10

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --nproc_per_node=4 train.py \
        --models gdn efla pgdn prism \
        --config large_130m --epochs 10 \
        --batch_size 8 --grad_accum_steps 2

Usage (eval only):
    torchrun --nproc_per_node=4 train.py \
        --models gdn prism --eval_only --ckpt_dir checkpoints
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from models import CONFIGS, CausalLM, MODEL_REGISTRY, count_parameters


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world_size = 0, 0, 1

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WikiTextDataset(Dataset):
    """Causal LM dataset loaded from Arrow files or WikiText-103 fallback."""

    def __init__(self, split: str, seq_len: int, tokenizer, data_dir: str = None):
        self.seq_len = seq_len

        if data_dir and os.path.isdir(os.path.join(data_dir, split)):
            import pyarrow as pa
            split_dir = os.path.join(data_dir, split)
            arrow_files = sorted([
                os.path.join(split_dir, f)
                for f in os.listdir(split_dir)
                if f.endswith(".arrow")
            ])
            tables = [pa.ipc.open_file(f).read_all() for f in arrow_files]
            import pyarrow as pa
            table = pa.concat_tables(tables)
            all_ids = []
            for col in ["input_ids", "text"]:
                if col in table.schema.names:
                    for row in table[col].to_pylist():
                        if isinstance(row, list):
                            all_ids.extend(row)
                        else:
                            all_ids.extend(tokenizer.encode(str(row)))
                    break
        else:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            all_ids = []
            for item in ds:
                text = item["text"].strip()
                if text:
                    all_ids.extend(tokenizer.encode(text))

        # Chunk into fixed-length sequences
        total = (len(all_ids) // (seq_len + 1)) * (seq_len + 1)
        all_ids = all_ids[:total]
        self.data = torch.tensor(all_ids, dtype=torch.long).view(-1, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device,
                    epoch, rank, grad_clip, grad_accum_steps):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        _, loss = model(input_ids, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            print_rank0(f"  [WARN] NaN/Inf loss at step {step}, skipping batch", rank)
            optimizer.zero_grad(set_to_none=True)
            continue

        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        if rank == 0 and step % 200 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch} step {step}/{len(dataloader)} "
                  f"loss={loss.item() * grad_accum_steps:.4f} lr={lr:.2e}", flush=True)

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        _, loss = model(input_ids, targets)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            n_tokens = targets.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    bpc = avg_loss / math.log(2)
    return avg_loss, ppl, bpc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Causal LM: PRISM and Baselines"
    )
    parser.add_argument("--models", nargs="+",
                        default=["gdn", "prism"],
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Models to train/evaluate")
    parser.add_argument("--config", type=str, default="large_130m",
                        choices=list(CONFIGS.keys()),
                        help="Model size config")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=1500)
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to pre-tokenized Arrow data (data/ folder)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--solver_steps", type=int, default=None,
                        help="Override solver_steps for PRISM (default: from config)")
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_distributed()

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    print_rank0(
        f"[Setup] world_size={world_size}, device={device}, "
        f"effective_batch={args.batch_size * world_size * args.grad_accum_steps}",
        rank
    )

    # Tokenizer
    from transformers import GPT2Tokenizer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(script_dir, "ft_local", "tokenizer_gpt2")
    if not os.path.isdir(tokenizer_path):
        tokenizer_path = "gpt2"   # fallback: download from HuggingFace
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Config
    cfg = CONFIGS[args.config]
    cfg.vocab_size = tokenizer.vocab_size
    if args.solver_steps is not None:
        cfg.solver_steps = args.solver_steps

    print_rank0(
        f"[Config] {args.config}: embed_dim={cfg.embed_dim}, layers={cfg.num_layers}, "
        f"heads={cfg.num_heads}, seq_len={cfg.seq_len}, solver_steps={cfg.solver_steps}",
        rank
    )

    # Auto-detect data directory
    data_dir = args.data_dir
    if data_dir is None:
        candidate = os.path.join(script_dir, "data")
        if os.path.exists(os.path.join(candidate, "train")):
            data_dir = candidate

    # Datasets
    train_dataset = WikiTextDataset("train", cfg.seq_len, tokenizer, data_dir)
    val_dataset = WikiTextDataset("validation", cfg.seq_len, tokenizer, data_dir)
    test_dataset = WikiTextDataset("test", cfg.seq_len, tokenizer, data_dir)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=True
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    results = {}

    # Parameter alignment check
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Parameter Alignment Check")
        print(f"{'='*60}")
        param_counts = {}
        for mname in args.models:
            block_cls = MODEL_REGISTRY[mname]
            tmp = CausalLM(cfg, block_cls)
            param_counts[mname] = count_parameters(tmp)
            del tmp
        if param_counts:
            target = max(param_counts.values())
            for mname, n in param_counts.items():
                pct = (n - target) / target * 100
                status = "OK" if abs(pct) < 5 else "WARN"
                print(f"  [{status}] {mname:<12} {n:>10,} ({n/1e6:.2f}M)  {pct:+.2f}%")
        print(f"{'='*60}\n", flush=True)

    for model_name in args.models:
        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"  Model: {model_name.upper()}", rank)
        print_rank0(f"{'='*60}", rank)

        block_cls = MODEL_REGISTRY[model_name]
        model = CausalLM(cfg, block_cls).to(device)
        n_params = count_parameters(model)
        print_rank0(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)", rank)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)

        ckpt_path = os.path.join(args.ckpt_dir, f"{model_name}_{args.config}_best.pt")

        if args.eval_only:
            if os.path.exists(ckpt_path):
                raw_model = model.module if hasattr(model, 'module') else model
                raw_model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print_rank0(f"  Loaded: {ckpt_path}", rank)
            else:
                print_rank0(f"  [WARN] No checkpoint at {ckpt_path}, skipping.", rank)
                continue
        else:
            raw_model = model.module if hasattr(model, 'module') else model
            optimizer = torch.optim.AdamW(
                raw_model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay, eps=1e-7
            )
            total_steps = len(train_loader) * args.epochs
            warmup_steps = min(args.warmup_steps, total_steps // 5)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            best_val_ppl = float("inf")
            best_epoch = -1
            train_start = time.time()

            for epoch in range(1, args.epochs + 1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                t0 = time.time()
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, scheduler,
                    device, epoch, rank, args.grad_clip, args.grad_accum_steps
                )
                train_ppl = math.exp(min(train_loss, 20))

                val_loss, val_ppl, val_bpc = evaluate(model, val_loader, device)
                epoch_time = time.time() - t0

                print_rank0(
                    f"  Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | "
                    f"Train Loss {train_loss:.4f} PPL {train_ppl:.2f} | "
                    f"Val Loss {val_loss:.4f} PPL {val_ppl:.2f} BPC {val_bpc:.4f}",
                    rank
                )

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_epoch = epoch
                    if rank == 0:
                        raw_m = model.module if hasattr(model, 'module') else model
                        torch.save(raw_m.state_dict(), ckpt_path)
                        print_rank0(f"  >> New best! Saved to {ckpt_path}", rank)

                if world_size > 1:
                    dist.barrier()

            total_time = time.time() - train_start
            print_rank0(f"\n  Best val PPL: {best_val_ppl:.2f} at epoch {best_epoch}", rank)
            print_rank0(f"  Total training time: {total_time/3600:.2f} hours", rank)

            # Reload best checkpoint for final test
            if rank == 0:
                raw_m = model.module if hasattr(model, 'module') else model
                raw_m.load_state_dict(torch.load(ckpt_path, map_location=device))
            if world_size > 1:
                dist.barrier()

        # Final test evaluation
        test_loss, test_ppl, test_bpc = evaluate(model, test_loader, device)
        print_rank0(f"\n  *** Test Results for {model_name.upper()} ***", rank)
        print_rank0(f"      Loss: {test_loss:.4f}", rank)
        print_rank0(f"      PPL:  {test_ppl:.2f}", rank)
        print_rank0(f"      BPC:  {test_bpc:.4f}", rank)

        results[model_name] = {
            "params": n_params,
            "params_M": round(n_params / 1e6, 2),
            "test_loss": round(test_loss, 4),
            "test_ppl": round(test_ppl, 2),
            "test_bpc": round(test_bpc, 4),
            "config": args.config,
            "world_size": world_size,
            "per_gpu_batch": args.batch_size,
            "effective_batch": args.batch_size * world_size * args.grad_accum_steps,
        }

        del model
        if not args.eval_only:
            del optimizer, scheduler
        torch.cuda.empty_cache()

    # Summary
    if rank == 0:
        print(f"\n{'='*60}")
        dataset_name = data_dir if data_dir else "WikiText-103 fallback"
        print(f"  SUMMARY ({dataset_name}, config={args.config}, {world_size} GPUs)")
        print(f"{'='*60}")
        print(f"  {'Model':<12} {'Params':>10} {'Test PPL':>10} {'Test BPC':>10}")
        print(f"  {'-'*44}")
        for name, r in results.items():
            print(f"  {name.upper():<12} {r['params_M']:>8.2f}M "
                  f"{r['test_ppl']:>10.2f} {r['test_bpc']:>10.4f}")

        results_path = os.path.join(args.ckpt_dir, f"results_{args.config}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {results_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
