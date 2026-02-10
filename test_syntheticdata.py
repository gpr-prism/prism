import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


# =============================================================================
# 1. Comprehensive task generator (keep original logic)
# =============================================================================
class ComprehensiveTaskGenerator(Dataset):
    def __init__(self, task_type='mqar', size=10000, seed=None, **kwargs):
        self.size = size
        self.task_type = task_type
        self.kwargs = kwargs
        if seed is not None:
            np.random.seed(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.task_type == 'mqar':
            return self.generate_mqar(**self.kwargs)
        elif self.task_type == 'poly_recall':
            return self.generate_poly_recall(**self.kwargs)
        elif self.task_type == 'needle':
            return self.generate_needle(**self.kwargs)
        elif self.task_type == 'multihop':
            return self.generate_multihop(**self.kwargs)
        elif self.task_type == 'variable_tracking':
            return self.generate_variable_tracking(**self.kwargs)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    # --- Task Logic ---
    def generate_mqar(self, seq_len=512, vocab_size=1000, num_pairs=16):
        input_seq = np.random.randint(10, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100
        keys = np.random.choice(np.arange(10, vocab_size), size=num_pairs, replace=False)
        values = np.random.choice(np.arange(10, vocab_size), size=num_pairs, replace=True)
        max_pos = seq_len // 2
        positions = np.random.choice(np.arange(0, max_pos, 2), size=num_pairs, replace=False)
        for i, pos in enumerate(positions):
            input_seq[pos], input_seq[pos + 1] = keys[i], values[i]
        q_idx = np.random.randint(0, num_pairs)
        input_seq[-2], input_seq[-1], labels[-1] = 1, keys[q_idx], values[q_idx]
        return torch.tensor(input_seq), torch.tensor(labels)

    def generate_poly_recall(self, seq_len=128, vocab_size=1000, rank_difficulty=4):
        input_seq = np.random.randint(10, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100
        keys = np.random.choice(np.arange(10, vocab_size), size=rank_difficulty, replace=False)
        values = np.random.choice(np.arange(10, vocab_size), size=rank_difficulty, replace=True)
        start_pos = np.random.randint(0, seq_len - rank_difficulty * 2 - 5)
        input_seq[start_pos] = 2
        for i in range(rank_difficulty):
            input_seq[start_pos + 1 + i * 2] = keys[i]
            input_seq[start_pos + 2 + i * 2] = values[i]
        q_idx = np.random.randint(0, rank_difficulty)
        input_seq[-2], input_seq[-1], labels[-1] = 1, keys[q_idx], values[q_idx]
        return torch.tensor(input_seq), torch.tensor(labels)

    def generate_needle(self, seq_len=512, vocab_size=1000, num_pairs=16):
        input_seq = np.random.randint(10, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100
        keys = np.random.choice(np.arange(10, vocab_size), size=num_pairs, replace=False)
        values = np.random.choice(np.arange(10, vocab_size), size=num_pairs, replace=True)
        occupied = np.zeros(seq_len, dtype=bool)
        occupied[-5:] = True
        for i in range(num_pairs):
            for _ in range(100):
                idx = np.random.randint(0, seq_len - 5)
                if not occupied[idx] and not occupied[idx + 1]:
                    input_seq[idx], input_seq[idx + 1] = keys[i], values[i]
                    occupied[idx] = occupied[idx + 1] = True
                    break
        q_idx = np.random.randint(0, num_pairs)
        input_seq[-2], input_seq[-1], labels[-1] = 1, keys[q_idx], values[q_idx]
        return torch.tensor(input_seq), torch.tensor(labels)

    def generate_multihop(self, seq_len=512, vocab_size=1000, num_hops=3):
        input_seq = np.random.randint(10, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100
        chain = np.random.choice(np.arange(10, vocab_size), size=num_hops + 1, replace=False)
        for i in range(num_hops):
            idx = np.random.randint(0, seq_len - 5)
            input_seq[idx], input_seq[idx + 1] = chain[i], chain[i + 1]
        input_seq[-2], input_seq[-1], labels[-1] = 1, chain[0], chain[-1]
        return torch.tensor(input_seq), torch.tensor(labels)

    def generate_variable_tracking(self, seq_len=512, vocab_size=1000, num_vars=4, updates=16):
        input_seq = np.random.randint(10, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100
        var_keys = np.random.choice(np.arange(10, vocab_size), size=num_vars, replace=False)
        state = {}
        indices = np.random.choice(np.arange(0, seq_len - 10), size=updates, replace=False)
        indices.sort()
        for idx in indices:
            k = np.random.choice(var_keys)
            v = np.random.randint(10, vocab_size)
            input_seq[idx], input_seq[idx + 1] = k, v
            state[k] = v
        valid_keys = list(state.keys())
        q_k = np.random.choice(valid_keys) if valid_keys else var_keys[0]
        tgt = state[q_k] if valid_keys else 0
        input_seq[-2], input_seq[-1], labels[-1] = 1, q_k, tgt
        return torch.tensor(input_seq), torch.tensor(labels)


# =============================================================================
# 2. Basic components
# =============================================================================
class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size - 1, groups=dim)

    def forward(self, x):
        return self.conv(x.transpose(1, 2))[:, :, :-(self.conv.kernel_size[0] - 1)].transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x): return self.net(x)


# =============================================================================
# 3. Core mixers (including new models)
# =============================================================================

# --- 1. Traditional Transformer (Baseline) ---
class TransformerMixer(nn.Module):
    def __init__(self, d_model, n_heads=4, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        B, L, _ = x.shape
        # Causal Mask (Triangular)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # attn_mask expects True for positions to IGNORE
        out, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
        return out


# --- 2. Mamba2 (Pure PyTorch Implementation of SSD) ---
class Mamba2Mixer(nn.Module):
    def __init__(self, d_model, n_heads=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Mamba2 Structure: In-proj expands, then splits
        # Simplified: x -> (z, x, B, C, dt)
        self.in_proj = nn.Linear(d_model, d_model * 2 + n_heads * 2 + n_heads)
        self.dt_bias = nn.Parameter(torch.rand(n_heads))
        self.A_log = nn.Parameter(torch.randn(n_heads))  # Parameter A is learnable
        self.D = nn.Parameter(torch.ones(n_heads))

        self.conv = CausalConv1d(d_model, kernel_size=3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)  # Mamba uses norm inside block

    def forward(self, u):
        B, L, _ = u.shape
        H, d = self.n_heads, self.d_head

        # 1. Projections
        # split: z(gate), x, B, C, dt
        # Simplify projection logic to roughly match parameter count
        z, x, B_ssm, C_ssm, dt = self.in_proj(u).split([
            self.d_model, self.d_model, self.n_heads, self.n_heads, self.n_heads
        ], dim=-1)

        # 2. Convolution & Gate
        x = self.conv(x)
        x = F.silu(x)

        # 3. SSD Algorithm (State Space Duality)
        # Reshape for multi-head: [B, L, H, d]
        x_reshaped = x.view(B, L, H, d)

        # Discretize A: Input-dependent decay
        # A_t = -exp(dt) (softplus logic usually)
        dt = F.softplus(dt + self.dt_bias)  # [B, L, H]
        A = -torch.exp(self.A_log)  # Scalar A parameter
        decay = torch.exp(dt * A)  # [B, L, H] input-dependent decay rate

        # B and C projection
        B_ssm = B_ssm.view(B, L, H, 1)
        C_ssm = C_ssm.view(B, L, H, 1)

        # Parallel Scan (Cumulative Sum in Log space)
        # h_t = sum_{i} (prod_{k=i+1}^t decay_k) * (B_i * x_i)
        # log_acc = cumsum(log(decay))
        log_decay = torch.log(decay + 1e-6)
        log_cum = torch.cumsum(log_decay, dim=1)  # [B, L, H]

        # KV Calculation
        # V = B * x
        V = B_ssm * x_reshaped  # [B, L, H, d]

        # Matrix mult simulation using cumsum
        # This is a simplified "1-semiseparable" multiplication
        # Standard Linear Attention trick: exp(S_t - S_i)

        # Prepare for causal integration
        # We need sum(exp(L_t - L_i) * V_i)
        # = exp(L_t) * sum(exp(-L_i) * V_i)

        exp_L = torch.exp(log_cum).unsqueeze(-1)  # [B, L, H, 1]
        exp_neg_L = torch.exp(-log_cum).unsqueeze(-1)

        # Cumsum
        S = torch.cumsum(exp_neg_L * V, dim=1)
        y = exp_L * S

        # Readout: y * C
        y = y * C_ssm  # [B, L, H, d]

        # Add skip connection (D term)
        y = y + x_reshaped * self.D.view(1, 1, H, 1)

        # 4. Output Gate (SwiGLU style)
        y = y.reshape(B, L, -1)
        y = y * F.silu(z)

        return self.out_proj(self.norm(y))


# --- 3. Mixture of Memory (MoM) ---
class MoMMixer(nn.Module):
    def __init__(self, d_model, n_heads=4, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Standard Linear Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Router: Decides contribution of each memory (head) per token
        self.router = nn.Linear(d_model, n_heads)

        self.conv = CausalConv1d(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head

        x_conv = F.silu(self.conv(x))

        # 1. Compute Memory Updates (Linear Attention Heads)
        q = self.q_proj(x_conv).view(B, L, H, d)
        k = self.k_proj(x_conv).view(B, L, H, d)
        v = self.v_proj(x_conv).view(B, L, H, d)

        q, k = F.elu(q) + 1.0, F.elu(k) + 1.0

        # State Update (Parallel Scan)
        kv = torch.einsum('blhd, blhm -> blhdm', k, v)
        kv_cumsum = torch.cumsum(kv, dim=1)

        # Retrieve from all memories
        # y_heads: [B, L, H, d]
        y_heads = torch.einsum('blhdm, blhd -> blhm', kv_cumsum, q)

        # 2. Dynamic Routing (Mixture)
        # Each token decides how much to trust each memory slot
        # weights: [B, L, H]
        router_logits = self.router(x_conv)
        weights = F.softmax(router_logits, dim=-1).unsqueeze(-1)  # [B, L, H, 1]

        # Weighted Sum
        # y: [B, L, H, d] * [B, L, H, 1] -> sum over H -> [B, L, d] -> [B, L, D]
        y = (y_heads * weights).reshape(B, L, D)

        return self.out_proj(y)


# --- Existing Mixers (Keep unchanged) ---
class LinearAttnMixer(nn.Module):
    def __init__(self, d_model, n_heads=4, **kwargs):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head
        qkv = self.qkv_proj(x).view(B, L, 3, H, d)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
        kv = torch.einsum('blhd, blhm -> blhdm', k, v)
        y = torch.einsum('blhdm, blhd -> blhm', torch.cumsum(kv, dim=1), q)
        return self.out_proj(y.reshape(B, L, D))


class PRISMMixer(nn.Module):
    def __init__(self, d_model, n_heads=4, L_refinement=2, **kwargs):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.L = L_refinement
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, n_heads)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.conv = CausalConv1d(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ref_k = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.L)])
        self.ref_p = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.L)])
        self.ref_lambda = nn.ModuleList([nn.Linear(d_model, n_heads) for _ in range(self.L)])
        nn.init.constant_(self.beta_proj.bias, -4.0)
        nn.init.constant_(self.gate_proj.bias, 2.0)

    def forward(self, x):
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head
        xc = F.silu(self.conv(x))
        q, v = self.q_proj(xc).view(B, L, H, d), self.v_proj(xc).view(B, L, H, d)
        alpha, beta = torch.sigmoid(self.gate_proj(xc)).view(B, L, H, 1, 1), torch.sigmoid(self.beta_proj(xc)).view(B,
                                                                                                                    L,
                                                                                                                    H,
                                                                                                                    1,
                                                                                                                    1)
        ref_k = [l(xc).view(B, L, H, d) for l in self.ref_k]
        ref_p = [l(xc).view(B, L, H, d) for l in self.ref_p]
        ref_lam = [l(xc).view(B, L, H, 1, 1) for l in self.ref_lambda]

        correction = xc.view(B, L, H, d)
        R_hat = v - correction
        S = torch.zeros(B, H, d, d, device=x.device)
        outputs = []
        for t in range(L):
            r_loc, B_acc = R_hat[:, t].clone(), 0
            for l in range(self.L):
                delta = F.gelu(ref_p[l][:, t] * r_loc)
                B_acc = B_acc + torch.einsum('bhd, bhm -> bhdm', delta, ref_k[l][:, t]) * ref_lam[l][:, t]
                r_loc = r_loc - delta
            k_decay = ref_k[0][:, t]
            decay_mat = torch.einsum('bhd, bhm -> bhdm', torch.einsum('bhdm, bhm -> bhd', S, k_decay), k_decay)
            S = alpha[:, t] * (S - beta[:, t] * decay_mat) + B_acc
            outputs.append(torch.einsum('bhdm, bhd -> bhm', S, q[:, t]))
        return self.out_proj(torch.stack(outputs, dim=1).reshape(B, L, D))


# =============================================================================
# 4. Model wrapper
# =============================================================================
class UniversalModel(nn.Module):
    def __init__(self, model_type, vocab_size, d_model, num_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if model_type == 'Transformer':
                mixer = TransformerMixer(d_model, n_heads)
            elif model_type == 'Mamba2':
                mixer = Mamba2Mixer(d_model, n_heads)
            elif model_type == 'MoM':
                mixer = MoMMixer(d_model, n_heads)
            elif model_type == 'LA':
                mixer = LinearAttnMixer(d_model, n_heads)
            elif model_type == 'PRISM':
                mixer = PRISMMixer(d_model, n_heads)
            else:
                raise ValueError(f"Unknown: {model_type}")

            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model), 'mixer': mixer,
                'ln2': nn.LayerNorm(d_model), 'mlp': MLP(d_model)
            }))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for l in self.layers:
            x = x + l['mixer'](l['ln1'](x))
            x = x + l['mlp'](l['ln2'](x))
        return self.head(self.norm(x))


# =============================================================================
# 5. Training and evaluation
# =============================================================================
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits.view(-1, logits.size(-1)), by.view(-1))
            total_loss += loss.item() * bx.size(0)
            mask = (by != -100)
            total_correct += (torch.argmax(logits, -1)[mask] == by[mask]).sum().item()
            total_tokens += mask.sum().item()
    return total_correct / (total_tokens + 1e-6), math.exp(min(total_loss / len(loader.dataset), 20))


def train_model(model, train_loader, device, steps=500):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss_hist, acc_hist = [], []
    iter_loader = iter(train_loader)

    for _ in range(steps):
        try:
            bx, by = next(iter_loader)
        except:
            iter_loader = iter(train_loader); bx, by = next(iter_loader)

        model.train()
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits.view(-1, logits.size(-1)), by.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_hist.append(loss.item())

        with torch.no_grad():
            mask = (by != -100)
            acc_hist.append((torch.argmax(logits, -1)[mask] == by[mask]).float().mean().item())

    return loss_hist, acc_hist


def smooth(data, weight=0.9):
    last, smoothed = data[0], []
    for p in data:
        last = last * weight + (1 - weight) * p
        smoothed.append(last)
    return smoothed


# =============================================================================
# 6. Main
# =============================================================================
def run_full_benchmark():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {DEVICE}")

    TASKS = ['mqar', 'poly_recall', 'variable_tracking']
    # Include Transformer, Mamba2, MoM
    MODELS = ['Transformer', 'Mamba2', 'MoM', 'LA', 'PRISM']

    VOCAB, DIM, LAYERS, STEPS, BS, HEADS = 200, 64, 2, 400, 32, 4
    results, final_metrics = {t: {} for t in TASKS}, {t: {} for t in TASKS}

    for task in TASKS:
        print(f"\n=== Task: {task} ===")
        # Shorten Poly Recall length to speed up the demo
        sl = 64 if task == 'poly_recall' else 128
        train_ds = ComprehensiveTaskGenerator(task, STEPS * BS, vocab_size=VOCAB, seq_len=sl)
        test_ds = ComprehensiveTaskGenerator(task, 500, vocab_size=VOCAB, seed=42, seq_len=sl)
        train_loader, test_loader = DataLoader(train_ds, BS), DataLoader(test_ds, BS)

        for m_name in MODELS:
            print(f"Training {m_name}...")
            model = UniversalModel(m_name, VOCAB, DIM, LAYERS, HEADS)
            loss, acc = train_model(model, train_loader, DEVICE, STEPS)
            test_acc, test_ppl = evaluate(model, test_loader, DEVICE)

            results[task][m_name] = {'loss': loss, 'acc': acc}
            final_metrics[task][m_name] = {'acc': test_acc, 'ppl': test_ppl}
            print(f"  -> Test Acc: {test_acc:.4f} | PPL: {test_ppl:.4f}")

    print("\n" + "=" * 70)
    print(f"{'FINAL BENCHMARK SUMMARY':^70}")
    print("=" * 70)

    for task in TASKS:
        print(f"\n>>> Task: {task.upper()}")
        print("-" * 70)
        # Header
        print(f"{'Model':<15} | {'Test Acc':<10} | {'Test PPL':<10} | {'Final Train Loss':<15}")
        print("-" * 70)

        # Output per model
        for m_name in MODELS:
            # Get test-set metrics
            test_acc = final_metrics[task][m_name]['acc']
            test_ppl = final_metrics[task][m_name]['ppl']

            # Compute average training loss over the last 20 steps to check convergence
            train_loss_hist = results[task][m_name]['loss']
            final_train_loss = np.mean(train_loss_hist[-20:]) if train_loss_hist else 0.0

            # Format output
            print(f"{m_name:<15} | {test_acc:<10.4f} | {test_ppl:<10.4f} | {final_train_loss:<15.4f}")

        print("-" * 70)


if __name__ == "__main__":
    run_full_benchmark()
