import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import time
from tqdm import tqdm

torch.backends.cudnn.enabled = False


# =============================================================================
# 0.  RoPE 
# =============================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())

    def forward(self, x, seq_len):
        return self.cos[:seq_len, :].unsqueeze(0).unsqueeze(2), \
            self.sin[:seq_len, :].unsqueeze(0).unsqueeze(2)


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ComprehensiveTaskGenerator(Dataset):
    def __init__(self, task_type='mqar', size=10000, seed=None, **kwargs):
        self.size = size
        self.task_type = task_type
        self.kwargs = kwargs
        if seed is not None: np.random.seed(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.task_type == 'mqar':
            return self.generate_mqar(**self.kwargs)
        elif self.task_type == 'poly_recall':
            return self.generate_poly_recall(**self.kwargs)
        elif self.task_type == 'variable_tracking':
            return self.generate_variable_tracking(**self.kwargs)
        elif self.task_type == 'local_xor':
            return self.generate_local_xor(**self.kwargs)
        elif self.task_type == 'silent_gate':
            return self.generate_silent_gate(**self.kwargs)
        elif self.task_type == 'mux_logic':
            return self.generate_mux_logic(**self.kwargs)
        # --- added higher-order nonlinear tasks ---
        elif self.task_type == 'parity_check':
            return self.generate_n_bit_parity(**self.kwargs)
        elif self.task_type == 'modulo_add':
            return self.generate_modulo_addition(**self.kwargs)
        elif self.task_type == 'palindrome':
            return self.generate_palindrome(**self.kwargs)
        else:
            raise ValueError(f"Unknown task: {self.task_type}")

    def generate_mqar(self, seq_len=64, vocab_size=200, num_pairs=2):
        all_tokens = np.arange(10, vocab_size)
        keys = np.random.choice(all_tokens, size=num_pairs, replace=False)
        values = np.random.choice(all_tokens, size=num_pairs, replace=True)
        noise_pool = np.setdiff1d(all_tokens, keys)
        if len(noise_pool) == 0: noise_pool = all_tokens

        input_seq = np.random.choice(noise_pool, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100

        max_pos = seq_len // 2
        positions = np.random.choice(np.arange(0, max_pos, 2), size=num_pairs, replace=False)
        for i, pos in enumerate(positions):
            input_seq[pos], input_seq[pos + 1] = keys[i], values[i]

        q_idx = np.random.randint(0, num_pairs)
        input_seq[-2], input_seq[-1], labels[-1] = 1, keys[q_idx], values[q_idx]
        return torch.tensor(input_seq), torch.tensor(labels)

    def generate_poly_recall(self, seq_len=64, vocab_size=200, rank_difficulty=2):
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

    def generate_variable_tracking(self, seq_len=64, vocab_size=200, num_vars=3, updates=3):
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

    def generate_local_xor(self, seq_len=128, vocab_size=1000):
        """
        Local XOR
        """
        # define special Token
        TOK_QUERY = 1
        TOK_OP_XOR = 2
        VAL_TRUE = 3
        VAL_FALSE = 4

        input_seq = np.random.randint(5, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100

        # 1. generate A, B
        val_a = np.random.randint(5, vocab_size-10)
        val_b = np.random.randint(5, vocab_size-10)

        # logic
        result = VAL_TRUE if (val_a % 2 != val_b % 2) else VAL_FALSE

        # 2. place logic [A, B, OP_XOR]
        burst_len = 3
        start_pos = np.random.randint(0, seq_len - burst_len - 5)

        input_seq[start_pos] = val_a
        input_seq[start_pos + 1] = val_b
        input_seq[start_pos + 2] = TOK_OP_XOR

        # 3. Query
        input_seq[-1] = TOK_QUERY
        labels[-1] = result

        return torch.tensor(input_seq), torch.tensor(labels)

    def generate_silent_gate(self, seq_len=32, vocab_size=20):
        # --- 1. define token ---
        # [0]: PAD / NULL
        # [1]: TRIGGER_ON
        # [2]: TRIGGER_OFF
        # [3]: QUERY_TOKEN
        # [4 ~ 10]: data (Key/Value)
        # [11 ~ 19]: noise (Noise)
        
        TOK_NULL = 0
        TOK_ON = 1
        TOK_OFF = 2
        TOK_QUERY = 3
        
        DATA_START = 4
        DATA_END = vocab_size-5  # 4,5,6,7,8,9,10
        
        NOISE_START = 0
        NOISE_END = vocab_size # 11 ~ 19
        
        input_seq = np.random.randint(NOISE_START, NOISE_END, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100 # ignore loss by default
        
        key = np.random.randint(DATA_START, DATA_END)
        val = np.random.randint(DATA_START, DATA_END)
        
        is_active = np.random.rand() > 0.5
        
        if is_active:
            trigger = TOK_ON
            target = val
        else:
            trigger = TOK_OFF
            target = TOK_NULL
            
        start_pos = np.random.randint(0, seq_len - 5)
        
        input_seq[start_pos]     = trigger
        input_seq[start_pos + 1] = key
        input_seq[start_pos + 2] = val

        input_seq[-2] = TOK_QUERY
        input_seq[-1] = key
        
        labels[-1] = target
        
        return torch.tensor(input_seq), torch.tensor(labels)


    def generate_mux_logic(self, seq_len=128, vocab_size=1000):

        TOK_QUERY = 1
        TOK_OP_MUX = 2

        input_seq = np.random.randint(0, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100

        # Selector: 0 (Token 3) or 1 (Token 4)
        selector_logic = np.random.randint(0, 2)
        selector_token = 3 if selector_logic == 0 else 4

        ch0_val = np.random.randint(5, vocab_size//2)
        ch1_val = np.random.randint(vocab_size//2, vocab_size)

        target = ch0_val if selector_logic == 0 else ch1_val

        #  [Selector, Ch0, Ch1, OP_MUX]
        start_pos = np.random.randint(0, seq_len - 10)
        input_seq[start_pos] = selector_token
        input_seq[start_pos + 1] = ch0_val
        input_seq[start_pos + 2] = ch1_val
        input_seq[start_pos + 3] = TOK_OP_MUX

        # Query
        input_seq[-1] = TOK_QUERY
        labels[-1] = target

        return torch.tensor(input_seq), torch.tensor(labels)

    def generate_n_bit_parity(self, vocab_size, seq_len=64, n=3):
        VAL_FALSE = 0
        VAL_TRUE = 1
        TOK_QUERY = 2
        TOK_OP_NBIT =  3

        input_seq = np.random.randint(4, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100
        
        bits = np.random.randint(4, 8, size=n)
        target = np.sum(bits) % 2
        
        start_pos = np.random.randint(0, seq_len - n - 5)
        input_seq[start_pos+n] = TOK_OP_NBIT
        for i in range(n):
            input_seq[start_pos + i] = bits[i]
            
        # Query
        input_seq[-1] = TOK_QUERY # Query Token
        labels[-1] = target
        
        return torch.tensor(input_seq), torch.tensor(labels)
    
    def generate_modulo_addition(self, vocab_size, seq_len=64, modulus=16):
        """
        input：A, B (0 ~ modulus-1)。
        output：(A + B) % modulus。
        """
        TOK_VAL = vocab_size-2
        TOK_QUERY = vocab_size-1
        input_seq = np.random.randint(0, vocab_size-2, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100
        
        val_a = np.random.randint(0, modulus)
        val_b = np.random.randint(0, modulus)
        target = (val_a + val_b) % modulus
        
        # [A, B]
        start_pos = np.random.randint(0, seq_len - 5)
        input_seq[start_pos + 2] = TOK_VAL
        input_seq[start_pos ] = val_a
        input_seq[start_pos + 1] = val_b
        
        input_seq[-1] = TOK_QUERY # Query
        labels[-1] = target
        
        return torch.tensor(input_seq), torch.tensor(labels)

    def generate_palindrome(self, vocab_size, seq_len=64):
        """
        input [A, B, C]
        output 1 if A == C else 0
        noise：[A, B, A] vs [A, A, B] vs [B, A, A]
        """
        VAL_FIRST = 0
        VAL_SECOND = 1
        VAL_THIRD = 2
        TOK_VAL = 3
        TOK_QUERY = 4

        input_seq = np.random.randint(5, vocab_size, size=seq_len)
        labels = np.zeros(seq_len, dtype=np.int64) - 100
        
        is_palindrome = np.random.rand() > 0.5
        is_D = np.random.rand() > 0.5
        
        val_a = np.random.randint(5, vocab_size)
        val_b = np.random.randint(5, vocab_size) 
        if is_palindrome:
            if is_D:
                choices = [x for x in range(5, vocab_size) if x != val_a]
                val_c = np.random.choice(choices) 
                val_d = val_a
                target = 1
            else:
                val_c = val_a
                choices = [x for x in range(5, vocab_size) if x != val_a]
                val_d = np.random.choice(choices)
                target = 1
        else:
            choices = [x for x in range(5, vocab_size) if x != val_a]
            val_c = np.random.choice(choices)
            val_d = np.random.choice(choices)
            target = 0
            
        start_pos = np.random.randint(0, seq_len - 5)
        input_seq[start_pos + 4] = TOK_VAL
        input_seq[start_pos] = val_a
        input_seq[start_pos + 1] = val_b
        input_seq[start_pos + 2] = val_c
        input_seq[start_pos + 3] = val_d
        
        input_seq[-1] = TOK_QUERY
        labels[-1] = target
        
        return torch.tensor(input_seq), torch.tensor(labels)

class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size - 1, groups=dim)

    def forward(self, x):
        return self.conv(x.transpose(1, 2))[:, :, :-(self.conv.kernel_size[0] - 1)].transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x): return self.net(x)


def get_decay_mask(decay_rates, L, device):
    log_decay = decay_rates.transpose(1, 2)
    log_cum = torch.cumsum(log_decay, dim=-1)
    mask = torch.exp(log_cum.unsqueeze(-1) - log_cum.unsqueeze(-2))
    causal = torch.tril(torch.ones(L, L, device=device), diagonal=0)
    return mask * causal.view(1, 1, L, L)


class GroupedLinear(nn.Module):
    def __init__(self, d_in, d_out, groups=2):
        super().__init__()
        self.proj = nn.Conv1d(d_in, d_out, kernel_size=1, groups=groups, bias=False)
    def forward(self, x):
        # x: [B, L, D] -> [B, D, L] -> Conv -> [B, D_out, L] -> [B, L, D_out]
        return self.proj(x.transpose(1, 2)).transpose(1, 2)

# (for PRISM)
class LowRankLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.down = nn.Linear(d_in, d_in//4, bias=False)
        self.up = nn.Linear(d_in//4, d_out, bias=False)
    def forward(self, x):
        return self.up(self.down(x))

# --- 1. Transformer (RoPE + Conv) ---
class TransformerMixer(nn.Module):
    def __init__(self, d_model, n_heads=4, **kwargs):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.conv = CausalConv1d(d_model, kernel_size=3)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head)

    def forward(self, x):
        B, L, _ = x.shape
        H, d = self.n_heads, self.d_head
        x_mixed = F.silu(self.conv(x))
        q = self.q_proj(x_mixed).view(B, L, H, d)
        k = self.k_proj(x_mixed).view(B, L, H, d)
        v = self.v_proj(x_mixed).view(B, L, H, d)
        cos, sin = self.rope(v, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(y)


# --- 2. Gated DeltaNet (GDN) [CORRECTED] ---
# # --- 5. MoM (Vectorized Soft-Gating) ---
class MoMixer(nn.Module):
    def __init__(self, d_model, n_memories=4, topk=2, **kwargs):
        super().__init__()
        self.n_memories = n_memories
        self.topk = min(topk, n_memories)
        self.scale = d_model ** -0.5

        self.router = nn.Linear(d_model, n_memories)
        self.conv = CausalConv1d(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_memories * d_model, bias=False)
        self.v_proj = nn.Linear(d_model, n_memories * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        M = self.n_memories
        x_c = F.silu(self.conv(x))

        # 1. Soft Router
        scores = F.softmax(self.router(x_c), dim=-1).unsqueeze(-1)  # [B, L, M, 1]

        # 2. Projections
        q = (F.elu(self.q_proj(x_c)) + 1.0).view(B, L, 1, D) * self.scale
        k = (F.elu(self.k_proj(x_c)) + 1.0).view(B, L, M, D)
        v = self.v_proj(x_c).view(B, L, M, D)

        # 3. Parallel Scan (Cumsum)
        kv = torch.einsum('blmd, blmn -> blmdn', k, v)
        kv_cum = torch.cumsum(kv, dim=1)

        # 4. Read & Weighted Mix
        y_all = torch.einsum('blmdn, blid -> blmn', kv_cum, q)
        y_mixed = (y_all * scores).sum(dim=2)

        return self.out_proj(y_mixed)


# --- 2. Linear Attention (Einsum + Cumsum) ---
class LinearAttnMixer(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.scale = d_model ** -0.5
        self.conv = CausalConv1d(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = F.silu(self.conv(x))
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Stability Tricks
        q = (F.elu(q) + 1.0) * self.scale
        k = (F.elu(k) + 1.0)

        # Parallel Scan (No Decay)
        kv = torch.einsum('bld, blm -> bldm', k, v)
        kv_cum = torch.cumsum(kv, dim=1)  # [B, L, D, D]
        y = torch.einsum('bldm, bld -> blm', kv_cum, q)

        return self.out(y)

# --- 6. Mamba2 (Parallel SSD) ---
class Mamba2Mixer(nn.Module):
    def __init__(self, d_model, n_heads=4, **kwargs):
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.in_proj = nn.Linear(d_model, d_model * 2 + n_heads * 3)
        self.dt_bias = nn.Parameter(torch.rand(n_heads))
        self.A_log = nn.Parameter(torch.randn(n_heads))
        self.D = nn.Parameter(torch.ones(n_heads))
        self.conv = CausalConv1d(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u):
        B, L, _ = u.shape
        H, d = self.n_heads, self.d_head

        z, x, B_ssm, C_ssm, dt = self.in_proj(u).split(
            [self.d_model, self.d_model, self.n_heads, self.n_heads, self.n_heads], dim=-1
        )
        x = F.silu(self.conv(x))
        x_r = x.view(B, L, H, d)

        dt = F.softplus(dt + self.dt_bias)
        log_decay = dt * -torch.exp(self.A_log)

        # Mask Generation (Parallel)
        log_cum = torch.cumsum(log_decay, dim=1)  # [B, L, H]
        # Mask[t, s] = exp(log_cum[t] - log_cum[s])
        # Expand dims to [B, H, L, L]
        log_cum = log_cum.transpose(1, 2)
        mask = torch.exp(log_cum.unsqueeze(-1) - log_cum.unsqueeze(-2))
        causal = torch.tril(torch.ones(L, L, device=u.device))
        mask = mask * causal.view(1, 1, L, L)

        # SSD Attention
        Q = x_r * B_ssm.view(B, L, H, 1)
        K = x_r * C_ssm.view(B, L, H, 1)

        # Einsum: Q @ K.T -> [B, H, L, L]
        # Correct indices: b:batch, h:head, l:len, k:len_k
        qk = torch.einsum('blhd, bkhd -> bhlk', Q, K)
        qk = qk * mask
        y = torch.einsum('bhlk, bkhd -> blhd', qk, x_r)

        y = y + x_r * self.D.view(1, 1, H, 1)
        out = y.reshape(B, L, -1) * F.silu(z)
        return self.out_proj(self.norm(out))


# --- 7. PRISM (Parallel Refinement + Cumsum) ---
class PRISMMixer(nn.Module):
    def __init__(self, d_model, L_refinement=2, **kwargs):
        super().__init__()
        self.scale = d_model ** -0.5
        self.L = L_refinement

        self.conv = CausalConv1d(d_model) 
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Refinement params
        self.ref_k = nn.ModuleList([LowRankLinear(d_model, d_model) for _ in range(self.L)])
        self.ref_p = nn.ModuleList([LowRankLinear(d_model, d_model) for _ in range(self.L)])
        self.ref_lambda = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(self.L)])

    def forward(self, x):
        B, L, D = x.shape
        xc = F.silu(self.conv(x))

        # Stability
        q = (F.elu(self.q_proj(xc)) + 1.0) * self.scale
        v = self.v_proj(xc)

        # 1. Base Update (Parallel)
        total_update = 0

        # 2. Parallel Refinement Loo
        r_loc = v - xc
        for l in range(self.L):
            kl = self.ref_k[l](xc)
            pl = self.ref_p[l](xc)
            lam = self.ref_lambda[l](xc).unsqueeze(-1)

            delta = F.gelu(pl * r_loc)
            # accumulate update matrix
            update_l = torch.einsum('bld, blm -> bldm', delta, kl)
            total_update = total_update + update_l * lam

            r_loc = r_loc - delta

        # 3. Global Integration (Cumsum)
        S_global = torch.cumsum(total_update, dim=1)

        # 4. Read
        y = torch.einsum('bldm, bld -> blm', S_global, q)
        return self.out_proj(y)



# --- 5. MoM (Vectorized) ---
class MoMMixer(nn.Module):
    def __init__(self, d_model, n_heads=4, **kwargs):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // 2
        self.qkv = GroupedLinear(d_model, 6 * d_model, groups=2)
        self.router = nn.Linear(d_model, n_heads)
        self.conv = CausalConv1d(d_model)
        self.out = GroupedLinear(d_model*2, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head
        x_c = F.silu(self.conv(x))
        q, k, v = self.qkv(x_c).view(B, L, 3, H, d).unbind(2)
        q, k = F.elu(q) + 1, F.elu(k) + 1

        kv = torch.einsum('blhd, blhm -> blhdm', k, v)
        y_heads = torch.einsum('blhdm, blhd -> blhm', torch.cumsum(kv, dim=1), q)
        weights = F.softmax(self.router(x_c), dim=-1).unsqueeze(-1)
        return self.out((y_heads * weights).reshape(B, L, 2*D))


# =============================================================================
# 4. Universal Wrapper
# =============================================================================
class UniversalModel(nn.Module):
    def __init__(self, model_type, vocab_size, d_model, num_layers=2, n_heads=4):
        super().__init__()
        self.model_type = model_type
        self.embed = nn.Embedding(vocab_size, d_model)
        # Pos Embed only for Transformer
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if model_type == 'Transformer':
                mixer = TransformerMixer(d_model, 1)
            elif model_type == 'GDN':
                mixer = GDNMixer(d_model, n_heads)
            elif model_type == 'LA':
                mixer = LinearAttnMixer(d_model)
            elif model_type == 'PRISM':
                mixer = PRISMMixer(d_model, n_heads)
            elif model_type == 'Mamba2':
                mixer = Mamba2Mixer(d_model, n_heads)
            elif model_type == 'MoM':
                mixer = MoMMixer(d_model, 4)
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


def train_and_eval():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {DEVICE}")

    VOCAB, DIM, LAYERS, STEPS, BS, HEADS = 32, 16, 2, 10000, 32, 1

    TASKS = ['mqar', 'poly_recall', 'variable_tracking','parity_check', 'modulo_add', 'palindrome', 'mux_logic','local_xor', 'silent_gate']
    MODELS = ['Transformer', 'LA', 'PRISM', 'MoM']

    for task in TASKS:
        print(f"\n>>> Task: {task.upper()}")
        seq_len = 128 if task not in ['mqar', 'poly_recall', 'variable_tracking'] else 64 # small seq len, otherwise all linear attention models fail to converge
        ds_train = ComprehensiveTaskGenerator(task, STEPS * BS, vocab_size=VOCAB, seq_len=seq_len,seed = 42)
        ds_test = ComprehensiveTaskGenerator(task, 8000, vocab_size=VOCAB, seq_len=seq_len, seed=42)
        loader_train = DataLoader(ds_train, BS)
        loader_test = DataLoader(ds_test, BS)
        dim = DIM if task not in ['mqar', 'poly_recall', 'variable_tracking'] else 128 # large dim for memory task, otherwise all linear attention models fail to converge
        for m_name in MODELS:
            t0 = time.time()
            model = UniversalModel(m_name, VOCAB, dim, LAYERS, HEADS).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss(ignore_index=-100)

            # Tqdm setup
            pbar = tqdm(range(STEPS), desc=f"{m_name}", dynamic_ncols=True)
            iter_loader = iter(loader_train)

            model.train()
            for step in pbar:
                try:
                    bx, by = next(iter_loader)
                except:
                    iter_loader = iter(loader_train); bx, by = next(iter_loader)
                bx, by = bx.to(DEVICE), by.to(DEVICE)

                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits.view(-1, VOCAB), by.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                if step % 50 == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            model.eval()
            correct, total = 0, 0
            total_nll = 0.0 
            total_tokens = 0  

            with torch.no_grad():
                for bx, by in loader_test:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    
                    logits = model(bx)
                    
                    preds = torch.argmax(logits, dim=-1)
                    mask = by != -100
                    correct += (preds[mask] == by[mask]).sum().item()
                    total += mask.sum().item()
                    
                    loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), by.view(-1))
                    
                    valid_loss = loss[by.view(-1) != -100]
                    if len(valid_loss) > 0:
                        total_nll += valid_loss.sum().item()
                        total_tokens += len(valid_loss)

            acc = correct / (total + 1e-6)

            if total_tokens > 0:
                avg_nll = total_nll / total_tokens
                ppl = torch.exp(torch.tensor(avg_nll)).item()
            else:
                ppl = float('inf')

            print(f"  > {m_name:<12} | Acc: {acc:.4f} | PPL: {ppl:.4f} | Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    train_and_eval()
