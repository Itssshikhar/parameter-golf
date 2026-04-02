# --- WARP: Word boundary detection, attention bias, and output logit bias ---

def compute_word_boundary_maps(input_ids: Tensor, has_leading_space_lut: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Detect word boundaries from SentencePiece tokens. All ops are torch.compile friendly."""
    B, T = input_ids.shape
    device = input_ids.device
    is_word_start = has_leading_space_lut[input_ids]
    is_word_start[:, 0] = True
    word_ids = is_word_start.long().cumsum(dim=1) - 1
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    word_last_pos = torch.zeros(B, T, dtype=torch.long, device=device)
    word_last_pos.scatter_(1, word_ids, positions)
    reversed_positions = (T - 1) - positions
    word_first_pos_rev = torch.zeros(B, T, dtype=torch.long, device=device)
    word_first_pos_rev.scatter_(1, word_ids, reversed_positions)
    word_first_pos = (T - 1) - word_first_pos_rev
    prev_word_ids = (word_ids - 1).clamp(min=0)
    last_tok_of_prev_word = word_last_pos.gather(1, prev_word_ids)
    last_tok_of_prev_word[word_ids == 0] = 0
    first_tok_of_curr_word = word_first_pos.gather(1, word_ids)
    return word_ids, last_tok_of_prev_word, first_tok_of_curr_word

class WordLengthEmbed(nn.Module):
    def __init__(self, model_dim, max_len=13):
        super().__init__()
        self.embed = nn.Embedding(max_len, model_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, word_length):
        return self.embed(word_length) * self.scale

class WARPAttn(nn.Module):
    """Word position embeddings added to BOTH Q and K in attention."""
    def __init__(self, head_dim: int, num_layers: int, max_word_pos: int = 8):
        super().__init__()
        self.q_word_pos_embed = nn.Embedding(max_word_pos, head_dim)
        self.k_word_pos_embed = nn.Embedding(max_word_pos, head_dim)
        nn.init.normal_(self.q_word_pos_embed.weight, std=0.01)
        nn.init.normal_(self.k_word_pos_embed.weight, std=0.01)
        # Per-layer learned scale (num_layers layers)
        self.layer_scales = nn.Parameter(torch.full((num_layers,), 0.15))

    def get_q_bias(self, position_in_word: Tensor, layer_idx: int) -> Tensor:
        scale = self.layer_scales[layer_idx]
        return self.q_word_pos_embed(position_in_word) * scale

    def get_k_bias(self, position_in_word: Tensor, layer_idx: int) -> Tensor:
        scale = self.layer_scales[layer_idx]
        return self.k_word_pos_embed(position_in_word) * scale

class WARPOutput(nn.Module):
    """Classify word type from final hidden states, produce direct logit bias."""
    def __init__(self, model_dim: int, num_types: int, vocab_size: int):
        super().__init__()
        self.classifier_fc1 = CastedLinear(model_dim, 192, bias=False)
        self.classifier_fc2 = CastedLinear(192, num_types, bias=False)
        # DIRECT logit bias per type -- initialized to zeros, gets gradient from CE loss
        self.type_vocab_bias = nn.Parameter(torch.zeros(num_types, vocab_size))

    def forward(self, x_flat: Tensor) -> tuple[Tensor, Tensor]:
        """
        x_flat: [B*T, D] -- final normalized hidden states (after final_norm, reshaped)
        Returns:
            vocab_bias: [B*T, vocab_size] -- per-token logit bias
            type_logits: [B*T, num_types] -- raw type logits (for logging)
        """
        h = F.rms_norm(x_flat.float(), (x_flat.size(-1),))
        h = F.leaky_relu(self.classifier_fc1(h.to(x_flat.dtype)), negative_slope=0.5)
        type_logits = self.classifier_fc2(h).float()
        type_probs = F.softmax(type_logits, dim=-1)
        # Direct logit bias: type_probs @ type_vocab_bias
        vocab_bias = type_probs @ self.type_vocab_bias.float()  # [B*T, vocab_size]
        return vocab_bias, type_logits

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        # WARP-Output params
        num_word_types: int = 64,
        has_leading_space_lut: Tensor | None = None,
        # unused legacy args kept for call-site compat
        dtg: bool = False,
        gated_attention: bool = False,
        value_residual: bool = False,
    ):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        # Parameter banks: contiguous 3D tensors for batched optimizer
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self.num_layers = num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    ln_scale=ln_scale,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim_ve = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        # --- WARP-Attn: shared across all layers ---
        head_dim_warp = model_dim // num_heads
        self.warp_attn = WARPAttn(head_dim_warp, num_layers, max_word_pos=8)
        for i, block in enumerate(self.blocks):
            block.attn.warp_attn = self.warp_attn
            block.attn.layer_idx = i
        # --- WARP-Output module ---
        self.warp_output = WARPOutput(model_dim, num_word_types, vocab_size)
        # --- Word-Length Embedding ---
        self.word_length_embed = WordLengthEmbed(model_dim, max_len=13)
        # Register has_leading_space_lut as buffer for word boundary detection
        if has_leading_space_lut is not None:
            self.register_buffer("has_leading_space_lut", has_leading_space_lut, persistent=False)
