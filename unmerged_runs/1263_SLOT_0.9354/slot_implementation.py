    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        """Return hidden states (bsz, seq_len, dim) after final norm, before projection."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        return self.final_norm(x)
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self.forward_hidden(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte
def eval_val_sliding_slot(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window eval with SLOT: per-sample delta + logit bias optimization."""
    seq_len = eval_seq_len or args.train_seq_len
    slot_steps = args.slot_steps
    slot_lr = args.slot_lr
    slot_lr_min = args.slot_lr_min
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_hidden = torch.compile(base_model.forward_hidden, dynamic=False, fullgraph=True)
    proj_w = (base_model.tok_emb.weight if base_model.tie_embeddings
              else base_model.lm_head.weight).detach().float()
    softcap = base_model.logit_softcap
    for bi in range(0, len(my_windows), batch_seqs):
        batch_ws = my_windows[bi:bi + batch_seqs]
        bsz = len(batch_ws)
        x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        wlens: list[int] = []
        for i, ws in enumerate(batch_ws):
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws; wlens.append(wlen)
            chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
            x_batch[i, :wlen] = chunk[:-1]; y_batch[i, :wlen] = chunk[1:]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = compiled_hidden(x_batch)
        hidden_f = hidden.float()
        mask = torch.zeros(bsz, seq_len, device=device)
        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            s = 0 if ws == 0 else max(wlen - stride, 0)
            mask[i, s:wlen] = 1.0
        valid_count = mask.sum()
        delta = torch.zeros(bsz, 1, hidden_f.size(-1), device=device, dtype=torch.float32, requires_grad=True)
        logit_bias = torch.zeros(bsz, 1, proj_w.size(0), device=device, dtype=torch.float32, requires_grad=True)
        slot_opt = torch.optim.AdamW([delta, logit_bias], lr=slot_lr)
        targets_flat = y_batch.reshape(-1)
        for _step in range(slot_steps):
            _lr = slot_lr_min + 0.5 * (slot_lr - slot_lr_min) * (1 + math.cos(math.pi * _step / slot_steps))
            for _pg in slot_opt.param_groups: _pg['lr'] = _lr
            h = hidden_f + delta
            logits_proj = F.linear(h, proj_w) + logit_bias
            logits = softcap * torch.tanh(logits_proj / softcap)
            nll_opt = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      targets_flat, reduction="none").reshape(bsz, seq_len)
            slot_loss = (nll_opt * mask).sum() / valid_count
            slot_opt.zero_grad(); slot_loss.backward(); slot_opt.step()
        with torch.no_grad():
            h = hidden_f + delta
            logits_proj = F.linear(h, proj_w) + logit_bias
            logits = softcap * torch.tanh(logits_proj / softcap)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                  targets_flat, reduction="none").reshape(bsz, seq_len)
        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            s = 0 if ws == 0 else max(wlen - stride, 0)
            scored_nll = nll[i, s:wlen].to(torch.float64)
            loss_sum += scored_nll.sum(); token_count += float(wlen - s)
            tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    base_model.train()
    return val_loss, val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
