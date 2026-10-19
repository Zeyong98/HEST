"""Greedy generation with residual-stream interventions at one decoder layer.

One prompt, K sequences (K = 1 in every reported run), greedy decoding.  A forward hook on decoder layer L edits the
residual stream at the last position of the current forward pass.  Modes (see scripts/run_eval.py):
  last    no edit during generation (the greedy baseline when no delta is given)
  all     a fixed delta added at the last prompt position and at every generated position (all-token steering)
  reroll  entropy gate with cache rollback: after each generated token, the next-token entropy of the unedited logits
          is compared with ent_thr; on a hit the KV cache is cropped by one position, that forward pass is redone with
          the edit applied to the state of the token just read, and the next token is re-decoded.  The edited state
          stays in the cache.
  allgen  the chart step applied in the normal forward pass of every generated position (no gate, no rollback)
Edits: a ChartField geodesic step (kind "chart", direction recomputed from the current state) or a fixed unit vector
scaled to rel * |h| of the current state (kind "vecgate", gate hits only).  Trigger "boundary" fires on generated tokens
that contain a double newline instead of on entropy.
Environment: CX_MODEL (HF model id), CX_LAYER (0-based decoder block whose output is edited), CX_MEMFRAC (fraction of
GPU memory this process may use), CX_ATTN (attention implementation, default sdpa).
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hest.evaluation.judge_safe import equiv, extract_bbox  # noqa: E402  (grader in a killable worker process: sympy can spin forever)

MODEL = os.environ.get("CX_MODEL", "Qwen/Qwen2.5-Math-1.5B-Instruct")
LAYER = int(os.environ.get("CX_LAYER", "19"))
INIT_PROMPT = "Solve the problem."
FORMAT_PROMPT = ("Generate the final answer for the query {query} based on the reasoning process {reasoning} "
                 "in this format: \"Answer: \[ \boxed{{your_answer_here}} \]\". The entire answer should be "
                 "contained completely within the \boxed{{}} command. Do not include any other text.")


class LMB:
    def __init__(self, layer=LAYER, memfrac=float(os.environ.get("CX_MEMFRAC", "0.72"))):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        torch.cuda.set_per_process_memory_fraction(memfrac)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                          attn_implementation=os.environ.get("CX_ATTN", "sdpa")).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
        if self.eot is None or self.eot == self.tok.unk_token_id:          # Qwen family: <|im_end|> ends the assistant turn
            self.eot = self.tok.convert_tokens_to_ids("<|im_end|>")
        if self.eot is None or self.eot == self.tok.unk_token_id:
            self.eot = self.tok.eos_token_id if not isinstance(self.tok.eos_token_id, list) else self.tok.eos_token_id[0]
        self.layer = layer
        self.delta = None          # [K, d] or None
        self.active = False        # gate for the hook
        self.field = None          # optional ChartField: direction recomputed from the current state at every steered token
        self.gate_rows = None      # reroll: bool mask of rows to steer in the current (re-run) forward pass
        self.cur_step = 0
        self.pos = "last"
        self.vec_rel = None        # vecgate: relative step |dh|/|h| of a fixed unit vector applied at gate hits
        self.vec_log = []          # per fire: (intended |dh|/|h|, intended cos(dh, v), realised (bf16) |dh|/|h|, realised cos)
        self.is_boundary = None    # trigger "boundary": bool [vocab], token decodes to text containing "\n\n"
        self.model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not self.active or self.delta is None:
            return output
        d = self.delta.to(hs.dtype)
        if d.shape[0] != hs.shape[0]:
            return output
        if self.field is not None:                                 # geodesic step in the chart at THIS token's state
            H = hs[:, -1, :].float()
            if self.gate_rows is not None:                          # reroll: the caller says which rows fork here
                idx = self.gate_rows.nonzero().flatten()
                dh = torch.zeros_like(H)
                dh[idx] = self.field(H[idx])
                hs[:, -1, :] += dh.to(hs.dtype)
                return output
            hs[:, -1, :] += self.field(H).to(hs.dtype)              # allgen: every generated position
            return output
        if self.gate_rows is not None and self.vec_rel is not None:
            # vecgate: the gate and rollback of reroll, but the step is a FIXED unit vector (self.delta rows) scaled to
            # |dh| = vec_rel * |h| of the current state of each gated row.
            idx = self.gate_rows.nonzero().flatten()
            h0 = hs[idx, -1, :].float()
            v = self.delta[idx].float()
            dh = (self.vec_rel * h0.norm(dim=-1, keepdim=True)) * v
            hs[idx, -1, :] += dh.to(hs.dtype)
            real = hs[idx, -1, :].float() - h0
            cs = torch.nn.functional.cosine_similarity
            for j in range(len(idx)):
                self.vec_log.append((float(dh[j].norm() / h0[j].norm()), float(cs(dh[j], v[j], dim=0)),
                                     float(real[j].norm() / h0[j].norm()), float(cs(real[j], v[j], dim=0))))
            return output
        hs[:, -1, :] += d                                           # fixed delta at the last position (mode "all")
        return output

    def chat(self, q):
        return self.tok.apply_chat_template([{"role": "user", "content": f"{INIT_PROMPT} {q}"}],
                                            tokenize=False, add_generation_prompt=True)

    def ids(self, text, n=1):
        x = self.tok(text, return_tensors="pt").input_ids.to("cuda")
        return x.expand(n, -1).contiguous() if n > 1 else x

    @torch.no_grad()
    def state(self, q, prefix=""):
        """clean layer-L last-token activation of prompt(+prefix)."""
        self.active = False
        out = self.model.model(input_ids=self.ids(self.chat(q) + prefix), output_hidden_states=True)
        return out.hidden_states[self.layer + 1][0, -1, :].float()

    def boundary_table(self):
        """bool [vocab]: the token decodes to text containing a double newline (a reasoning-step boundary)."""
        if self.is_boundary is None:
            V = int(self.model.config.vocab_size)
            tb = torch.zeros(V, dtype=torch.bool)
            for t in range(V):
                try:
                    tb[t] = "\n\n" in self.tok.decode([t])
                except Exception:  # noqa: BLE001
                    pass
            self.is_boundary = tb.cuda()
        return self.is_boundary

    @torch.no_grad()
    def gen_batch(self, q, deltas, K=None, mode="last", max_new=1024, ent_thr=0.7, retrigger=False, refractory=1,
                  trigger="entropy"):
        """deltas: [K, d] cuda tensor (or None for the greedy run).  Returns K texts and token counts."""
        self.retrigger, self.refractory = retrigger, refractory
        if trigger == "boundary":
            self.boundary_table()
        K = deltas.shape[0] if deltas is not None else (K or 1)
        ids = self.ids(self.chat(q), K)
        self.delta = deltas
        # prompt forward: mode "all" also edits the last prompt position (the state that produces the first token)
        self.active = deltas is not None and mode == "all"
        self.pos = "last"
        out = self.model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        alive = torch.ones(K, dtype=torch.bool, device="cuda")
        gen = torch.zeros(K, 0, dtype=torch.long, device="cuda")
        step = 0
        logits = out.logits[:, -1]
        lp_sum = torch.zeros(K, device="cuda"); ent_sum = torch.zeros(K, device="cuda"); nstep = torch.zeros(K, device="cuda")
        fired = torch.zeros(K, dtype=torch.bool, device="cuda")
        self.fire_step = [-1] * K
        self.fire_count = [0] * K
        self.fire_steps = [[] for _ in range(K)]                 # every trigger position (reroll)
        self.last_fire = torch.full((K,), -10.0 ** 9, device="cuda", dtype=torch.float32)   # reroll refractory clock
        while step < max_new:
            nxt = logits.argmax(-1)
            lsm = torch.log_softmax(logits.float(), -1)
            ent = -(lsm.exp() * lsm).sum(-1)
            lp_sum += torch.where(alive, lsm.gather(-1, nxt[:, None]).squeeze(-1), torch.zeros_like(lp_sum))
            ent_sum += torch.where(alive, ent, torch.zeros_like(ent_sum))
            nstep += alive.float()
            nxt = torch.where(alive, nxt, torch.full_like(nxt, self.eot))
            gen = torch.cat([gen, nxt[:, None]], 1)
            alive = alive & (nxt != self.eot)
            step += 1
            if not alive.any() or step >= max_new:
                break
            if mode == "reroll":
                self.active = False                                   # switched on only for the re-run forward pass
            elif mode == "allgen":
                # every generated position is edited in its normal forward pass (never the prompt position), i.e.
                # reroll with ent_thr = 0 in one pass; fire_count counts the edited positions.
                self.active = deltas is not None
                for j in alive.nonzero().flatten().tolist():
                    self.fire_count[j] += 1
                    if self.fire_step[j] < 0:
                        self.fire_step[j] = step
            else:
                self.active = deltas is not None and mode == "all"
            self.pos = "last"
            self.cur_step = step
            out = self.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]
            if mode == "reroll" and (self.field is not None or self.vec_rel is not None):
                # the entropy of THESE (unedited) logits says whether position (step) is a fork; roll the cache back one
                # position, redo that forward pass WITH the step, and re-decode the token it commits to.
                lsm2 = torch.log_softmax(logits.float(), -1)
                e2 = -(lsm2.exp() * lsm2).sum(-1)
                if trigger == "boundary":                     # fire when the token just read contains "\n\n"
                    fire = alive & self.is_boundary[nxt]
                else:
                    fire = alive & (e2 > ent_thr)
                if retrigger:
                    fire = fire & ((step - self.last_fire) >= max(1, refractory))
                else:
                    fire = fire & (~fired)
                if bool(fire.any()):
                    for j in fire.nonzero().flatten().tolist():
                        if self.fire_step[j] < 0:
                            self.fire_step[j] = step          # step of the first trigger (logging)
                        self.fire_count[j] += 1
                        self.fire_steps[j].append(step)
                    self.last_fire = torch.where(fire, torch.full_like(self.last_fire, float(step)), self.last_fire)
                    fired = fired | fire
                    past.crop(past.get_seq_length() - 1)
                    self.gate_rows = fire
                    self.active, self.delta = True, deltas
                    out = self.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
                    self.active, self.gate_rows = False, None
                    past = out.past_key_values
                    logits = out.logits[:, -1]
        self.active = False
        self.delta = None
        texts, ntok = [], []
        for i in range(K):
            toks = gen[i].tolist()
            L = next((k for k, t in enumerate(toks) if t == self.eot), len(toks))
            texts.append(self.tok.decode(toks[:L], skip_special_tokens=True))
            ntok.append(L)
        del past, out
        self.last_logp = (lp_sum / nstep.clamp_min(1)).tolist()
        self.last_ent = (ent_sum / nstep.clamp_min(1)).tolist()
        return texts, ntok

    @torch.no_grad()
    def judge_batch(self, q, texts, gold):
        """Decode-time answer extraction: the boxed answer of the text when every text has one, otherwise the model
        restates its own final answer (reformat prompt) and the boxed answer of that restatement is taken.  The
        restatement is stored as the row's `preds` and used by judge_unified.py as the model's own answer."""
        self.active = False
        direct = [extract_bbox(t) for t in texts]
        if all(d is not None for d in direct) and os.environ.get("CX_DIRECT_BOX", "1") == "1":
            return direct, [bool(equiv(gold, p)) for p in direct]
        msgs = [self.tok.apply_chat_template([{"role": "user", "content": FORMAT_PROMPT.format(query=q, reasoning=t)}],
                                             tokenize=False, add_generation_prompt=True) for t in texts]
        enc = self.tok(msgs, return_tensors="pt", padding=True).to("cuda")
        g = self.model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=self.tok.pad_token_id,
                                eos_token_id=self.eot)
        seq = g[:, enc.input_ids.shape[1]:]
        outs = self.tok.batch_decode(seq, skip_special_tokens=True)
        del g, seq, enc
        torch.cuda.empty_cache()
        preds = [extract_bbox(o) for o in outs]
        return preds, [bool(equiv(gold, p)) for p in preds]
