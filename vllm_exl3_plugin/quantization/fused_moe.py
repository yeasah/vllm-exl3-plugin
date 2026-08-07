"""EXL3 method for routed experts.

EXL3 stores MoE experts as *separate per-expert 2D tensors* —
`...experts.{E}.{gate,up,down}_proj.{trellis,suh,svh,mcg|mul1}` — rather than
the packed 3D `[num_experts, ...]` weights an unquantized checkpoint carries.
That shape difference drives the whole design here.

**Why `exl3_mgemm` works here when it did not for fused QKV.** Phase 1
established that merged QKV cannot use exllamav3's pointer-table kernel,
because EXL3 assigns bit widths per tensor and real checkpoints mix them inside
one layer (q=4, k=5, v=5), while `MultiLinear` asserts equal K. Experts turn out
not to have that problem: scanning every expert tensor in Laguna-XS (121,212
tensors) and gemma-4-26B (47,652) found a single bit width throughout, with no
mixed-K group anywhere. `process_weights_after_loading` re-checks that per layer
rather than trusting it, because a mixed-K checkpoint would otherwise be decoded
with the wrong K and produce plausible-looking nonsense.

**Why parameters are collectors rather than one per tensor.** Laguna has 256
experts x 3 projections x 39 layers. Registering an `nn.Parameter` per stored
tensor would mean ~120,000 of them, which is not a workable amount of Python
object. Instead each layer registers eight collectors — `w13_*` and `w2_*` for
each EXL3 sub-tensor — that accumulate per-expert tensors as vLLM's loader
delivers them, and `process_weights_after_loading` builds an int64 table of
device pointers over them, which is what the kernel actually consumes. The
per-expert tensors are deliberately left where they are rather than stacked; see
`_pointers`.

The `w13_` / `w2_` naming is not ours to choose: vLLM's expert mapping rewrites
`experts.{id}.gate_proj.` to `experts.w13_` and keeps whatever suffix follows,
so `experts.0.gate_proj.trellis` arrives as `w13_trellis`.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from .. import format, ops, tp
from ..log import init_logger

logger = init_logger(__name__)

#: vLLM's shard ids: w1 = gate, w3 = up (both land in w13_*), w2 = down.
_GATE, _UP, _DOWN = "w1", "w3", "w2"


class EXL3MoEParameter(torch.nn.Parameter):
    """Collector for one EXL3 sub-tensor across every expert of a layer."""

    def __new__(cls, device: torch.device, name: str, tp_rank: int, tp_size: int):
        obj = torch.Tensor._make_subclass(
            cls, torch.empty(0, device=device), False
        )
        obj.exl3_device = device
        # Which EXL3 sub-tensor this collects, and where this rank sits: the
        # loader needs both to slice, and it only ever sees the parameter.
        obj.exl3_name = name
        obj.exl3_tp_rank = tp_rank
        obj.exl3_tp_size = tp_size
        obj.shards: dict[tuple[str, int], torch.Tensor] = {}
        return obj

    def release(self) -> None:
        self.shards.clear()


def _tp_shard(
    name: str, t: torch.Tensor, shard_id: str, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """This rank's slice of one expert sub-tensor.

    Routed experts shard on the *intermediate* dimension, which is a column
    (output) split for gate/up and a row (input) split for down -- the same pair
    of roles `tp.py` already describes for dense linears, applied per expert.

    The dimension being split is the **stored** intermediate, not the model's.
    exllamav3 pads before quantizing (gemma-4-26B: 704 -> 768), so the shard
    boundaries have to come from the tensor in hand rather than from vLLM's
    `intermediate_size_per_partition`. Each sub-tensor carries that width in a
    different place, which is why the role decides where to read it.
    """
    role = tp.role_of(name)
    if tp_size == 1:
        return t
    column = shard_id in (_GATE, _UP)
    if role == tp.ROLE_TRELLIS:
        dim = t.shape[1 if column else 0] * format.TILE
    elif role == (tp.ROLE_SVH if column else tp.ROLE_SUH):
        dim = t.shape[0]
    else:
        # The other scale vector indexes the hidden dimension, which this split
        # leaves whole, and mcg/mul1 are scalars. Both replicate.
        return t
    first, last = format.shard_bounds(
        dim, tp_rank, tp_size, f"MoE intermediate ({shard_id})"
    )
    split = tp.shard_column if column else tp.shard_row
    return split(role, t, first, last)


def _moe_weight_loader(
    param: EXL3MoEParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
):
    """Store one expert's tensor. vLLM's own loader would copy into a packed
    parameter, which is exactly what EXL3's layout cannot provide."""
    del weight_name
    # plugin._patch_moe_trellis_rank lifts 3D trellis tensors to 4D so vLLM
    # does not mistake them for packed multi-expert weights; undo that here.
    if loaded_weight.dim() == 4 and loaded_weight.shape[0] == 1:
        loaded_weight = loaded_weight.squeeze(0)
    loaded_weight = _tp_shard(
        param.exl3_name, loaded_weight, shard_id, param.exl3_tp_rank,
        param.exl3_tp_size,
    )
    param.shards[(shard_id, int(expert_id))] = loaded_weight.to(
        param.exl3_device
    ).contiguous()
    return True if return_success else None


class EXL3MoEMethod(FusedMoEMethodBase):
    """Routed experts backed by `exl3_mgemm`."""

    def __init__(self, quant_config, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        names = quant_config.stored_tensor_names()
        self.mcg = "mcg" in names
        self.mul1 = "mul1" in names

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        tp_size, tp_rank = self.moe.tp_size, self.moe.tp_rank
        if self.moe.ep_size > 1:
            raise NotImplementedError(
                "EXL3 routed experts do not support expert parallelism. "
                "exl3_mgemm can filter an expert range (`min_index`/`max_index`) "
                "but refuses to combine that with the multi-token weighted "
                "reduction this method relies on."
            )
        extra_weight_attrs.pop("weight_loader", None)
        device = torch.empty(0).device

        for prefix in ("w13_", "w2_"):
            for name in self.quant_config.stored_tensor_names():
                param = EXL3MoEParameter(device, name, tp_rank, tp_size)
                set_weight_attrs(param, {"weight_loader": _moe_weight_loader})
                set_weight_attrs(param, extra_weight_attrs)
                layer.register_parameter(prefix + name, param)

        layer.exl3_num_experts = num_experts
        layer.exl3_hidden_size = hidden_size
        layer.exl3_intermediate = intermediate_size_per_partition
        layer.exl3_tp_size = tp_size

    # ------------------------------------------------------------------ load

    def _collect(self, layer, prefix: str, shard_id: str, name: str):
        """Every expert's `name` tensor for one projection, ordered by id."""
        param = getattr(layer, prefix + name, None)
        if param is None:
            raise format.EXL3FormatError(
                f"expected a '{prefix}{name}' parameter for this MoE layer"
            )
        experts = layer.exl3_num_experts
        missing = [e for e in range(experts) if (shard_id, e) not in param.shards]
        if missing:
            raise format.EXL3FormatError(
                f"{prefix}{name}: no tensor loaded for experts {missing[:4]}"
                f"{'...' if len(missing) > 4 else ''} of {experts}"
            )
        return [param.shards[(shard_id, e)] for e in range(experts)]

    @staticmethod
    def _check_uniform(tensors, what: str) -> None:
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) != 1:
            raise format.EXL3FormatError(
                f"{what}: experts disagree on shape ({sorted(shapes)[:3]}). "
                "exl3_mgemm takes one K and one shape per call, so mixed "
                "experts would need a per-expert loop instead of its "
                "pointer table."
            )

    @staticmethod
    def _interm_divisor(gate_suh, up_suh) -> float:
        """The constant exllamav3 folded into this layer's up projection.

        Measured rather than configured -- see `format.infer_interm_divisor` for
        why the checkpoint cannot tell us. The median over experts is what makes
        this safe: per-channel calibration puts a few percent on each expert's
        ratio, but 256 of them agreeing on 128 is not something calibration
        noise produces.
        """
        ratios = torch.stack(
            [
                g.float().abs().mean() / u.float().abs().mean()
                for g, u in zip(gate_suh, up_suh)
            ]
        )
        return format.infer_interm_divisor(ratios.median().item())

    @staticmethod
    def _pointers(tensors) -> torch.Tensor:
        """int64 device addresses, one per expert -- what the kernel indexes.

        Deliberately *not* stacked into a contiguous `[num_experts, ...]`
        tensor. The kernel dereferences each address independently, so stacking
        buys nothing and costs a second full copy of every expert weight while
        both exist -- on a 256-expert model that is enough to exhaust a 16 GB
        card during load. exllamav3's own `MultiLinear` builds its tables the
        same way, from separately allocated tensors.
        """
        return torch.tensor(
            [t.data_ptr() for t in tensors], dtype=torch.long, device=tensors[0].device
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        projections = {"gate": ("w13_", _GATE), "up": ("w13_", _UP),
                       "down": ("w2_", _DOWN)}
        bits = {}
        for label, (prefix, shard_id) in projections.items():
            groups = {}
            for name in ("trellis", "suh", "svh"):
                tensors = self._collect(layer, prefix, shard_id, name)
                self._check_uniform(tensors, f"{prefix}{name}")
                groups[name] = tensors
            bits[label] = format.bits_from_trellis_shape(groups["trellis"][0].shape)

            for name, tensors in groups.items():
                # Strong references, so the storage the pointer table addresses
                # cannot be collected.
                setattr(layer, f"_exl3_{label}_{name}", tensors)
                setattr(layer, f"exl3_{label}_{name}_ptrs", self._pointers(tensors))

        if bits["gate"] != bits["up"]:
            raise format.EXL3FormatError(
                f"gate and up projections have different bit widths "
                f"({bits['gate']} vs {bits['up']}); exl3_mgemm takes one K per call"
            )
        layer.exl3_gate_bits = bits["gate"]
        layer.exl3_down_bits = bits["down"]
        layer.exl3_interm_div = self._interm_divisor(
            layer._exl3_gate_suh, layer._exl3_up_suh
        )
        if layer.exl3_interm_div != 1.0:
            # Once, not once per layer: every MoE layer in a checkpoint carries
            # the same divisor, and there are 39 of them in Laguna-XS.
            logger.info_once(
                "EXL3 routed experts: up projection is pre-scaled by 1/%g in "
                "this checkpoint; compensating in the routing weights.",
                layer.exl3_interm_div,
            )
        # exllamav3 pads the intermediate dimension to a multiple of 128 before
        # quantizing (gemma-4-26B: 704 -> 768), so the width the kernels work in
        # is the stored one, not the model's.
        layer.exl3_stored_intermediate = (
            layer._exl3_gate_trellis[0].shape[1] * format.TILE
        )

        for prefix in ("w13_", "w2_"):
            for name in self.quant_config.stored_tensor_names():
                getattr(layer, prefix + name).release()

    # ----------------------------------------------------------------- apply

    @staticmethod
    def activation_name(layer: torch.nn.Module) -> str:
        """vLLM's activation for this MoE, as a plain string.

        Not always silu: gemma-4 uses `gelu_tanh`. Getting this wrong produces
        a plausible-looking but wrong model rather than an error.
        """
        act = getattr(layer, "activation", "silu")
        return getattr(act, "value", act)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        # Only used by vLLM's modular kernel stack, which this method does not
        # route through -- `apply` calls exl3_mgemm directly.
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # vLLM runs shared experts itself when the method cannot overlap them,
        # which is our case (`mk_can_overlap_shared_experts` is False).
        del shared_experts, shared_experts_input
        if getattr(layer, "apply_router_weight_on_input", False):
            raise NotImplementedError(
                "apply_router_weight_on_input is not supported by the EXL3 MoE "
                "method; exl3_mgemm applies routing weights in its reduction."
            )

        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        out = ops.exl3_moe_mm(
            flat,
            topk_ids,
            topk_weights,
            layer.exl3_gate_trellis_ptrs,
            layer.exl3_gate_suh_ptrs,
            layer.exl3_gate_svh_ptrs,
            layer.exl3_up_trellis_ptrs,
            layer.exl3_up_suh_ptrs,
            layer.exl3_up_svh_ptrs,
            layer.exl3_down_trellis_ptrs,
            layer.exl3_down_suh_ptrs,
            layer.exl3_down_svh_ptrs,
            layer.exl3_gate_bits,
            layer.exl3_down_bits,
            layer.exl3_stored_intermediate,
            layer.exl3_num_experts,
            self.mcg,
            self.mul1,
            self.activation_name(layer),
        )
        # Restore the magnitude the checkpoint's pre-scaled up projection gave
        # away, *outside* the kernel. Folding it into the routing weights
        # instead is algebraically identical -- sum_j (d*w_j) y_j == d * sum_j
        # w_j y_j -- but it puts the factor inside exl3_mgemm's fp16 output
        # accumulator, where on Laguna-XS it drove the layer-38 routed output to
        # 155648 against an fp16 ceiling of 65504. That overflowed to inf, and
        # one inf in the residual stream turns the whole prefill hidden state
        # NaN by the final layer. Applied here the reduction stays 128x smaller
        # and the scale lands in the model's own (bf16) dtype, where it is exact
        # because the divisor is a power of two.
        #
        # exllamav3 avoids the same overflow differently, by giving the routed
        # down projection `out_dtype = torch.float`; that works too but costs a
        # second full-size fp32 scratch buffer at max batch.
        if layer.exl3_interm_div != 1.0:
            out = out * layer.exl3_interm_div
        return out.reshape(orig_shape)
