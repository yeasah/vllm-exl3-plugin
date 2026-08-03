"""EXL3 linear method.

Phase 0 strategy: dequantize once, at load time.

`process_weights_after_loading` turns every EXL3 tensor into a dense fp16
weight and `apply` is a plain `F.linear`. That costs the entire memory saving of
the quantization, which is obviously not the point of the project -- but it
isolates the two things Phase 0 is actually trying to prove (that the plugin
registers, and that vLLM's loader can be driven to fill EXL3's tensor layout)
from the things Phase 1 will prove (that the fused kernels work under vLLM's
serving loop). Swapping `apply` over to `exl3_gemm` behind a custom op is the
Phase 1 change and touches only this file plus `ops.py`.

The one structural decision here that is *not* provisional: merged linears keep
one sub-tensor per shard instead of a single concatenated tensor. EXL3 assigns
bit widths per tensor, and mixed-bpw checkpoints really do use different widths
for q, k and v inside one layer (turboderp/Llama-3.2-1B-Instruct-exl3 at 3.5bpw
has q=4, k=5, v=5 bits in every layer). Different K means a different trellis
last dimension, so there is no concatenated representation to build, and
exllamav3's own pointer-table kernel (`exl3_mgemm`, via `MultiLinear`) asserts
equal K across its inputs. Fused QKV therefore has to be N launches plus a
concat, at every phase.
"""

from __future__ import annotations

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs

from .. import format, ops


class EXL3Parameter(BasevLLMParameter):
    """Placeholder that collects one EXL3 sub-tensor per output shard.

    EXL3 tensors cannot be preallocated the way vLLM's parameter classes expect:
    the trellis shape depends on a per-tensor bit width that is only known once
    the tensor has been seen. So this parameter allocates nothing and simply
    records what the loader hands it, keyed by shard. The four `load_*` entry
    points are the whole weight_loader_v2 protocol; each one just stores.
    """

    def __new__(cls, **kwargs):
        return super().__new__(cls, data=None)

    def __init__(self, num_shards: int, device: torch.device, weight_loader):
        super().__init__(data=None, weight_loader=weight_loader)
        self.num_shards = num_shards
        self.exl3_device = device
        self.shards: dict[int, torch.Tensor] = {}

    def _store(self, loaded_weight: torch.Tensor, shard_id=0) -> None:
        index = self._shard_id_as_int(shard_id)
        # The loader hands us mmapped CPU tensors; nothing else will move these
        # onto the device, because we never allocated a device-resident param.
        self.shards[index] = loaded_weight.to(self.exl3_device).contiguous()

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._store(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._store(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs) -> None:
        self._store(loaded_weight, kwargs.get("shard_id", 0))

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs) -> None:
        self._store(loaded_weight, kwargs.get("shard_id", 0))

    def release(self) -> None:
        self.shards.clear()
        self.data = torch.empty(0, device=self.exl3_device)


@register_weight_loader_v2_supported_method
class EXL3LinearMethod(LinearMethodBase):
    """Linear method for EXL3-quantized checkpoints."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size

        tp_size = getattr(layer, "tp_size", 1)
        if tp_size > 1:
            raise NotImplementedError(
                "vllm-exl3-plugin does not support tensor parallelism yet "
                "(Phase 2). EXL3's Hadamard transform is block-diagonal in "
                f"blocks of {format.HAD_BLOCK}, so shards are only exact on "
                "128-channel boundaries; see format.check_tp_split."
            )

        weight_loader = extra_weight_attrs.pop("weight_loader", None)
        assert weight_loader is not None
        num_shards = len(output_partition_sizes)

        # vLLM builds the model inside a device context; picking the device up
        # from an empty allocation is how we learn where loaded shards belong.
        device = torch.empty(0).device

        for name in self.quant_config.stored_tensor_names():
            param = EXL3Parameter(
                num_shards=num_shards, device=device, weight_loader=weight_loader
            )
            set_weight_attrs(param, extra_weight_attrs)
            layer.register_parameter(name, param)

        layer.exl3_input_size = input_size_per_partition
        layer.exl3_output_sizes = list(output_partition_sizes)
        layer.exl3_params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        names = self.quant_config.stored_tensor_names()
        params = {name: getattr(layer, name) for name in names}
        has_mcg = "mcg" in params
        has_mul1 = "mul1" in params

        shards = []
        for index, out_size in enumerate(layer.exl3_output_sizes):
            trellis = self._shard(params, "trellis", index, layer)
            suh = self._shard(params, "suh", index, layer)
            svh = self._shard(params, "svh", index, layer)

            bits = format.bits_from_trellis_shape(trellis.shape)
            stored_in, stored_out = format.dims_from_trellis_shape(trellis.shape)
            self._check_shapes(
                layer, index, stored_in, stored_out, out_size, suh, svh
            )

            weight = ops.dense_weight(trellis, suh, svh, bits, has_mcg, has_mul1)
            # exllamav3 pads both dimensions up to a multiple of 128 before
            # quantizing. Padded output columns hold quantization noise and
            # padded input rows only ever multiply zeros, so trimming both is
            # exact -- this is the same trim exllamav3's own `Linear.forward`
            # applies via `trim_padded_out`.
            weight = weight[:out_size, : layer.exl3_input_size]
            shards.append(weight)

        weight = torch.cat(shards, dim=0) if len(shards) > 1 else shards[0]
        layer.register_parameter(
            "exl3_weight", Parameter(weight.contiguous(), requires_grad=False)
        )

        for param in params.values():
            param.release()

    @staticmethod
    def _shard(params, name, index, layer):
        param = params.get(name)
        if param is None:
            raise format.EXL3FormatError(
                f"{getattr(layer, 'prefix', '<layer>')}: expected a '{name}' "
                "tensor for this EXL3 layer but the checkpoint has none"
            )
        if index not in param.shards:
            raise format.EXL3FormatError(
                f"{getattr(layer, 'prefix', '<layer>')}: no '{name}' tensor was "
                f"loaded for output shard {index} of {param.num_shards}"
            )
        return param.shards[index]

    @staticmethod
    def _check_shapes(layer, index, stored_in, stored_out, out_size, suh, svh):
        expected_in = format.pad_dim(layer.exl3_input_size)
        expected_out = format.pad_dim(out_size)
        if stored_in != expected_in or stored_out != expected_out:
            raise format.EXL3FormatError(
                f"shard {index}: checkpoint stores a {stored_in}x{stored_out} "
                f"EXL3 tensor, but vLLM's layer wants {layer.exl3_input_size}x"
                f"{out_size} (padded to {expected_in}x{expected_out})"
            )
        if suh.numel() != stored_in or svh.numel() != stored_out:
            raise format.EXL3FormatError(
                f"shard {index}: suh/svh lengths {suh.numel()}/{svh.numel()} do "
                f"not match trellis dimensions {stored_in}/{stored_out}"
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.exl3_weight, bias)
