# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernel.utils import (binary_packer, binary_unpacker, gemlite_nanoquant_packer, get_kernel_function,
                            marlin_nanoquant_packer)


class NanoQuantLinear(nn.Module):
    def __quant_convert__(
        self,
        do_train: bool,
        *,
        rank: int = 1024,
        factor_results: argparse.Namespace = None,
        **kwargs,
    ):
        self.do_train = do_train
        self.rank = rank
        self._binarized = not self.do_train
        self.dtype = torch.bfloat16

        assert factor_results is not None, "factor_results must be provided"
        self._setup_path(factor_results)

        if not self.do_train:
            for param in self.parameters():
                param.requires_grad_(False)

        if self.bias is not None and self.do_train:
            self.bias.requires_grad = True
            self.bias.optim_group = "bias"

        if hasattr(self, "weight"):
            del self.weight
        self.register_parameter("weight", None)

    def _setup_path(self, factors):
        if factors is not None:
            vals = {
                "scale_pre": factors.scale_pre.float(),
                "scale_post": factors.scale_post.float(),
            }
            if hasattr(factors, "scale_mid") and factors.scale_mid is not None:
                vals["scale_mid"] = factors.scale_mid.float()

            if self.do_train:
                vals["V_latent"] = factors.B_latent.float()
                vals["U_latent"] = factors.A_latent.mT.float()
            else:
                vals["V"] = self.binary_ste(factors.B.float())
                vals["U"] = self.binary_ste(factors.A.float().mT)
        else:
            device = None
            if hasattr(self, "bias") and self.bias is not None:
                device = self.bias.device
            if device is None:
                device = torch.device("cpu")

            vals = {
                "scale_pre": torch.empty(1, self.in_features, device=device),
                "scale_post": torch.empty(1, self.out_features, device=device),
                "V": torch.empty(self.rank, self.in_features, device=device),
                "U": torch.empty(self.out_features, self.rank, device=device),
                "scale_mid": torch.empty(1, self.rank, device=device),
            }

        for name, tensor in vals.items():
            requires_grad = self.do_train
            param = nn.Parameter(tensor.to(self.dtype), requires_grad=requires_grad)

            if self.do_train:
                if "scale" in name:
                    param.optim_group = "scale"
                elif "latent" in name:
                    param.optim_group = "binary"

            setattr(self, name, param)

    def init_for_inference(self, rank, has_scale_mid=False):
        self.rank = rank
        self.do_train = False
        self._binarized = True
        self.dtype = torch.bfloat16

        if "weight" in self._parameters:
            del self._parameters["weight"]
        self.register_parameter("weight", None)

        self._setup_path(factors=None)

        if not has_scale_mid and hasattr(self, "scale_mid"):
            delattr(self, "scale_mid")

    def binary_ste(self, x):
        y = x.sign()
        y[y == 0] = 1
        return (y - x).detach() + x

    def forward(self, x):
        if getattr(self, "do_kernel_inference", False):
            orig_dtype = x.dtype
            if x.dtype != self.dtype:
                x = x.to(self.dtype)

            original_shape = None
            if x.ndim == 3:
                original_shape = x.shape
                x = x.contiguous().view(-1, x.size(-1))
            elif x.ndim == 2:
                x = x.contiguous()
            else:
                raise ValueError(f"Expected 2D or 3D input tensor, got {x.ndim}D")

            if self.kernel_type == "gemv":
                y = self.kernel_forward(
                    x,
                    self.scale_pre,
                    self.V_int8,
                    self.scale_mid,
                    self.U_int8,
                    self.scale_post,
                    **self.kernel_kwargs,
                )
            elif self.kernel_type == "gemm":
                y = self.kernel_forward(x, self.V_int8, self.U_int8, self.s1, self.s2, self.s3, self.workspace)
            elif self.kernel_type == "gemlite":
                y = self._gemlite_forward(x)
            else:
                raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

            if original_shape is not None:
                y = y.view(original_shape[0], original_shape[1], -1)

            return y.to(orig_dtype) if y.dtype != orig_dtype else y

        V_main = self.V_latent if self.do_train and hasattr(self, "V_latent") else self.V
        U_main = self.U_latent if self.do_train and hasattr(self, "U_latent") else self.U
        s_mid_m = getattr(self, "scale_mid", None)

        y = self._compute_forward(x, V_main, U_main, self.scale_pre, s_mid_m, self.scale_post)
        if self.bias is not None:
            y += self.bias
        return y

    def _prepare_kernel(self, kernel_type="gemv", dtype=torch.float16):
        self.do_kernel_inference = True
        self.kernel_type = kernel_type
        self.dtype = dtype

        if kernel_type == "gemlite":
            pad_multiple = int(getattr(self, "gemlite_pad_multiple", int(os.environ.get("GEMLITE_PAD_MULTIPLE",
                                                                                        "128"))))
            self._prepare_gemlite_kernel(dtype=dtype, pad_multiple=pad_multiple)
            return

        # prepare scales
        self.scale_pre = nn.Parameter(self.scale_pre.to(dtype).squeeze(0), requires_grad=False)
        self.scale_post = nn.Parameter(self.scale_post.to(dtype).squeeze(0), requires_grad=False)
        self.scale_mid = getattr(self, "scale_mid", None)
        if self.scale_mid is not None:
            self.scale_mid = nn.Parameter(self.scale_mid.to(dtype).squeeze(0), requires_grad=False)

        if kernel_type == "gemv":
            self.U_int8 = binary_packer(self.U.to(torch.int8), method="half2")
            self.V_int8 = binary_packer(self.V.to(torch.int8), method="half2")
            delattr(self, "U")
            delattr(self, "V")
            self.kernel_forward = get_kernel_function("nanoquant", dtype, kernel_type="dynamic")
            self.kernel_kwargs = {
                "num_thread_stage1": 128,
                "num_thread_stage2": 128,
                "num_row_per_warp_stage1": 2,
                "num_row_per_warp_stage2": 2,
                "num_acc_stage1": 1,
                "num_acc_stage2": 1,
                "num_pipeline_stage1": 8,
                "num_pipeline_stage2": 8,
            }
        elif kernel_type == "gemm":
            if self.scale_mid is None:
                self.scale_mid = nn.Parameter(
                    torch.ones(self.rank, device=self.scale_pre.device, dtype=dtype),
                    requires_grad=False,
                )
            (self.V_int8, self.U_int8, self.s1, self.s2, self.s3, self.workspace) = marlin_nanoquant_packer(
                self.V.to(torch.int8).T,
                self.U.to(torch.int8).T,
                self.scale_pre,
                self.scale_mid,
                self.scale_post,
            )
            for key in ["U", "V", "scale_pre", "scale_mid", "scale_post"]:
                delattr(self, key)
            import binary_kernels  # noqa: F401
            self.kernel_forward = getattr(torch.ops.binary_kernels, "marlin_nanoquant_forward")
        else:
            raise ValueError("kernel_type must be one of ['gemv', 'gemm', 'gemlite'].")

    def _prepare_gemlite_kernel(self, dtype: torch.dtype, pad_multiple: int = 128):
        packed = gemlite_nanoquant_packer(
            U=self.U,
            V=self.V,
            scale_pre=self.scale_pre,
            scale_mid=getattr(self, "scale_mid", None),
            scale_post=self.scale_post,
            pad_multiple=pad_multiple,
            dtype=dtype,
            config_path=os.environ.get("GEMLITE_CONFIG_PATH", None),
            save_config=False,
        )

        self._gemlite_pad = packed["pad"]
        self._gemlite_V_linear = packed["V_linear"]
        self._gemlite_U_linear = packed["U_linear"]

        self._gemlite_scale_pre0 = nn.Parameter(packed["scale_pre0"], requires_grad=False)  # [K0]
        self._gemlite_scale_mid_pad = nn.Parameter(packed["scale_mid_pad"], requires_grad=False)  # [rankp]
        self._gemlite_scale_post0 = nn.Parameter(packed["scale_post0"], requires_grad=False)  # [N0]

        # delete originals (GemLite owns packed weights; scales stored separately)
        for key in ["U", "V", "scale_pre", "scale_post"]:
            if hasattr(self, key):
                delattr(self, key)
        if hasattr(self, "scale_mid"):
            delattr(self, "scale_mid")

    def _gemlite_forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self._gemlite_pad
        N0 = pad["N0"]

        x_scaled = x * self._gemlite_scale_pre0
        h = self._gemlite_V_linear.forward_auto_no_warmup(x_scaled)  # [M, rankp]
        h = h * self._gemlite_scale_mid_pad  # zero padded ranks
        y_pad = self._gemlite_U_linear.forward_auto_no_warmup(h)  # [M, Np]

        y = y_pad[..., :N0] * self._gemlite_scale_post0
        if self.bias is not None:
            y = y + self.bias
        return y

    def _compute_forward(self, x, V, U, scale_pre, scale_mid, scale_post):
        Vq = self.quantize(V)
        Uq = self.quantize(U)

        y = F.linear(x * scale_pre, Vq)
        if scale_mid is not None:
            y = y * scale_mid
        y = F.linear(y, Uq)
        y = y * scale_post
        return y

    def quantize(self, x):
        if self._binarized:
            return x
        return self.binary_ste(x)

    def finalize(self):
        if not self.do_train:
            return
        with torch.no_grad():
            self.do_train = False
            self._binarized = True

            latent_attrs = [name for name, _ in self.named_parameters() if "latent" in name]

            for attr_name in latent_attrs:
                base_name = attr_name.replace("_latent", "")
                final_value = self.binary_ste(getattr(self, attr_name))
                setattr(self, base_name, nn.Parameter(final_value.detach(), requires_grad=False))

            for param in self.parameters():
                param.requires_grad_(False)

            for attr_name in latent_attrs:
                if hasattr(self, attr_name):
                    delattr(self, attr_name)

    # -------------------------
    # packing / state_dict
    # -------------------------
    def pack_weights(self):
        packed_data = {}

        def pack_param(param_name):
            if not hasattr(self, param_name):
                return
            param = getattr(self, param_name)
            if param is None:
                return
            param_bin = self.binary_ste(param.data) if self.do_train else param.data
            packed_data[f"{param_name}_packed"] = binary_packer(param_bin.to(torch.int8))
            packed_data[f"{param_name}_shape"] = torch.tensor(param.shape, dtype=torch.long)

        pack_param("V")
        pack_param("U")
        return packed_data

    def state_dict(self, *args, **kwargs):
        if self.do_train:
            return super().state_dict(*args, **kwargs)

        state = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")

        keys_to_remove = [k for k in state.keys() if ".V" in k or ".U" in k]
        for k in keys_to_remove:
            if k in state:
                del state[k]

        packed_weights = self.pack_weights()
        for k, v in packed_weights.items():
            state[prefix + k] = v

        return state

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        has_mid = (prefix + "scale_mid") in state_dict
        if not has_mid and hasattr(self, "scale_mid"):
            delattr(self, "scale_mid")

        def unpack_param(param_name):
            packed_key = prefix + f"{param_name}_packed"
            shape_key = prefix + f"{param_name}_shape"
            if packed_key in state_dict and shape_key in state_dict:
                packed_val = state_dict.pop(packed_key)
                shape = state_dict.pop(shape_key)
                unpacked_tensor = binary_unpacker(packed_val).view(tuple(shape)).to(self.dtype)
                setattr(self, param_name, nn.Parameter(unpacked_tensor, requires_grad=False))

        unpack_param("V")
        unpack_param("U")

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
