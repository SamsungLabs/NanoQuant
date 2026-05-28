from collections.abc import Callable, Iterable
from typing import Any
from warnings import warn

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import HAS_TRITON, MIN_TORCH_2_1, MIN_TORCH_2_6


class OptimiOptimizer(Optimizer):
    """Provides common functionality for optimi optimizers."""
    def __init__(self, params: Iterable[Tensor] | Iterable[dict], defaults: dict[str, Any]):
        super().__init__(params, defaults)

        high_wds: list[float] = []
        for group in self.param_groups:
            # Validate hyper‑parameters for each group
            if not 0.0 <= group["lr"]:
                raise ValueError(f"Invalid learning rate: lr={group['lr']}")
            if not 0.0 <= group["weight_decay"]:
                raise ValueError(f"Invalid weight decay: weight_decay={group['weight_decay']}")

            # Set decouple_lr and max_lr
            if group["decouple_lr"] and group["max_lr"] is None:
                group["max_lr"] = group["lr"]
            if group["max_lr"] is not None and not 0.0 <= group["max_lr"]:
                raise ValueError(f"Invalid maximum learning rate: max_lr={group['max_lr']}")

            # Check PyTorch version and external library (Triton) dependencies
            if not MIN_TORCH_2_1:
                if group["foreach"]:
                    raise ValueError(f"foreach={group['foreach']} requires PyTorch 2.1 or later. "
                                     "Set foreach=False or upgrade PyTorch.")
                else:
                    group["foreach"] = False
                if group["gradient_release"]:
                    raise ValueError(f"gradient_release={group['gradient_release']} requires PyTorch 2.1 or later. "
                                     "Upgrade PyTorch to use.")

            if group["foreach"]:
                warn(
                    "Parameter `foreach` is deprecated in favor of the faster `triton` implementation "
                    "and will be removed in a future release. Set `foreach=False` to silence this warning.",
                    category=DeprecationWarning,
                )

            if not MIN_TORCH_2_6 and group.get("triton", False):
                raise ValueError(f"triton={group['triton']} requires PyTorch 2.6 or later. "
                                 "Set triton=False or upgrade PyTorch.")
            if not HAS_TRITON and group.get("triton", False):
                raise ImportError("Triton could not be imported on this system. Set triton=False or install Triton.")

            # Collect warnings for high weight_decay when decouple_lr=True
            if group.get("decouple_lr", False) and group.get("weight_decay", 0.0) >= 1e-3:
                high_wds.append(group["weight_decay"])

            # Handle groups where gradient_release is enabled
            if group["gradient_release"]:
                if group["foreach"]:
                    warn(
                        "Gradient release (gradient_release=True) and foreach (foreach=True) cannot be used together. "
                        "Disabling foreach.",
                        category=UserWarning,
                    )
                    group["foreach"] = False

                # Set up state information needed for gradient_release
                for p in group["params"]:
                    self.state[p]["group"] = group

        # After the loop, issue a single consolidated warning based on the collected information
        if high_wds:
            warn(
                f"You are using weight_decay up to {max(high_wds)} which is potentially high for decouple_lr=True. "
                "Unlike decoupled weight decay, fully decoupled weight decay does not reduce weight decay by the learning rate.",
                category=UserWarning,
            )

        # Set the default value for optimizer_accumulation
        self._optimizer_accumulation = False

    @property
    def optimizer_accumulation(self) -> bool:
        """Accumulate gradients in optimizer states during gradient release instead of a full step."""
        return self._optimizer_accumulation

    @optimizer_accumulation.setter
    def optimizer_accumulation(self, optimizer_accumulation: bool):
        """Accumulate gradients in optimizer states during gradient release instead of a full step."""
        self._optimizer_accumulation = optimizer_accumulation

    def step(self, closure: Callable | None = None, param: Tensor | None = None):
        """Performs a single optimization step on the whole model or an individual parameter.

        Args:
            closure: A closure which re‑evaluates the model and returns the loss.
                Incompatible with performing an optimization step on a single `param`.
            param: An individual parameter to perform a fused optimization step during the backward
                pass. Requires the optimizer to be initialized with `gradient_release=True` and
                model hooks created with `register_gradient_release`. Incompatible with `closure`.
        """
        raise NotImplementedError

    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True, param: Tensor | None = None):
        """Resets the gradients of all optimized parameters or an individual parameter.

        Args:
            set_to_none: If True, the gradients will be deallocated after the call (default: True).
            param: Resets the gradients of the supplied `param`. For use with `gradient_release=True`.
        """
        if param is None:
            super().zero_grad(set_to_none=set_to_none)
        else:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
