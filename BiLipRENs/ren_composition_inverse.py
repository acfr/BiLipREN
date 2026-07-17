import jax
import jax.numpy as jnp
from typing import Tuple

from flax import linen as nn

from BiLipRENs.orthogonal_layer import Orthogonal
from BiLipRENs.ren_model import BiLipREN

jax.config.update("jax_default_matmul_precision", "highest")


class CompRENinv(nn.Module):
    nu: int
    nx: int
    nv: int
    num_layers: int
    lower_bound: float
    upper_bound: float
    dyn_orth: bool = False
    init_mode: str = "random"  # "random" or "identity" for BiLipREN inner blocks
    
    def setup(self):
        lower = self.lower_bound ** (1 / self.num_layers)
        upper = self.upper_bound ** (1 / self.num_layers)

        # First inverse operation corresponds to the output orth layer.
        self.output_orth = Orthogonal()

        # Inverse of each forward block (reversed order via reverse_params):
        # static orth inverse, then REN inverse.
        self.block_orth = [Orthogonal() for _ in range(self.num_layers)]
        self.block_ren = [
            BiLipREN(self.nu, self.nx, self.nv, lower, upper,
                     init_mode=self.init_mode)
            for i in range(self.num_layers)
        ]

        # Include inverse of forward input orth only when dyn_orth=False.
        self.input_orth = Orthogonal() if not self.dyn_orth else None
        
    def __call__(self, state: jnp.array, inputs: jnp.array) -> Tuple[jnp.array, jnp.array]:
        new_state = [[] for _ in range(self.num_layers)]

        output = self.output_orth(inputs, inv=True)
        for i in range(self.num_layers):
            orth_layer = self.block_orth[i]
            output = orth_layer(output, inv=True)

            ren_layer = self.block_ren[i]
            ren_state, output = ren_layer(state[i][0], output, inv=True)
            new_state[i].append(ren_state)

        if self.input_orth is not None:
            output = self.input_orth(output, inv=True)

        return new_state, output
    
    @staticmethod
    def reverse_params(params, num_layers, dyn_orth: bool = False):
        """Reverse ORTHREN params into ORTHRENinv order.

        For forward ORTHREN params:
          - models_0: input orth (dynamic when dyn_orth=True)
          - models_1..models_num_layers: [ren, orth]
          - models_{num_layers+1}: output orth

        For inverse ORTHRENinv params:
          - models_0: output orth
          - models_1..models_num_layers: [orth, ren] (reversed block order)
          - models_{num_layers+1}: input orth (only when dyn_orth=False)
        """
        old = params["params"]
        new_params = {}

        # output orth -> first inverse module
        new_params["output_orth"] = old[f"models_{num_layers + 1}"]

        # reverse internal blocks and swap [ren, orth] -> [orth, ren]
        for i in range(num_layers):
            src_block_idx = num_layers - i
            new_params[f"block_orth_{i}"] = old[f"models_{src_block_idx}_1"]
            new_params[f"block_ren_{i}"] = old[f"models_{src_block_idx}_0"]

        # include original input orth only when dyn_orth=False
        if not dyn_orth:
            new_params["input_orth"] = old["models_0"]

        return {"params": new_params}