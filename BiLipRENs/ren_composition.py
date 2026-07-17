import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Union, Callable, Any, Tuple, Sequence, List, Optional

from flax import linen as nn
from flax.linen import initializers

from BiLipRENs.orthogonal_layer import Orthogonal, DynOrthogonal
from BiLipRENs.ren_model import BiLipREN

jax.config.update("jax_default_matmul_precision", "highest")

class StateCMapping(nn.Module):
    hidden_units: list
    m: int
    n: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_units[0])(x)
        x = nn.relu(x)
        x = nn.Dense(self.m * self.n+1)(x)
        return x[:,:-1], x[:,-1]

class StateCMappingInit(nn.Module):
    hidden_units: List[int]
    h0_dim: int
    y_min_bias_init: Optional[float] = 0.0

    @nn.compact
    def __call__(self, x):
        for i, units in enumerate(self.hidden_units):
            x = nn.Dense(features=units, name=f"hidden_layer_{i}")(x)
            x = nn.relu(x)      
        shared_output = x

        h0_head = nn.Dense(
            features=self.h0_dim, 
            name="h0_head"
        )
        h0 = h0_head(shared_output)

        ymin_head = nn.Dense(
            features=1,
            name="ymin_head",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(self.y_min_bias_init)
        )
        y_min = ymin_head(shared_output)
        y_min = jnp.squeeze(y_min, axis=-1)

        return h0, y_min

class CompREN(nn.Module):
    nu: int
    nx: int
    nv: int
    num_layers: int
    lower_bound: float
    upper_bound: float
    dyn_orth: bool = False
    dyn_orth_state_multiplier: int = 50
    dyn_orth_at_output: bool = False  # place DynOrth after OutOrth (output side)
    init_mode: str = "random"  # "random" (default) or "identity" for BiLipREN inner blocks
    
    def setup(self):
        lower = self.lower_bound**(1/self.num_layers)
        upper = self.upper_bound**(1/self.num_layers)
        models = []
        # Only prepend one dynamic orthogonal layer at the input when enabled.
        # The remaining stacked blocks keep the static orthogonal + REN pattern.
        if self.dyn_orth:
            orth_layer_in = DynOrthogonal((self.dyn_orth_state_multiplier*self.nx, self.nu))
        else:
            orth_layer_in = Orthogonal()
        models.append(orth_layer_in)

        for i in range(self.num_layers):
            ren_layer = BiLipREN(self.nu, self.nx, self.nv, lower, upper,
                                 init_mode=self.init_mode)
            orth_layer = Orthogonal()
            models.append([ren_layer, orth_layer])   
        orth_layer_out = Orthogonal()
        models.append(orth_layer_out)

        if self.dyn_orth_at_output:
            dyn_orth_out = DynOrthogonal((self.dyn_orth_state_multiplier*self.nx, self.nu))
            models.append(dyn_orth_out)

        self.models = models  # List of layers  
        
    def __call__(self, state: jnp.array, inputs: jnp.array, return_jacobians:bool = False) -> Tuple[jnp.array, jnp.array]:
        new_state = [[] for _ in range(self.num_layers)]
        if self.dyn_orth:
            orth_state, output = self.models[0](state[0][1], inputs)
            new_state[0].append(orth_state)
        else:
            output = self.models[0](inputs)
        jacobians = [] if return_jacobians else None

        for i in range(1, self.num_layers+1):
            ren_layer = self.models[i][0]
            if return_jacobians:
                # Use the analytic IFT-based Jacobian (Jac=True) instead of
                # autodiff so no second backward pass is needed during training.
                ren_state, output, J_ren = ren_layer(state[i-1][0], output, inv=False, Jac=True)
                jacobians.append({'layer': i, 'type': 'REN', 'jacobian': J_ren})
            else:
                ren_state, output = ren_layer(state[i-1][0], output)
            if self.dyn_orth and i == 1:
                new_state[i-1].insert(0, ren_state)
            else:
                new_state[i-1].append(ren_state)
            orth_layer = self.models[i][1]
            output = orth_layer(output)
        if self.dyn_orth_at_output:
            # models[-2] = OutOrth, models[-1] = DynOrth (output delay)
            out = self.models[-2](output)
            dyn_orth_state_in = state[self.num_layers - 1][1]
            new_dyn_state, out = self.models[-1](dyn_orth_state_in, out)
            new_state[self.num_layers - 1].append(new_dyn_state)
        else:
            out = self.models[-1](output)
        if return_jacobians:
            return new_state, out, jacobians
        else:
            return new_state, out


# Backward-compatible alias used across experiment scripts.
ORTHREN = CompREN