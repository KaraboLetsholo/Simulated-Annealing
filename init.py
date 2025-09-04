import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.lax as lax

print("Imports completed successfully.")
print(f"num_devices: {jax.device_count()}")