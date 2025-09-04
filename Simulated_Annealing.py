import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.lax as lax
import wandb

# --- IGNORE ---
#This is a script for simulated annealing, it builds
#on from the stochastic hill climbing algorithm.

wandb.init(project="nonlinear_optimization", name="simulated_annealing")

@jax.jit
def objective(params):
    return 1*jnp.sum((params) ** 2)

obj = lambda params: objective(params)

params = jnp.array([100.0])
cooling_schedule = jnp.linspace(1.0, 0.01, num=1000)

distribution = jax.random.uniform(jax.random.PRNGKey(0), shape=(1000,), minval=-1.0, maxval=1.0)
neighbor = params + distribution
new_param_value = jax.random.choice(jax.random.PRNGKey(0), a=neighbor, shape=(), replace=True)

count = 0

while count < 1000:
        criterion = jax.random.uniform(jax.random.PRNGKey(0), shape=(1,), minval=0, maxval=1.0)
        probability = jnp.exp((obj(params) - obj(new_param_value)) / cooling_schedule[count])
        if obj(new_param_value) < obj(params):
            params = new_param_value
            distribution = jax.random.uniform(jax.random.PRNGKey(0), shape=(1000,), minval=-1.0, maxval=1.0)
            neighbor = params + distribution
            new_param_value = jax.random.choice(jax.random.PRNGKey(0), a=neighbor, shape=(), replace=True)
            count += 1
            #print(f"Iteration: {count}, Objective: {obj(params)}, Params: {params}")
            wandb.log({"Iteration": count, "Objective": obj(params), "Params": params})
        elif probability > criterion:
            params = new_param_value
            distribution = jax.random.uniform(jax.random.PRNGKey(0), shape=(1000,), minval=-1.0, maxval=1.0)
            neighbor = params + distribution
            new_param_value = jax.random.choice(jax.random.PRNGKey(0), a=neighbor, shape=(), replace=True)
            count += 1
            #print(f"Iteration: {count}, Objective: {obj(params)}, Params: {params}")
            wandb.log({"Iteration": count, "Objective": obj(params), "Params": params})
        else:
            distribution = jax.random.uniform(jax.random.PRNGKey(0), shape=(1000,), minval=-1.0, maxval=1.0)
            neighbor = params + distribution
            new_param_value = jax.random.choice(jax.random.PRNGKey(0), a=neighbor, shape=(), replace=True)
            count += 1
            #print(f"Iteration: {count}, Objective: {obj(params)}, Params: {params}")
            wandb.log({"Iteration": count, "Objective": obj(params), "Params": params})