# Simulated-Annealing
This is a script for simulated annealing, it builds on from the stochastic hill climbing algorithm.

# How to use it

## 1. Create a virtual environment:
```python
python -m venv .venv
```

the activate you virtual environment:
Run:

```python
.venv\Scripts\Activate
```
But if you dont see (venv) in bracket in your terminal, run:

```python
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .venv\Scripts\Activate
```

The virtual environment will allow you to run the project is a environment with all the dependecies need.

## 2. Clone github repo

Next clone the github repo, this will download all the needed files into your workspace:

```python
git clone https://github.com/KaraboLetsholo/Simulated-Annealing.git
```

## 3. Install the Requirements

Install all the need dependecies and requirements like JAX and wandb.

```python
pip install -r requirements.txt
```

## 4. Run the script

```
python Simulated_Annealing.py
```

### Results

## for $f(x) = x^2$:

<img width="861" height="315" alt="image" src="https://github.com/user-attachments/assets/467d1da1-1e7b-4dbc-9f28-84ca61d5f602" />

For a thousand iterations.

To change the objective function, change the objective_function() function in Stochastic_Hill_Climbing.py and change the parameters to match the dimension. Also create a wandb account to see the results or you can uncomment the print statements. You can also change the cooling schedule and the probability distribution from which a sample is drawn.

go to ```https://wandb.ai/authorize``` and copy the api key to paste it.

