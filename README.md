# FAIM-RL

Implementation of "FAIM-RL: A Reinforcement Learning Approach for Fairness-aware Adaptive Influence Maximization"

Run the code
------------

#### Train FAIM model

	python main.py --graph train_data \
                     --model GCN_MLP_Model \
                     --budget 5 \
                     --epoch 20000 \
                     --lr 0.001 \
                     --bs 64 \
                     --n_step 1

#### Test FAIM model

	python main.py --graph test_data\
                     --model GCN_MLP_Model \
                     --model_file FAIM.ckpt \
                     --budget 10 \
                     --test

Dependency requirement
----------------------

- Python 3.6.13
- NumPy 1.19.5
- PyTorch 1.10.1+cu102
- PyG (PyTorch Geometric) 2.0.3
- PyTorch Scatter 2.0.9
- Tqdm 4.64.0
- SciPy 1.5.4

Code files
----------

- main.py: load program arguments, graphs and set up RL agent and environment.
- runner.py: conduct simulation, train and test RL agent.
- models.py: define parameters and structures of FAIM.  
- rl_agents.py: define agents to follow reinforcement learning procedure.
- environment.py: store the process of simulation.  
- utils/graph_utils.py: utility functions to load graphs.   

