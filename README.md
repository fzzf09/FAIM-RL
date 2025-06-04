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



### Dependency requirement

---

* **Python** 3.10+
* **NumPy** 1.19.5
* **PyTorch** 2.2.0 + CUDA 12.1
* **PyTorch Geometric** 2.6.1
* **PyTorch Scatter** 2.1.2 + CUDA 12.1
* **Tqdm** 4.67.1
* **SciPy** 1.13.1
* **NetworkX** 3.2.1
* **Gensim** 4.3.3
* **aiohttp** 3.11.18
* **Matplotlib** 3.9.2

---


Code files
----------

- main.py: load program arguments, graphs and set up RL agent and environment.
- runner.py: conduct simulation, train and test RL agent.
- models.py: define parameters and structures of FAIM.  
- rl_agents.py: define agents to follow reinforcement learning procedure.
- environment.py: store the process of simulation.  
- utils/graph_utils.py: utility functions to load graphs.   

