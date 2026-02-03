This project functions mainly as a folder structure and experiment manager. The purpose is for it to be general enough to add any RL environment and use any RL algorithm implementation, while retaining the same folder structure.
### To install
Project requires Python 3.12

Run the following commands inside after cloning:
1. `sudo apt install swig build-essential python3-dev`
2. `git submodule init`
3. `git submodule update`
4. `cd VectorizedMultiAgentSimulator`
5. `pip install -e .`
6. `cd ../src`
7. `pip install -e .`
8. `cd ..`
9. `pip install -r requirements.txt`

### To build an experiment

Run the jupyter notebook `src/learning/builders/experiment_builder.ipynb`

### To run experiment

Run the following command for running experiments in different modalities:

- `python3 src/run_trial.py --batch=salp_navigate_8a --name=gcn --algorithm=ppo --environment=salp_navigate`

Run N number of trials in parallel (Requires GNU Parallel Package)

- `parallel bash run_trial.sh salp_navigate_8a ppo salp_navigate test_id ::: gcn gat graph_transformer transformer_full transformer_encoder transformer_decoder`
