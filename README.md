# 270A-Project

Reproduce the following paper:

> Jain, Prateek, Raghu Meka, and Inderjit Dhillon. "[Guaranteed rank minimization via singular value projection](https://proceedings.neurips.cc/paper/2010/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf)." Advances in Neural Information Processing Systems 23 (2010).

## Dependencies

### install miniconda
https://docs.conda.io/en/latest/miniconda.html

### create virtual environment
Activate the base environment of conda. Then execute the following:
```
conda create -n env_name_of_your_choice python=3.10
conda activate env_name_of_your_choice
```

### install torch
https://pytorch.org/get-started/locally/

### install other depandencies
pip install -r requirements.txt


## Run
```python
python run.py --config=example/random_matrix.py 
```
