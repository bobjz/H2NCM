# Set-up

Create conda environment with python version=3.7:
```console
conda create -n my_env python=3.7
```
After activating the created environment:
```console
conda activate my_env
```
Install the following packages with pip:
```console
pip install numpy torch torchvision
```

# Run experiments for MNODE
place the following data file in the same directory as MNODE.py
```console
cp /dfs/scratch1/bobjz/ICML_paper_data/*.npy .
```
run python3 MNODE.py to get results
```console
python3 MNODE.py
```
