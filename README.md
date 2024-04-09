## Setup

Should be on a linux computer. Install RAxML-NG, and then create a new python environment. Then,
run `pip install -r requirements.txt` to install needed packages.

Note: to run this program on docker without installation, run `setup.sh`

## Basics

All commands start with `python main.py -m [MODE]`
The different modes are:
- Test: For testing the NNs and calculating metrics. Command: `python main.py -m test -l [location of test data] -n [location of neural networks to test]`
- Train: To train neural networks. Command: `python main.py -m train -l [location of training data] -o [location of neural network output]`
- Complete: Similar to training neural networks, but with hyperparameter tuning. Command: `python main.py -m complete -l [location of training data] -o [location of neural network output]`
- Algorithm: Run the algorithm with trained networks. Command: `python main.py -m train -l [location of data] -n [location of neural networks]` 

If you do not have RAxML-NG installed on your computer, you may use the `-w` flag at the end of any of the commands to skip RAxML-NG. This will set values calculated by RAxML-NG to 0, and therefore should only be used for debugging purposes.
