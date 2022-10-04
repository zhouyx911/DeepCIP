# DeepCIP
DeepCIP is a **D**eep learning method for **C**ircRNA **I**RES **P**rediction.

## Installation of DeepCIP and its environment
First, download the repository and create the environment.
```
git clone https://github.com/zjupgx/DeepCIP.git
cd ./DeepCIP
conda env create -f environment.yml
```
Then, activate the "DeepCIP_pytorch" environment.
```
conda activate DeepCIP_pytorch
```

## Usage
### Run DeepCIP for circRNA IRES prediction
Example:
```
python DeepCIP.py --data_name example --input_file ./data/human_circires.fa --outfile ./results/predict.out
```
For more options:
```
python DeepCIP.py --help
```
