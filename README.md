# DeepCIP
DeepCIP is a **D**eep learning method for **C**ircRNA **I**RES **P**rediction.

## The ViennaRNA package installation
RNAplfold from ViennaRNA version 2.5.1 is required to predict RNA secondary structure in DeepCIP. You need to install the ViennaRNA package before you start to use DeepCIP.

First, download the ViennaRNA package from https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_5_x/ViennaRNA-2.5.1.tar.gz .

Then, install ViennaRNA package:
```
tar -zxvf ViennaRNA-2.5.1.tar.gz
cd ViennaRNA-2.5.1
./configure
make
make install
```
For more details, see https://github.com/ViennaRNA/ViennaRNA

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
The file prepared for prediction can be put into folder ./data or any other path. 
Then, type the following command to start the prediction, with parameters set according to your needs.

For example:
```
python DeepCIP.py --data_name example --input_file ./data/human_circires.fa --outfile ./results/predict.out
```
For more options:
```
python DeepCIP.py --help
```
```
usage: DeepCIP.py [-h] [--data_name] [--input_file] [--outfile] [--bs] [--c] [--mode] [--w] [--s]

optional arguments:
  -h, --help            show this help message and exit
  --data_name           The name of your input dataset.
  --input_file
                        Input file for prediction. (*.fasta or *.fa file)
  --outfile             Output file storage path.
  --bs                  Batch size. (default=32) "--bs 32" means every 32 sampels constitute a prediction batch. This parameter affects the speed and results of the 
                        prediction. The larger the batch size, the faster the prediction, as far as your machine allows. (If the lengths of your input sequences vary 
                        greatly, it is recommended that you do not use a large batch size, or you can put sequences of similar lengths together for prediction)
  --c                   Prediction threshold. (default=0.5)
  --mode                The mode of prediction. (default=0) mode 0: Prediction directly on the input sequence. mode 1: The input sequence is partitioned by length w 
                        and interval s, and then the partitioned sequence is predicted. (w and s can be set by --w and --s, respectively)
  --w                   window size (default=174). See --mode description. It can be ignore when mode=0.
  --s                   step (default=50). See --mode description. It can be ignore when mode=0.
  ```
