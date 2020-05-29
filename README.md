# RecGAN_Implement
## Usage: 
**In the directory of IRecGAN**, type command: 

python main.py --click ../main/temp_data/gen_click.txt --reward ../main/temp_data/gen_reward.txt --action ../main/temp_data/gen_action.txt --model LSTM --nhid 128 --n_layers_usr 2 --optim_nll adam --optim_adv adam --batch_size 128

Link to paper: https://openreview.net/forum?id=SJxSlrHeLS

implement fo RecGAN paper
requirement:
Python 2.7
Numpy 1.14.5
Tensorflow 1.3.1