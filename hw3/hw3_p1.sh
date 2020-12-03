# TODO: create shell script for running your VAE model

# Example
#python3 p1.py $1 
wget -O p1_fixed_noise_bash.pt https://www.dropbox.com/s/51og50w3h60lc3a/p1_fixed_noise.pt?dl=0
wget -O p1_model_bash.pth https://www.dropbox.com/s/ykgdh513xxpz5yr/p1_latest.pth?dl=0
python3 p1_bash.py p1_fixed_noise_bash.pt p1_model_bash.pth $1