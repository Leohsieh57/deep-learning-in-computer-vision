# TODO: create shell script for running your VAE model

# Example
#python3 p1.py $1 
wget -O p2_fixed_noise_bash.pt https://www.dropbox.com/s/gn5coiinci1qy5t/p2_fixed_noise.pt?dl=0
wget -O p2_model_bash.pth https://www.dropbox.com/s/ymkvmelxpx2ea7r/p2_latest_G.pth?dl=0
python3 p2_bash.py p2_fixed_noise_bash.pt p2_model_bash.pth $1