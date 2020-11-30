wget -O p1_vgg19_bn.pth https://www.dropbox.com/s/3j2bu9ur0zd26lg/p1_current_best-70.72.pth?dl=0
python3 p1_save_prediction.py p1_vgg19_bn.pth $1 $2