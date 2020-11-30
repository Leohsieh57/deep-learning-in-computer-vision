wget -O p2_fcn32.pth https://www.dropbox.com/s/krcnlckl0n6sy4q/p2_current_best_fcn32-67.94.pth?dl=0
python3 p2_save_prediction.py p2_fcn32.pth $1 $2