wget -O p2_fcn16.pth https://www.dropbox.com/s/nji3zqf89jjmi5w/p2_current_best_fcn16-69.03.pth?dl=0
python3 p2_save_prediction.py p2_fcn16.pth $1 $2