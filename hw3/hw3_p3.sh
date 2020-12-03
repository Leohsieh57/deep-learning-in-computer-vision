wget -O mnistm.pth https://www.dropbox.com/s/dxclm0e0hnh4juv/p3_usps-_mnistm_final.pth?dl=0
wget -O svhn.pth https://www.dropbox.com/s/bexgdw7s8aln6bi/p3_mnistm-_svhn_final.pth?dl=0
wget -O usps.pth https://www.dropbox.com/s/0xbr1r3s40sh305/p3_svhn-_usps_final.pth?dl=0
python3 p3_bash.py $1 $2 $3