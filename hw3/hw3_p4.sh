wget -O mnistm.pth https://www.dropbox.com/s/01u7vr9azhz9xlf/p4_usps-_mnistm_final.pth?dl=0
wget -O svhn.pth https://www.dropbox.com/s/1ae4z1lwycj0buq/p4_mnistm-_svhn_final.pth?dl=0
wget -O usps.pth https://www.dropbox.com/s/vhtyjsdu4zrzatv/p4_svhn-_usps_final.pth?dl=0
python3 p4_bash.py $1 $2 $3