echo "RAW"
python -m src.q3 --transform raw

echo && echo "blurred and equalised"
python -m src.q3 --transform blur_equal

echo && echo "edge detected"
python -m src.q3 --transform edge_detect

echo && echo "hog features"
python -m src.q3 --transform hog_feat
