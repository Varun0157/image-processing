echo "RAW"
python -m src.q3 --transform raw --lr 5e-4

echo && echo "blurred and equalised"
python -m src.q3 --transform blur_equal --lr 5e-4

echo && echo "edge detected"
python -m src.q3 --transform edge_detect --lr 5e-4

echo && echo "hog features"
python -m src.q3 --transform hog_feat --lr 5e-4
