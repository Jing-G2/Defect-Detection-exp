python main.py --data_name='KSDD' \
    --setting='knn' \
    --epochs=200 \
    --weight_decay=0 \
    --prop_epochs=2 \
    --knn=3


# echo "--AUG-RE--"
# python main.py --data_name='KSDD' \
# --setting='aug' --aug='RE' \
#  --epochs=1000 --weight_decay=0 --aug_ratio=0.01


# echo "--AUG-DN--"
# python main.py --data_name='KSDD' \
# --setting='aug' --aug='DN' \
#  --epochs=1000 --weight_decay=0 --aug_ratio=0.01


# echo "--knn_aug-RE--"
# python main.py --data_name='KSDD' \
# --setting='knn_aug' --aug='RE' \
#  --epochs=1000 --weight_decay=0  --aug_ratio=0.01 --prop_epochs=3 --knn=3


# echo "--knn_aug-DN--"
# python main.py --data_name='KSDD' \
# --setting='knn_aug' --aug='DN' \
#  --epochs=1000 --weight_decay=0  --aug_ratio=0.01 --prop_epochs=3 --knn=3
