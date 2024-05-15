# 在斯坦福 车 数据集上进行实验
# 首先预训练一个网络,
python train_classifier.py --config "configs/train_classifier_folder_localpatch_car.yaml" --tag "Car"

# 做进一步的实验
python train_meta.py --config "configs/train_meta_Car_5shot_localpatch.yaml" --name "full_pSize26_thd2_k5_pNum36_car" --tag "5_CR"
python test_few_shot.py --config "configs/test_few_shot_Car5shot_localpatch.yaml" --loadpath "save/full_pSize26_thd2_k5_pNum36_car_5_CR/max-va.pth" --patchNum 36

python train_meta.py --config "configs/train_meta_Car_1shot_localpatch.yaml" --name "full_pSize26_thd2_k5_pNum36_car" --tag "1_CR"
python test_few_shot.py --config "configs/test_few_shot_Car1shot_localpatch.yaml" --loadpath "save/full_pSize26_thd2_k5_pNum36_car_1_CR/max-va.pth" --patchNum 36
