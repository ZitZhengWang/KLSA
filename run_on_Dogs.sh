# 在斯坦福狗数据集上进行实验
# 首先预训练一个网络,
python train_classifier.py --config "configs/train_classifier_folder_localpatch_dog.yaml" --tag "Dogs"

# 做进一步的实验
# 虽然名称中美由 woWeight，但是为了不让名称太长，就不写了；以后均默认没有加权模块
python train_meta.py --config "configs/train_meta_Dog_5shot_localpatch.yaml" --name "full_pSize26_thd2_k5_pNum36_dogs" --tag "5_CR"
python test_few_shot.py --config "configs/test_few_shot_Dogs5shot_localpatch.yaml" --loadpath "save/full_pSize26_thd2_k5_pNum36_dogs_5_CR/max-va.pth" --patchNum 36

python train_meta.py --config "configs/train_meta_Dog_1shot_localpatch.yaml" --name "full_pSize26_thd2_k5_pNum36_dogs" --tag "1_CR"
python test_few_shot.py --config "configs/test_few_shot_Dogs1shot_localpatch.yaml" --loadpath "save/full_pSize26_thd2_k5_pNum36_dogs_1_CR/max-va.pth" --patchNum 36
