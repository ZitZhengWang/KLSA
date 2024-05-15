
python train_classifier.py --config "configs/train_classifier_folder_localpatch.yaml" --tag "CUB"

python train_meta.py --config "configs/train_meta_CUB_5shot_localpatch.yaml" --tag "CUB"
python test_few_shot.py --config "configs/test_few_shot_CUB5shot_localpatch.yaml"

python train_meta.py --config "configs/train_meta_CUB_1shot_localpatch.yaml" --tag "CUB"
python test_few_shot.py --config "configs/test_few_shot_CUB1shot_localpatch.yaml"

