#!/bin/bash
# declare an array called array and define 3 vales

experimatn_folder=./experiment_deeplab3_ddcat/
mkdir $experimatn_folder

scripts=(./test_normal.py ./test_base_cosine.py ./test_base_cosine_combination.py ./test_base_line_fgsm.py ./test_base_line_pgd.py)

for i in "${!scripts[@]}"
do
        python3 "${scripts[$i]}" --config ./config/conf_deeplabv3_ddcat.yaml
done

echo "-------------------------Finished-----------------------------"
