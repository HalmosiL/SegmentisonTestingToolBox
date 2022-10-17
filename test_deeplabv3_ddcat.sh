#!/bin/bash
# declare an array called array and define 3 vales

experimatn_folder=./experiment_deeplab3_ddcat/
mkdir $experimatn_folder

scripts=(./test_base_cosine_combination.py ./test_base_line_fgsm.py ./test_base_line_pgd.py)
folders=(results_normal_500_image_valset results_cosine_step_120_e_0.03_500_image_valset results_combination_step_120_e_0.03_500_image_valset results_fgsm_step_120_e_0.03_500_image_valset results_pgd_step_120_e_0.03_500_image_valset)

for i in "${!scripts[@]}"
do
        python3 "${scripts[$i]}" --config ./config/conf_deeplabv3_ddcat.yaml
        mv ./results_ $experimatn_folder"${folders[$i]}"
done

echo "-------------------------Finished-----------------------------"
