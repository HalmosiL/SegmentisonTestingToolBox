#!/bin/bash
# declare an array called array and define 3 vales

experimatn_folder=./experiment/

scripts=(./test_normal.py ./test_base_cosine.py ./test_base_cosine_combination.py ./test_base_line_fgsm.py ./test_base_line_pgd.py)
folders=(results_normal_500_image_valset_psnetDDCAT results_cosine_step_120_e_0.03_500_image_valset_psnetDDCAT results_combination_step_120_e_0.03_500_image_valset_psnetDDCAT results_fgsm_step_120_e_0.03_500_image_valset_psnetDDCAT results_pgd_step_120_e_0.03_500_image_valset_psnetDDCAT)

for i in "${!scripts[@]}"
do
        CUDA_VISIBLE_DEVICES=1 python3 "${scripts[$i]}"
        mv -r ./results $experimatn_folder"${folders[$i]}"
done

echo "-------------------------Finished-----------------------------"
