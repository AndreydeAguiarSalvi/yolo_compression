# Tiny
python computing_flops.py --model weights/voc_yolov3-tiny/size-multi_scale/2020_03_30/10_14_23/best.pt --darknet default --cfg cfg/voc_yolov3-tiny.cfg --device cuda:3 | tee results/flops/tiny_0.txt
python computing_flops.py --model weights/voc_yolov3-tiny/size-multi_scale/2020_03_31/01_17_27/best.pt --darknet default --cfg cfg/voc_yolov3-tiny.cfg --device cuda:3 | tee results/flops/tiny_1.txt
python computing_flops.py --model weights/voc_yolov3-tiny/size-multi_scale/2020_03_31/15_08_25/best.pt --darknet default --cfg cfg/voc_yolov3-tiny.cfg --device cuda:3 | tee results/flops/tiny_2.txt
python computing_flops.py --model weights/voc_yolov3-tiny/size-multi_scale/2020_04_01/07_03_47/best.pt --darknet default --cfg cfg/voc_yolov3-tiny.cfg --device cuda:3 | tee results/flops/tiny_3.txt
python computing_flops.py --model weights/voc_yolov3-tiny/size-multi_scale/2020_04_02/00_37_26/best.pt --darknet default --cfg cfg/voc_yolov3-tiny.cfg --device cuda:3 | tee results/flops/tiny_4.txt

# V3
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_02/17_14_20/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --device cuda:3 | tee results/flops/v3_0.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_21/11_16_25/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --device cuda:3 | tee results/flops/v3_1.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_05/10_02_02/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --device cuda:3 | tee results/flops/v3_2.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_26/11_04_40/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --device cuda:3 | tee results/flops/v3_3.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_28/09_08_14/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --device cuda:3 | tee results/flops/v3_4.txt

# CS 1IT
# python computing_flops.py --model weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_05_13/19_14_12/best_it_1.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_1it_1.txt
# python computing_flops.py --model weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_05_13/19_37_54/best_it_1.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_1it_2.txt
# python computing_flops.py --model weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_05_18/12_09_53/best_it_1.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_1it_3.txt
# python computing_flops.py --model weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_05_22/14_45_57/best_it_1.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_1it_4.txt
# python computing_flops.py --model weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_05_23/18_05_17/best_it_1.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_1it_5.txt

# IMP LOCAL
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_07/18_43_16/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --mask weights/voc_yolov3/size-multi_scale/2020_03_07/18_43_16/mask_1_prune.pt --device cuda:3 | tee results/flops/lth_local_0.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_16/13_19_34/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --mask weights/voc_yolov3/size-multi_scale/2020_03_16/13_19_34/mask_1_prune.pt --device cuda:3 | tee results/flops/lth_local_1.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_25/09_08_46/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --mask weights/voc_yolov3/size-multi_scale/2020_03_25/09_08_46/mask_1_prune.pt --device cuda:3 | tee results/flops/lth_local_2.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_01/20_59_26/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --mask weights/voc_yolov3/size-multi_scale/2020_04_01/20_59_26/mask_1_prune.pt --device cuda:3 | tee results/flops/lth_local_3.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_10/08_10_00/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --embbed --device cuda:3 | tee results/flops/lth_local_4.txt

# IMP GLOBAL
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_22/15_54_35/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --mask weights/voc_yolov3/size-multi_scale/2020_03_22/15_54_35/mask_1_prune.pt --device cuda:3 | tee results/flops/lth_global_0.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_18/18_00_18/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --mask weights/voc_yolov3/size-multi_scale/2020_03_18/18_00_18/mask_1_prune.pt --device cuda:3 | tee results/flops/lth_global_1.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_03_31/17_20_13/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --mask weights/voc_yolov3/size-multi_scale/2020_03_31/17_20_13/mask_1_prune.pt --device cuda:3 | tee results/flops/lth_global_2.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_09/11_29_14/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --embbed --device cuda:3 | tee results/flops/lth_global_3.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_17/11_31_07/best.pt --darknet default --cfg cfg/voc_yolov3.cfg --embbed --device cuda:3 | tee results/flops/lth_global_4.txt

# Nano
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_03/21_44_20/best.pt --darknet nano --device cuda:3 | tee results/flops/nano_0.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_05/09_07_33/best.pt --darknet nano --device cuda:3 | tee results/flops/nano_1.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_06/16_54_26/best.pt --darknet nano --device cuda:3 | tee results/flops/nano_2.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_07/23_22_59/best.pt --darknet nano --device cuda:3 | tee results/flops/nano_3.txt
python computing_flops.py --model weights/voc_yolov3/size-multi_scale/2020_04_09/05_49_30/best.pt --darknet nano --device cuda:3 | tee results/flops/nano_4.txt


# CS 3IT
# python computing_flops.py --model weights/MotoZ/size-multi_scale/2020_05_13/11_06_00/best_it_3.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_3it_0.txt
# python computing_flops.py --model weights/MotoZ/size-multi_scale//2020_05_18/15_20_17/best_it_3.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_3it_1.txt
# python computing_flops.py --model weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_05_27/21_01_36/best_it_3.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_3it_2.txt
# python computing_flops.py --model weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_06_03/18_18_29/best_it_3.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_3it_3.txt
# python computing_flops.py --model weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_05_31/19_42_26/best_it_3.pt --darknet soft --cfg cfg/voc_yolov3_soft_orig-output.cfg --device cuda:3 | tee results/flops/cs_3it_4.txt
