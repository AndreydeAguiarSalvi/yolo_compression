# yolo tiny
#python test_pascal.py --cfg cfg/voc_yolov3-tiny.cfg --weights weights/voc_yolov3-tiny/size-multi_scale/2020_03_30/10_14_23/best.pt --output results/test/voc_yolov3_tiny_0/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3-tiny.cfg --weights weights/voc_yolov3-tiny/size-multi_scale/2020_03_31/01_17_27/best.pt --output results/test/voc_yolov3_tiny_1/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3-tiny.cfg --weights weights/voc_yolov3-tiny/size-multi_scale/2020_03_31/15_08_25/best.pt --output results/test/voc_yolov3_tiny_2/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3-tiny.cfg --weights weights/voc_yolov3-tiny/size-multi_scale/2020_04_01/07_03_47/best.pt --output results/test/voc_yolov3_tiny_3/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3-tiny.cfg --weights weights/voc_yolov3-tiny/size-multi_scale/2020_04_02/00_37_26/best.pt --output results/test/voc_yolov3_tiny_4/results/VOC2012/Detection/ --device 1
# yolo 
#python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_04_02/17_14_20/best.pt --output results/test/voc_yolov3_0/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_21/11_16_25/best.pt --output results/test/voc_yolov3_1/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_04_05/10_02_02/best.pt --output results/test/voc_yolov3_2/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_26/11_04_40/best.pt --output results/test/voc_yolov3_3/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_28/09_08_14/best.pt --output results/test/voc_yolov3_4/results/VOC2012/Detection/ --device 1
# yolo IMP LOCAL
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_07/18_43_16/best.pt --mask_weight weights/voc_yolov3/size-multi_scale/2020_03_07/18_43_16/mask_1_prune.pt --output results/test/voc_yolov3_imp_local_0/results/VOC2012/Main/ --device 1
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_16/13_19_34/best.pt --mask_weight weights/voc_yolov3/size-multi_scale/2020_03_16/13_19_34/mask_1_prune.pt --output results/test/voc_yolov3_imp_local_1/results/VOC2012/Main/ --device 1
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_25/09_08_46/last.pt --mask_weight weights/voc_yolov3/size-multi_scale/2020_03_25/09_08_46/mask_1_prune.pt --output results/test/voc_yolov3_imp_local_2/results/VOC2012/Main/ --device 1
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_04_01/20_59_26/best.pt --mask_weight weights/voc_yolov3/size-multi_scale/2020_04_01/20_59_26/mask_1_prune.pt --output results/test/voc_yolov3_imp_local_3/results/VOC2012/Main/ --device 1
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_04_10/08_10_00/best.pt --mask --output results/test/voc_yolov3_imp_local_4/results/VOC2012/Main/ --device 1
# yolo IMP GLOBAL
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_22/15_54_35/best.pt --mask_weight weights/voc_yolov3/size-multi_scale/2020_03_22/15_54_35/mask_1_prune.pt --output results/test/voc_yolov3_imp_global_0/results/VOC2012/Main/ --device 1
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_18/18_00_18/best.pt --mask_weight weights/voc_yolov3/size-multi_scale/2020_03_18/18_00_18/mask_1_prune.pt --output results/test/voc_yolov3_imp_global_1/results/VOC2012/Main/ --device 1
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_03_31/17_20_13/best.pt --mask_weight weights/voc_yolov3/size-multi_scale/2020_03_31/17_20_13/mask_1_prune.pt --output results/test/voc_yolov3_imp_global_2/results/VOC2012/Main/ --device 1
python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_04_09/11_29_14/best.pt --mask --output results/test/voc_yolov3_imp_global_3/results/VOC2012/Main/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3.cfg --weights weights/voc_yolov3/size-multi_scale/2020_04_17/11_31_07/best.pt --mask --output results/test/voc_yolov3_imp_global_4/results/VOC2012/Main/ --device 1
# yolo CS
#python test_pascal.py --cfg cfg/voc_yolov3_soft.cfg --weights  --output results/test/voc_yolov3_/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3_soft.cfg --weights  --output results/test/voc_yolov3_/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3_soft.cfg --weights  --output results/test/voc_yolov3_/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3_soft.cfg --weights  --output results/test/voc_yolov3_/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3_soft.cfg --weights  --output results/test/voc_yolov3_/results/VOC2012/Detection/ --device 1
# yolo nano
#python test_pascal.py --cfg cfg/voc_yolov3-nano.cfg --weights weights/voc_yolov3_nano/size-multi_scale/2020_04_03/21_44_20/best.pt --output results/test/voc_yolov3_nano_0/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3-nano.cfg --weights weights/voc_yolov3_nano/size-multi_scale/2020_04_05/09_07_33/best.pt --output results/test/voc_yolov3_nano_1/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3-nano.cfg --weights weights/voc_yolov3_nano/size-multi_scale/2020_04_06/16_54_26/best.pt --output results/test/voc_yolov3_nano_2/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3-nano.cfg --weights weights/voc_yolov3_nano/size-multi_scale/2020_04_07/23_22_59/best.pt --output results/test/voc_yolov3_nano_3/results/VOC2012/Detection/ --device 1
#python test_pascal.py --cfg cfg/voc_yolov3-nano.cfg --weights weights/voc_yolov3_nano/size-multi_scale/2020_04_09/05_49_30/best.pt --output results/test/voc_yolov3_nano_4/results/VOC2012/Detection/ --device 1
