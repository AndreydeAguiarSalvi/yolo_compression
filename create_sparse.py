import torch
from utils.datasets import LoadImages
from models import Darknet, SparseYOLO, SoftDarknet
from utils.utils import non_max_suppression
from utils.pruning import apply_mask_LTH, create_mask_LTH

is_CS = True
device = 'cpu' if is_CS else 'cuda:2'
if is_CS:
    ck_model = torch.load('weights/voc_yolov3_soft_orig-output/size-multi_scale/2020_05_13/19_14_12/best_it_1.pt', map_location=device)
else:
    ck_model = torch.load('weights/voc_yolov3/size-multi_scale/2020_03_22/15_54_35/best.pt', map_location=device)
    ck_mask = torch.load('weights/voc_yolov3/size-multi_scale/2020_03_22/15_54_35/mask_1_prune.pt', map_location=device)

dataset = LoadImages('output/', img_size=416)

path, img, im0s, _ = next(iter(dataset))
img = torch.from_numpy(img).to(device)
img = img.float()
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

if is_CS: 
    yolo = SoftDarknet(cfg='cfg/voc_yolov3_soft_orig-output.cfg').to(device)
    yolo.load_state_dict(ck_model['model'])
    yolo.ticket = True
    _ = yolo(img)
else: 
    yolo = Darknet(cfg='cfg/voc_yolov3.cfg').to(device)
    yolo.load_state_dict(ck_model['model'])
    mask = create_mask_LTH(yolo)
    mask.load_state_dict(ck_mask)
    apply_mask_LTH(yolo, mask)

sparse = SparseYOLO(yolo).to(device)

yolo.eval()
sparse.eval()
# Inference
pred1 = yolo(img)[0]
pred2 = sparse(img)[0]

# Apply NMS
pred1 = non_max_suppression(pred1, 0.3, 0.6)
pred2 = non_max_suppression(pred2, 0.3, 0.6)

for i, (det1, det2) in enumerate(zip(pred1, pred2)):
    print(det1, det2)
    print(torch.abs(det1-det2))
