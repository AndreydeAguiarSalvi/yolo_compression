import torch
from utils.datasets import LoadImages
from models import Darknet, SparseYOLO
from utils.utils import non_max_suppression
from utils.pruning import apply_mask_LTH, create_mask_LTH

ck_model = torch.load('weights/voc_yolov3/size-multi_scale/2020_03_22/15_54_35/best.pt', map_location='cuda:2')
ck_mask = torch.load('weights/voc_yolov3/size-multi_scale/2020_03_22/15_54_35/mask_1_prune.pt', map_location='cuda:2')

dataset = LoadImages('output/', img_size=416)

yolo = Darknet(cfg='cfg/voc_yolov3.cfg').to('cuda:2')
yolo.load_state_dict(ck_model['model'])
mask = create_mask_LTH(yolo)
mask.load_state_dict(ck_mask)

apply_mask_LTH(yolo, mask)
sparse = SparseYOLO(yolo)

path, img, im0s, _ = next(iter(dataset))
img = torch.from_numpy(img).to('cuda:2')
img = img.float()
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

yolo.eval()
sparse.eval()
# Inference
pred1 = yolo(img)[0]
pred2 = sparse(img)[0]

# Apply NMS
pred1 = non_max_suppression(pred1, 0,3, 0.6)
pred2 = non_max_suppression(pred2, 0.3, 0.6)

for i, (det1, det2) in enumerate(zip(pred1, pred2)):
    print(det1, det2)
    print(torch.abs(det1-det2))
