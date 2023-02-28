import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import gc

import importlib
psmnet_models = importlib.import_module("thirdparty.PSMNet.models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class PSMNetBlock:
    def __init__(self, max_disparity = 192, model_type = "stackhourglass", device = "cpu", verbose=False):
        self.logName = "PSMNet Block"
        self.verbose = verbose

        # if max_disparity % 16 != 0:
        #     max_disparity = 16 * math.floor(max_disparity/16)
        #     max_disparity = int(max_disparity)

        self.max_disparity = max_disparity
        self.model_type = model_type
        self.device = device
        self.disposed = False

        self.infer_transform = transforms.Compose([transforms.ToTensor()])    

    def log(self, x):
        if self.verbose:
            print(f"{self.logName}: {x}")

    def build_model(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(f"Building Model...")
        if self.model_type == 'stackhourglass':
            self.model = psmnet_models.stackhourglass(self.max_disparity)
        elif self.model_type == 'basic':
            self.model = psmnet_models.basic(self.max_disparity)
        else:
            raise Exception("Model type not found")

        if not self.device == "cpu":
            self.model = nn.DataParallel(self.model).to(self.device)

    def load(self, model_path):
        # load the checkpoint file specified by model_path.loadckpt
        self.log("loading model {}".format(model_path))
        
        pretrained_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    def dispose(self):
        if not self.disposed:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.disposed = True


    def _conv_image(self, img):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.uint8) / 255.0
        ht,wt = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
                
        # pad to width and hight to 16 times
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
        _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

        img = F.pad(img, _pad, mode='replicate')

        self.log(f"Original shape: {(ht,wt)}, padding: {_pad}, new shape: {img.shape}")

        return img.unsqueeze(0).to(self.device), _pad

    def test(self, left_vpp, right_vpp):
        #Input conversion
        left_vpp, _pad = self._conv_image(left_vpp)
        right_vpp, _ = self._conv_image(right_vpp)

        #left_vpp = Variable(torch.FloatTensor(left_vpp))
        #right_vpp = Variable(torch.FloatTensor(right_vpp))

        self.model.eval()
        with torch.no_grad():
            pred_disp = self.model(left_vpp, right_vpp)
            pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

            ht, wd = pred_disp.shape[-2:]
            c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
            pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]
            
            return pred_disp