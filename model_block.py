import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

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

        self.infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])    

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
            torch.cuda.empty_cache()
            self.disposed = True


    def _conv_image(self, img):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img = img.astype(np.float32)
        h,w = img.shape[:2]

        img = self.infer_transform(img)

        # pad to width and hight to 16 times
        if img.shape[1] % 16 != 0:
            times = img.shape[1]//16       
            top_pad = (times+1)*16 -img.shape[1]
        else:
            top_pad = 0

        if img.shape[2] % 16 != 0:
            times = img.shape[2]//16                       
            right_pad = (times+1)*16-img.shape[2]
        else:
            right_pad = 0 

        img = F.pad(img,(0,right_pad, top_pad,0)).unsqueeze(0)

        self.log(f"Original shape: {(h,w)}, padding: {(top_pad, right_pad)}, new shape: {img.shape}")

        return img.to(self.device), top_pad, right_pad

    def test(self, left_vpp, right_vpp):
        #Input conversion
        left_vpp, top_pad, right_pad = self._conv_image(left_vpp)
        right_vpp, _, _ = self._conv_image(right_vpp)

        #left_vpp = Variable(torch.FloatTensor(left_vpp))
        #right_vpp = Variable(torch.FloatTensor(right_vpp))

        self.model.eval()
        with torch.no_grad():
            pred_disp = self.model(left_vpp, right_vpp)
            pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

            if top_pad !=0 or right_pad != 0:
                pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-right_pad] 
            
            return pred_disp