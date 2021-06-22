import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from PIL import Image

# from .model import Net
from .model_new import ft_net as Net

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(751)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path)
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.eval() # change the model to mode evaluate
        self.size = (256, 128)
        self.norm = transforms.Compose([
            # transforms.Resize((384,192), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            # print("_resize im : ", im.shape)
            return cv2.resize(im, size)
        # print("_preprocess im_crops : ", np.array(im_crops).shape)
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        # print("_preprocess im_batch : ", im_batch.shape)
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            # print("im_batch size: {}".format(im_batch.shape))
            _, features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint\\net_last.pth")
    feature = extr([img])
    print(feature.shape)

