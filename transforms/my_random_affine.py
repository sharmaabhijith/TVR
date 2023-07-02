import torchvision
from torch import Tensor
from torchvision.transforms import functional as F


class MyRandomAffine(torchvision.transforms.RandomAffine):
    def forward(self, img, mask):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                try:
                    fill = [float(fill)] * F.get_image_num_channels(img)
                except:
                    fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        try:
            img_size = F.get_image_size(img)
        except:
            img_size = F._get_image_size(img)
        ret = self.get_params(self.degrees, self.translate, self.scale,
                              self.shear, img_size)
        transf_img = F.affine(img, *ret, interpolation=self.interpolation,
                              fill=fill)
        transf_mask = F.affine(mask, *ret, interpolation=self.interpolation,
                               fill=fill)
        return transf_img, transf_mask
