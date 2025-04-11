from torchvision.datasets import Imagenette as Imagenette_TV
from torchvision import transforms as transforms_tv

class Imagenette(Imagenette_TV):
    def __init__(self, root: str, train: bool = True, dim: int = 224):
        self.dim = dim
        assert self.dim == 224
        super().__init__(
            root=root,
            split='train' if train else 'val',
            size='full',
            download=True,
            transform=None,
            target_transform=None
        )
    
    def enable_data_transforms(self, set: bool = True, aug: bool = False):
        if set:
            transforms = []
            if aug:
                transforms.extend([
                    transforms_tv.RandomResizedCrop(self.dim),
                    transforms_tv.RandomHorizontalFlip()
                ])
            else:
                transforms.extend([
                    transforms_tv.Resize(256),
                    transforms_tv.CenterCrop(self.dim)
                ])
            transforms.extend([
                transforms_tv.ToTensor(),
                transforms_tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.transform = transforms_tv.Compose(transforms)
        else:
            self.transform = None
            self.target_transform = None