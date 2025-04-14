from torchvision.datasets import Imagenette as Imagenette_TV
from torchvision import transforms as transforms_tv

class Imagenette(Imagenette_TV):
    def __init__(self, root: str, train: bool = True, dim: int = 224):
        self.dim = dim
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
                    transforms_tv.RandomResizedCrop(self.dim, scale=(0.08, 0.1)),
                    transforms_tv.RandomHorizontalFlip(),
                    transforms_tv.RandAugment(num_ops=9, magnitude=5, interpolation=transforms_tv.InterpolationMode.BICUBIC)
                ])
            else:
                transforms.extend([
                    transforms_tv.Resize(int(256/224)*self.dim),
                    transforms_tv.CenterCrop(self.dim)
                ])
            transforms.extend([
                transforms_tv.ToTensor(),
                transforms_tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if aug:
                transforms.extend([
                    transforms_tv.RandomErasing(p=0.25)
                ])
            self.transform = transforms_tv.Compose(transforms)
        else:
            self.transform = None
            self.target_transform = None