from transforms import OnlineMeanStd
from src.datasets.polyp import Polyp
import torchvision

ds = Polyp(transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
print(ds)
