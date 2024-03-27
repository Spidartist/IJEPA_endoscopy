import torch
import src.models.vision_transformer as vit


DEVICE = torch.device("cuda:2")
ckpt = torch.load("./logs/jepa-latest.pth.tar").to(DEVICE)
print(ckpt["lr"])

model = vit.__dict__["vit_base"](img_size=[224], patch_size=14).to(DEVICE)