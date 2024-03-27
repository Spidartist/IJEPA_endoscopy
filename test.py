# from src.masks.multiblock import MaskCollator as MBMaskCollator
# import torchvision
# from src.datasets.polyp import make_polyp
# from src.datasets.imagenet1k import make_imagenet1k
# import torch


# mask_collator = MBMaskCollator(patch_size=16)
# g = torch.Generator()
# g.manual_seed(43)
# print(mask_collator._sample_block_size(generator=g, 
#                                        scale=mask_collator.pred_mask_scale,
#                                        aspect_ratio_scale=mask_collator.aspect_ratio
#                                        ))
# print(mask_collator.width)

# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor()])

# _, unsupervised_loader = make_polyp(
#     transform=transform,
#     batch_size=1,
#     collator=mask_collator,
#     num_workers=2,
#     training=True)


# for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
#     print(udata[0].shape)
#     print(masks_enc[0].shape)
#     print(masks_pred[0].shape)

#     break

from src.transforms import OnlineMeanStd
from src.datasets.polyp import Polyp
import torchvision
transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
ds = Polyp(transform=transform)
print(len(ds))
mean_std_calculator = OnlineMeanStd()
mean, std = mean_std_calculator(dataset=ds, batch_size=16)
print(mean, std)