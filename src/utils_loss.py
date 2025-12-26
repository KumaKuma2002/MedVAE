import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import datetime
import os



def get_segmentation(seg_model, estimated_image_for_nnunet, mini_batch=False):
    """
    Get the segmentation logits from reconstructed CT.
    The CT needs recover to HU.
    """
    
    # mini-batch to reduce memory used.
    if mini_batch:
        pred_logits = []
        for i in range(estimated_image_for_nnunet.size(0)):
            mini_slice = estimated_image_for_nnunet[i:i+1].float()
            pred = seg_model(mini_slice)
            pred_logits.append(pred)
        pred_logits = torch.cat(pred_logits, dim=0)
    else:
        pred_logits = seg_model(estimated_image_for_nnunet.float())
    
    return pred_logits


