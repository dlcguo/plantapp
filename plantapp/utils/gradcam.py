import torch, cv2
import numpy as np

# This file is part generating a Grad-CAM diagram for the plantapp project

def grad_cam(model, img, head, class_idx=None, layer_name="backbone.layer4"):
    """
    Compute Grad-CAM for a given model and image
    
    Args:
        model: The model to compute Grad-CAM for
        img: The input image (tensor)
        head: The head to compute Grad-CAM for ("plant" or "disease")
        class_idx: The class index to compute Grad-CAM for (optional)
        layer_name: The name of the layer to compute Grad-CAM from

    Returns:
        cam: The computed Grad-CAM
    """
    acts, grads = [], []
    layer = dict([*model.named_modules()])[layer_name]
    fh = layer.register_forward_hook(lambda _, _2, i: acts.append(i))
    bh = layer.register_backward_hook(lambda _, _2, i: grads.append(i[0]))

    logits = model(img)[0 if head=="plant" else 1]
    if class_idx is None: 
        class_idx = logits.argmax(1).item()
    score = logits[0, class_idx]
    model.zero_grad()
    score.backward()

    A = acts[0][0]
    G = grads[0][0]
    weights = G.mean(dim=(1,2))
    cam = (weights[:,None,None] * A).sum(0).relu()
    cam = cam / cam.max().clamp(min=1e-5)
    H,W = img.shape[-2:]
    cam = cv2.resize(cam.detach().cpu().numpy(), (W, H))
    fh.remove() 
    bh.remove()
    return cam