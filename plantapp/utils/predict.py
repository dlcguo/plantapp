import json
import torch
import torchvision.transforms as T
from pathlib import Path

from plantapp.config import cfg
from plantapp.models.resnet_twohead import ResNetTwoHead
from plantapp.utils.gradcam import grad_cam

# This file contains the prediction components for the plantapp project

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

def load_model():
    """
    Load the model from the checkpoint path specified in the config
    
    Returns:
        model: The loaded ResNetTwoHead model
    """
    ckpt_path = cfg.checkpoint_path
    states = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state" in states:
        state_dict = states["model_state"]
    elif isinstance(states, dict) and next(iter(states)).startswith("backbone."):
        state_dict = states
    else:
        raise RuntimeError("Unexpected checkpoint format")

    model = ResNetTwoHead(
        n_plants   = len(state_dict["plant_head.weight"]),
        n_diseases = len(state_dict["disease_head.weight"]),
        pretrained = False,
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()

preproc = T.Compose([
            T.Resize(int(cfg.img_size*1.14)),
            T.CenterCrop(cfg.img_size),
            T.ToTensor(),
            T.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
          ])

def predict(model, img):
    """
    Predict the plant and disease indices, Grad-CAM, and probabilities for a given image
    
    Args:
        model: The ResNetTwoHead model
        img: The input image (PIL Image)
        
    Returns:
        plant_idx: The predicted plant index
        dis_idx: The predicted disease index
        cam: The Grad-CAM for the predicted disease
        plant_prob: The probability of the predicted plant
        disease_prob: The probability of the predicted disease
    """
    x = preproc(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        plant_logits, disease_logits = model(x)

    plant_prob = plant_logits.softmax(1).max().item()
    disease_prob   = disease_logits.softmax(1).max().item()

    plant_idx = plant_logits.argmax(1).item()
    dis_idx = disease_logits.argmax(1).item()
    cam = grad_cam(model, x, head="disease", class_idx=dis_idx,
                   layer_name=cfg.gradcam_layer)

    return plant_idx, dis_idx, cam, plant_prob, disease_prob