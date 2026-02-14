import cv2
import numpy as np
import torch
from PIL import Image
from src.config import *
from src.model import get_torch_device, get_transforms, load_checkpoint

def generate_gradcam(img_path):
    device = get_torch_device()
    model, model_type = load_checkpoint(os.path.join(MODEL_DIR, MODEL_FILENAME), map_location=device)
    model = model.to(device)
    model.eval()

    if model_type != "resnet":
        raise ValueError("Grad-CAM currently supports only resnet checkpoints.")

    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE)))
    transform = get_transforms(model_type, train=False)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def save_gradient(_module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(_module, _input, output):
        activations.append(output)

    target_layer = model.layer4[-1].conv3
    handle_fwd = target_layer.register_forward_hook(save_activation)
    handle_bwd = target_layer.register_full_backward_hook(save_gradient)

    output = model(input_tensor)
    class_idx = int(torch.argmax(output, dim=1).item())
    score = output[:, class_idx]

    model.zero_grad()
    score.backward()

    grads = gradients[0]
    acts = activations[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze(0)
    cam = torch.relu(cam)
    cam = cam / (torch.max(cam) + 1e-8)
    cam = cam.detach().cpu().numpy()

    handle_fwd.remove()
    handle_bwd.remove()

    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(heatmap, 0.4, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), 0.6, 0)
    return superimposed
