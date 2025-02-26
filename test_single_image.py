from tmfnet import TMFNet
from PIL import Image
import numpy as np
import torch
import os
from download_test_images import download_test_images

def main():
    download_test_images()

    image_path = "image.png"
    trimap_path = "trimap.png"
    output_alpha_path = "alpha.png"
    gt_alpha_path = "ground_truth_alpha.png"
    checkpoint_path = "comp1k.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Complain if model has not been downloaded yet
    if not os.path.exists(checkpoint_path):
        url = "https://github.com/Serge-weihao/TMF-Matting?tab=readme-ov-file#results-and-models"
        raise FileNotFoundError(f"Download checkpoint {checkpoint_path} from {url} and place it in this directory")

    # Load model
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
    model = TMFNet()
    model.load_state_dict(state_dict)
    model.to(device)

    # Load and preprocess images
    image = np.array(Image.open(image_path).convert("RGB")) / 255.0
    trimap = np.array(Image.open(trimap_path).convert("L")) / 255.0
    gt_alpha = np.array(Image.open(gt_alpha_path).convert("L")) / 255.0

    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(device)
    trimap = torch.from_numpy(trimap.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # Predict alpha using TMF-Net
        outputs = model(image, trimap).clip(0, 1)

        # Fill in known foreground and background from trimap
        outputs[trimap == 1] = 1
        outputs[trimap == 0] = 0

        alpha = outputs[0, 0].detach().cpu().numpy()

    # Save alpha
    Image.fromarray((alpha * 255).astype(np.uint8)).save(output_alpha_path)

    # Check if error to ground truth alpha matte is small
    mse = np.mean(np.square(alpha - gt_alpha))
    assert mse < 0.0031642, f"Mean squared error {mse} is larger than expected, something went wrong"

if __name__ == "__main__":
    main()
