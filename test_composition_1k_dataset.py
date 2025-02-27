import tmfnet, torch, io, math, zipfile, numpy as np
from PIL import Image

adobe_zip = zipfile.ZipFile("Adobe_Deep_Matting_Dataset.zip")
pascal_background_dir = "PascalVOC2012"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = tmfnet.load_model()
model.to(device)

# Load lists of foreground and background image names
fg_names = adobe_zip.read("Combined_Dataset/Test_set/test_fg_names.txt").decode("utf-8").strip().replace("\r", "").split("\n")
bg_names = adobe_zip.read("Combined_Dataset/Test_set/test_bg_names.txt").decode("utf-8").strip().replace("\r", "").split("\n")

bgs_per_fg = len(bg_names) // len(fg_names)

mses = []
sads = []

image_cache = {}

def load_from_zip(path, z, mode):
    if path not in image_cache:
        image_cache[path] = Image.open(io.BytesIO(z.read(path))).convert(mode)
    return image_cache[path]

print("# Composition-1K Test Results\n")
print("| /1000 | MSE ×1K | SAD /1K | AVG MSE | AVG SAD | bg_name | fg_name |")
print("| ----- | ------- | ------- | ------- | ------- | ------- | ------- |")

for i in range(len(bg_names)):
    # One foreground image is composited onto 20 different background images
    fg_name = fg_names[i // bgs_per_fg]
    bg_name = bg_names[i]
    i_trimap = i % bgs_per_fg
    trimap_name = fg_name.replace(".png", f"_{i_trimap}.png")
    trimap_path = f"Combined_Dataset/Test_set/Adobe-licensed images/trimaps/{trimap_name}"
    alpha_path = f"Combined_Dataset/Test_set/Adobe-licensed images/alpha/{fg_name}"
    fg_path = f"Combined_Dataset/Test_set/Adobe-licensed images/fg/{fg_name}"
    bg_path = f"{pascal_background_dir}/{bg_name}"

    trimap = load_from_zip(trimap_path, adobe_zip, "L")
    gt_alpha = load_from_zip(alpha_path, adobe_zip, "L")
    fg = load_from_zip(fg_path, adobe_zip, "RGB")
    bg = Image.open(bg_path).convert("RGB")

    # Resize background to cover foreground if background is smaller
    w, h = fg.size
    bw, bh = bg.size
    scale = max(w / bw, h / bh)
    if scale > 1:
        bw = math.ceil(scale * bw)
        bh = math.ceil(scale * bh)
        bg = bg.resize((bw, bh), Image.BICUBIC)

    # [0, 255] -> [0, 1]
    trimap = np.array(trimap) / 255.0
    gt_alpha = np.array(gt_alpha) / 255.0
    fg = np.array(fg) / 255.0
    bg = np.array(bg) / 255.0

    # Unknown region is neither foreground nor background
    is_fg = trimap == 1
    is_bg = trimap == 0
    is_unknown = 1 - is_fg - is_bg

    # Blend foreground and background to create image
    a = gt_alpha[:, :, np.newaxis]
    image = a * fg + (1 - a) * bg[:h, :w]

    # Normalize (standardize) image by subtracting mean and dividing by standard deviation
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    with torch.no_grad():
        # Convert Numpy (h, w, c) to PyTorch (n, c, h, w) on device
        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(device)
        trimap = torch.from_numpy(trimap.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        alpha = model(image, trimap).clip(0, 1)[0, 0].detach().cpu().numpy()

        # Fill in known foreground and background from trimap
        alpha[is_fg] = 1
        alpha[is_bg] = 0

        # Compute and print metrics
        sad = np.sum(np.abs(alpha - gt_alpha) * is_unknown) / 1000.0
        mse = np.sum(np.square(alpha - gt_alpha) * is_unknown) / np.sum(is_unknown) * 1000

        sads.append(sad)
        mses.append(mse)

        print(f"| {i + 1:5d} | {mse:7.3f} | {sad:7.3f} | {np.mean(mses):7.3f} | {np.mean(sads):7.3f} | {bg_name} | {fg_name} |")
