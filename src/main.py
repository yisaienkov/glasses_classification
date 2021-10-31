import glob

import cv2
import torch

from modules.models import get_model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device)

    for path in sorted(glob.glob("images/*"), key=lambda x: int(x.split("/")[-1].split(".")[0])):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h, _ = image.shape
        if w > h:
            image = image[(w - h) // 2 : (w - h) // 2 + h, :, :]
        else:
            image = image[:, (h - w) // 2 : (h - w) // 2 + w, :]

        predict, probability = model(image)
        if predict:
            print(f"Found glasses ({probability:.3f}): {path}")