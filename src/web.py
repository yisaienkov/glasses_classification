from albumentations.augmentations import transforms
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
import streamlit as st
import matplotlib.pyplot as plt

from modules.models import Model, get_valid_transforms

def get_images():
    uploaded_files = st.file_uploader("Choose a photo", accept_multiple_files=True)
    images = []
    for uploaded_file in uploaded_files:
        _stream = uploaded_file.getvalue()
        data = np.fromstring(_stream, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


@st.cache
def load_model(device):
    model = Model()
    state_dict = torch.load("resources/best_checkpoint.pt", map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def denormalize_image(
    image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
):
    return (max_pixel_value * (image * std + mean)).astype(int)


if __name__ == "__main__":
    st.title("Glasses Classification")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    transform = get_valid_transforms(128)
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())

    images = get_images()
    for image in images:
        w, h, _ = image.shape
        if w > h:
            image = image[(w - h) // 2 : (w - h) // 2 + h, :, :]
        else:
            image = image[:, (h - w) // 2 : (h - w) // 2 + w, :]
        transformed_image = transform(image=image)["image"].unsqueeze(dim=0)
        default_image = denormalize_image(transformed_image[0].numpy().transpose(1, 2, 0))

        with torch.no_grad():
            prob = torch.sigmoid(model(transformed_image.to(device)).cpu()).numpy()[0][0]
        
        container = st.container().columns(2)

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(default_image)
        plt.axis("off")
        container[0].header(f"Your image")
        container[0].pyplot(fig)
        
        grayscale_cam = cam(input_tensor=transformed_image)[0, :]

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(default_image)
        plt.imshow(grayscale_cam, cmap="jet", alpha=0.5)
        plt.axis("off")
        container[1].header(f"Probability: {prob * 100:.0f} %")
        container[1].pyplot(fig)