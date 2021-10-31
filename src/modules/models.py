import torch
from torch import nn
from torch.utils import model_zoo
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

URL = "https://github.com/yisaienkov/glasses_classification/releases/download/v0.0.1/best_checkpoint.pt"


def get_valid_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(p=1.0, max_size=image_size),
            A.Normalize(p=1.0),
            A.PadIfNeeded(
                p=1.0,
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()   
        self.features = nn.Sequential(
            self.make_block(3, 8, kernel_size=3),
            self.make_block(8, 16, kernel_size=3),
            self.make_block(16, 32, kernel_size=3),
            self.make_block(32, 64, kernel_size=3),
            self.make_block(64, 128, kernel_size=3),
        )       
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def make_block(self, input_channels, output_channels, kernel_size=3, stride=1, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class InferenceModel:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model = Model()
        self.model.to(self.device)

        self.transforms = get_valid_transforms(128)

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict["model_state_dict"])

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, image):
        self.eval()
        transformed_image = self.transforms(image=image)["image"].unsqueeze(dim=0)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(transformed_image.to(self.device)).cpu()).numpy()[0][0]

        return prob >= 0.5, prob


def get_model(device: str = "cpu"):
    model = InferenceModel(device=device)
    # state_dict = model_zoo.load_url(URL, progress=True, map_location=device)
    state_dict = torch.load("resources/best_checkpoint.pt", map_location=device)
    model.load_state_dict(state_dict)

    return model