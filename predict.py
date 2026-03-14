from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import sys

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 3)
model.load_state_dict(torch.load("checkpoints/best_efficientnet_b0.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

img = Image.open(sys.argv[1]).convert("RGB")
x = tfms(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
    pred = torch.argmax(probs).item()

print({"predicted_class": CLASS_NAMES[pred], "confidence": float(probs[pred])})