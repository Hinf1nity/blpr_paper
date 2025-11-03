import torch
import open_clip
from PIL import Image
from load_data import CHARS, CHARS_DICT, LPRDataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.load_state_dict(torch.load(
    "clip_finetuned_lpr.pth", map_location=device))
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
text_tokens = tokenizer(CHARS).to(device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

    logits_per_image, logits_per_text = model(image, text_tokens)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
