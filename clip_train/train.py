import torch
from open_clip import create_model_and_transforms, get_tokenizer
import clip
from load_data import LPRCLIPDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
tokenizer_fn = get_tokenizer('ViT-B-32')

dataset = LPRCLIPDataset(
    img_dirs=["/home/hinfinity/Documents/datasets/dataset_lpr/train"],
    imgSize=(224, 224), transform_clip=preprocess
)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(10):
    for imgs, textos in loader:
        imgs = imgs.to(device)
        textos_tok = tokenizer_fn(textos).to(device)

        image_features = model.encode_image(imgs)
        text_features = model.encode_text(textos_tok)

        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        labels = torch.arange(len(imgs), dtype=torch.long, device=device)

        loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
torch.save(model.state_dict(), "clip_finetuned_lpr.pth")
