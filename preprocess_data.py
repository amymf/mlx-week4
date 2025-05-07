import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from load_data import train_dataset as train_data
import transformers
from dataset import ClipDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "openai/clip-vit-base-patch32"
tokenizer = transformers.CLIPTokenizer.from_pretrained(path) # for debugging only
processor = transformers.CLIPProcessor.from_pretrained(path)
model = transformers.CLIPModel.from_pretrained(path)
model.eval()
model.to(device)

torch.save(model.text_model.embeddings.token_embedding, "clip_text_embedding_layer.pt")

train_dataset = ClipDataset(train_data, processor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

image_embeddings = []
token_ids = []
attention_masks = []
with torch.no_grad():
    for i, batch in enumerate(train_dataloader):
        images = batch["pixel_values"].to(device)
        captions = batch["input_ids"].to(device)
        masks = batch["attention_mask"].to(device)
        special_tokens_mask = batch["special_tokens_mask"].to(device)
        cleaned_masks = masks * (1 - special_tokens_mask)

        outputs = model.vision_model(images)

        image_embeddings.append(outputs.last_hidden_state[:, 1:, :]) # Remove CLS token
        token_ids.append(captions)
        attention_masks.append(cleaned_masks)

        if i == 0:
            sample_ids = captions[0].tolist()
            decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
            print("Decoded caption with special tokens:", decoded)
            print("Attention mask:", cleaned_masks[0])
        
        if i % 10 == 0:
            print(f"Processed batch {i}/{len(train_dataloader)}")

torch.save(
    {
        "image_embeddings": torch.cat(image_embeddings, dim=0),
        "input_ids": torch.cat(token_ids, dim=0),
        "attention_masks": torch.cat(attention_masks, dim=0),
    },
    "flickr30k_embeddings.pt",
)
print("Embeddings saved to flickr30k_embeddings.pt")