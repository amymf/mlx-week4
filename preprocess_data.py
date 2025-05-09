import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from load_data import train_dataset, test_dataset
import transformers
from dataset import ClipDataset

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # gpu memory limits

path = "openai/clip-vit-base-patch32"
tokenizer = transformers.CLIPTokenizer.from_pretrained(path)  # for debugging only
processor = transformers.CLIPProcessor.from_pretrained(path)
model = transformers.CLIPModel.from_pretrained(path)
model.eval()
model.to(device)

pad_token = "<pad>"
tokenizer.add_tokens(pad_token)
new_pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
tokenizer.pad_token_id = new_pad_token_id
tokenizer.add_special_tokens({'pad_token': pad_token})
tokenizer.save_pretrained("clip_tokenizer_with_pad")

# resize embeddings to accommodate new pad token
model.text_model.embeddings.token_embedding = nn.Embedding.from_pretrained(
    torch.cat(
        [
            model.text_model.embeddings.token_embedding.weight,
            torch.randn(1, model.text_model.embeddings.token_embedding.weight.size(1)),
        ],
        dim=0,
    )
)

torch.save(
    model.text_model.embeddings.token_embedding, "clip_text_embedding_layer_with_pad.pt"
)


def preprocess_data(split):
    data = train_dataset if split == "train" else test_dataset
    dataset = ClipDataset(data, processor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    image_embeddings = []
    token_ids = []
    attention_masks = []
    image_ids = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch["pixel_values"].to(device)
            captions = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            image_ids = batch["img_id"]

            # Replace tokens after the first <eos> token with the new <pad> token
            eos_token_id = tokenizer.eos_token_id
            for j in range(captions.size(0)):
                eos_idx = (captions[j] == eos_token_id).nonzero(as_tuple=True)
                if eos_idx[0].numel() > 0:  # If there's an <eos> token in the caption
                    first_eos_idx = eos_idx[0].min()
                    masks[j, first_eos_idx] = 1  # Keep the <eos> token in the mask
                    captions[j, first_eos_idx + 1 :] = (
                        new_pad_token_id  # Replace after <eos> with <pad>
                    )
                    masks[j, first_eos_idx + 1 :] = 0  # Mask these tokens as padding

            outputs = model.vision_model(images)

            image_embeddings.append(
                outputs.last_hidden_state[:, 1:, :]
            )  # Remove CLS token
            token_ids.append(captions)
            attention_masks.append(masks)

            if i == 0:
                sample_ids = captions[0].tolist()
                decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
                print("Decoded caption with special tokens:", decoded)
                print("Attention mask:", masks[0])
                print("Image ID:", image_ids[0])

            if i % 10 == 0:
                print(f"Processed batch {i}/{len(dataloader)}")

    torch.save(
        {
            "image_embeddings": torch.cat(image_embeddings, dim=0),
            "input_ids": torch.cat(token_ids, dim=0),
            "attention_masks": torch.cat(attention_masks, dim=0),
            "image_ids": image_ids
        },
        f"flickr30k_embeddings-{split}-with-pad.pt",
    )
    print(f"Embeddings saved to flickr30k_embeddings-{split}-with-pad.pt")


if __name__ == "__main__":
    preprocess_data("train")
    preprocess_data("test")
