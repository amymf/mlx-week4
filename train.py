import wandb
import torch
from model import TransformerDecoderFlickr
from tqdm import tqdm
from dataset import Flickr30kDataset
from transformers import CLIPTokenizer

wandb.init(project="flickr30k-captioning")

torch.manual_seed(42)  # For reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Flickr30kDataset("flickr30k_embeddings-train-with-pad.pt")
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model = TransformerDecoderFlickr(
    vocab_size=49409,  # vocab size of CLIP tokenizer + 1 for new pad token
    max_len=126,  # 77 tokens + 49 image patches
    d_model=512,  # CLIP text embedding size
    nhead=8,
    num_decoder_layers=6,
).to(device)

tokenizer = CLIPTokenizer.from_pretrained("clip_tokenizer_with_pad")
pad_token_id = 49408
start_token_id = tokenizer.bos_token_id
end_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = pad_token_id

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding tokens

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        image_embeddings = batch["image_embeddings"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_masks = batch["attention_masks"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(image_embeddings, input_ids, attention_masks)

        # Compute loss
        num_patches = image_embeddings.size(1)
        logits = outputs[:, num_patches:, :]

        target_ids = input_ids[:, 1:]         # Remove <BOS>, e.g., what the model should predict
        logits = logits[:, :-1, :]            # Exclude last logit, since there's no target for it

        loss = criterion(
            logits.reshape(-1, logits.size(-1)), 
            target_ids.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            image_embeddings = batch["image_embeddings"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)

            outputs = model(image_embeddings, input_ids, attention_masks)
            num_patches = image_embeddings.size(1)
            logits = outputs[:, num_patches:, :]

            target_ids = input_ids[:, 1:]         # Remove <BOS>, e.g., what the model should predict
            logits = logits[:, :-1, :]            # Exclude last logit, since there's no target for it

            loss = criterion(
                logits.reshape(-1, logits.size(-1)), 
                target_ids.reshape(-1)
            )
            val_loss += loss.item()

            if idx == 0:
                for j in range(min(3, image_embeddings.size(0))):  # log a few
                    generated = [start_token_id]
                    for _ in range(76):  # max token length
                        inp = torch.tensor(generated, device=device).unsqueeze(0)
                        mask = torch.ones_like(inp, dtype=torch.bool)
                        mask[:, :] = inp != pad_token_id  # Set padding tokens to False (ignore in attention)
                        out = model(image_embeddings[j:j+1], inp, mask)
                        next_token_logits = out[:, num_patches + len(generated) - 1, :]
                        next_token = next_token_logits.argmax(dim=-1).item()
                        generated.append(next_token)
                        if next_token == end_token_id:
                            break
                    decoded = tokenizer.decode(generated, skip_special_tokens=False)
                    reference = tokenizer.decode([token for token in input_ids[j].tolist() if token != pad_token_id], skip_special_tokens=False)
                    print(f"Reference {j} caption: {reference}")
                    print(f"Generated {j} caption: {decoded}")
                    wandb.log({"reference_caption": reference, "sample_caption": decoded})

    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    wandb.log({"train_loss": train_loss, "val_loss": val_loss})
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "flickr30k_model.pth")
wandb.save("flickr30k_model.pth")