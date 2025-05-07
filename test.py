import wandb
from dataset import Flickr30kDataset
import torch
from model import TransformerDecoderFlickr

wandb.init(project="flickr30k-captioning")

torch.manual_seed(42)  # For reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = Flickr30kDataset("flickr30k_embeddings-test.pt")
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

model = TransformerDecoderFlickr(
    vocab_size=49408,  # vocab size of CLIP tokenizer
    max_len=126,  # 77 tokens + 49 image patches
    d_model=512,  # CLIP text embedding size
    nhead=8,
    num_decoder_layers=6,
).to(device)
model.load_state_dict(torch.load("flickr30k_model.pth"))
model.eval()

criterion = torch.nn.CrossEntropyLoss(ignore_index=1)  # Ignore padding tokens

correct = 0
total = 0
with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        image_embeddings = batch["image_embeddings"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_masks = batch["attention_masks"].to(device)

        # Forward pass
        outputs = model(image_embeddings, input_ids, attention_masks)

        # Compute loss
        num_patches = image_embeddings.size(1)
        logits = outputs[:, num_patches:, :]
        loss = criterion(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1))

        # Compute accuracy
        _, predicted = torch.max(logits, dim=-1)
        correct += (predicted == input_ids).sum().item()
        total += input_ids.numel()

accuracy = 100 * correct / total
print(f"Test Loss: {loss.item()}")
print(f"Test Accuracy: {accuracy:.2f}%")
wandb.log({"test_loss": loss.item(), "test_accuracy": accuracy})