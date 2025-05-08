import wandb
from dataset import Flickr30kDataset
import torch
from model import TransformerDecoderFlickr
from transformers import CLIPTokenizer

wandb.init(project="flickr30k-captioning")

torch.manual_seed(42)  # For reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = Flickr30kDataset("flickr30k_embeddings-test-with-pad.pt")
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

model = TransformerDecoderFlickr(
    vocab_size=49409,  # vocab size of CLIP tokenizer + 1 for new pad token
    max_len=126,  # 77 tokens + 49 image patches
    d_model=512,  # CLIP text embedding size
    nhead=8,
    num_decoder_layers=6,
).to(device)
model.load_state_dict(torch.load("flickr30k_model.pth"))
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("clip_tokenizer_with_pad")
pad_token_id = 49408
start_token_id = tokenizer.bos_token_id
end_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = pad_token_id

criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding tokens

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
        target_ids = input_ids[:, 1:]         # Remove <BOS>, e.g., what the model should predict
        logits = logits[:, :-1, :]            # Exclude last logit, since there's no target for it

        loss = criterion(
            logits.reshape(-1, logits.size(-1)), 
            target_ids.reshape(-1)
        )
        
        # Compute accuracy
        pad_mask = target_ids != pad_token_id
        _, predicted = torch.max(logits, dim=-1)
        # print(f"example predicted: {predicted[0]}")
        # print(f"example target: {target_ids[0]}")
        correct += ((predicted == target_ids) & pad_mask).sum().item()
        total += pad_mask.sum().item()

        if i == 0:
            for j in range(min(3, image_embeddings.size(0))):  # log a few
                generated = [start_token_id]
                for _ in range(76):  # max token length
                    inp = torch.tensor(generated, device=device).unsqueeze(0)
                    mask = torch.ones_like(inp, dtype=torch.bool)
                    out = model(image_embeddings[j:j+1], inp, mask)
                    next_token_logits = out[:, num_patches + len(generated) - 1, :]
                    next_token = next_token_logits.argmax(dim=-1).item()
                    if next_token == end_token_id:
                        break
                    generated.append(next_token)

                decoded = tokenizer.decode(generated[1:], skip_special_tokens=True)
                reference = tokenizer.decode([token for token in target_ids[j].tolist() if token != pad_token_id], skip_special_tokens=True)
                print(f"Reference {j} caption: {reference}")
                print(f"Sample {j} caption: {decoded}")
                wandb.log({"reference_caption": reference, "sample_caption": decoded})

accuracy = 100 * correct / total
print(f"Test Loss: {loss.item()}")
print(f"Test Accuracy: {accuracy:.2f}%")
wandb.log({"test_loss": loss.item(), "test_accuracy": accuracy})