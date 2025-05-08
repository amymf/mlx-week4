import wandb
from dataset import Flickr30kDataset
import torch
from model import TransformerDecoderFlickr
from transformers import CLIPTokenizer
import torch.nn as nn

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
model.load_state_dict(torch.load("flickr30k_model.pth", map_location=device))
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("clip_tokenizer_with_pad")
pad_token_id = 49408
start_token_id = tokenizer.bos_token_id
end_token_id = tokenizer.eos_token_id

seq_len = 77

for batch in test_dataloader:
    image_embeddings = batch["image_embeddings"].to(device)
    batch_size = image_embeddings.size(0)
    start_sentences = torch.full(
        (batch_size, 1), start_token_id, dtype=torch.long, device=device
    )

    # Create attention mask - True means token is not masked
    token_padding_mask = torch.ones_like(
        start_sentences, dtype=torch.bool
    )  # True for non-padding
    num_patches = image_embeddings.size(1)

    # Track which sequences are still generating
    still_generating = torch.ones(batch_size, dtype=torch.bool, device=device)

    predictions = []
    for i in range(seq_len - 1):
        current_seq_len = start_sentences.size(1)
        total_seq_len = num_patches + current_seq_len

        outputs = model(image_embeddings, start_sentences, token_padding_mask)
        logits = outputs[
            :, num_patches + i : num_patches + i + 1, :
        ]  # shape: [batch_size, seq_len, vocab_size]
        logits = logits.squeeze(
            1
        )  # [batch_size, vocab_size] - get the logits for the current token
        probs = torch.softmax(logits, dim=-1)  # shape: [batch_size, vocab_size]
        predicted = torch.multinomial(probs, num_samples=1).squeeze()
        # _, predicted = torch.max(logits, dim=-1)  # shape: [batch_size]
        predictions.append(predicted)

        still_generating = still_generating & (predicted.squeeze(-1) != end_token_id)
        # If a sequence has generated an <eos> token, stop generating for that sequence
        next_token = torch.where(
            still_generating.unsqueeze(-1),
            predicted.unsqueeze(-1),
            torch.full_like(predicted.unsqueeze(-1), pad_token_id),
        )

        # Update start_sentences with the predicted token
        start_sentences = torch.cat((start_sentences, predicted.unsqueeze(1)), dim=1)

        # Update token_padding_mask
        token_padding_mask = torch.cat(
            [
                token_padding_mask,
                torch.ones(
                    (batch_size, 1), dtype=torch.bool, device=device
                ),  # real token
            ],
            dim=1,
        )

    # Convert predictions to text
    predictions = torch.stack(predictions, dim=1)  # shape: [batch_size, seq_len]
    predictions = predictions.cpu().numpy()
    for i in range(predictions.shape[0]):
        caption = tokenizer.decode(predictions[i], skip_special_tokens=True)
        print(f"Generated caption: {caption}")
        wandb.log({"generated_caption": caption})
