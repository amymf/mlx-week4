import wandb
from dataset import Flickr30kDataset
import torch
from model import TransformerDecoderFlickr
from transformers import CLIPTokenizer

wandb.init(project="flickr30k-captioning")

torch.manual_seed(42)  # For reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = Flickr30kDataset("flickr30k_embeddings-test.pt")
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

start_token_id = 49406
end_token_id = 49407

model = TransformerDecoderFlickr(
    vocab_size=49409,  # vocab size of CLIP tokenizer + 1 for new pad token
    max_len=126,  # 77 tokens + 49 image patches
    d_model=512,  # CLIP text embedding size
    nhead=8,
    num_decoder_layers=6,
).to(device)
model.load_state_dict(torch.load("flickr30k_model.pth"))
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

seq_len = 77
start_sentence = torch.tensor([start_token_id] + [end_token_id] * (seq_len - 1), dtype=torch.long,).to(device)

for batch in test_dataloader:
    image_embeddings = batch["image_embeddings"].to(device)
    print("Image embeddings shape:", image_embeddings.shape)

    batch_size = image_embeddings.size(0)
    start_sentences = start_sentence.unsqueeze(0).repeat(batch_size, 1)  # shape [batch_size, seq_len]
    print("Start sentences shape:", start_sentences.shape)

    token_padding_mask = (start_sentences != end_token_id)  # shape: [batch_size, seq_len] - False for padding tokens
    num_patches = image_embeddings.size(1)

    predictions = []
    for i in range(seq_len - 1):
        outputs = model(image_embeddings, start_sentences, token_padding_mask)
        logits = outputs[:, num_patches:, :]  # shape: [batch_size, seq_len, vocab_size]
        logits = logits[:, i, :]  # shape: [batch_size, vocab_size]
        _, predicted = torch.max(logits, dim=-1)  # shape: [batch_size]
        predictions.append(predicted)

        # Update start_sentences with the predicted token
        start_sentences[:, i + 1] = predicted
        
    # Convert predictions to text
    predictions = torch.stack(predictions, dim=1)  # shape: [batch_size, seq_len]
    predictions = predictions.cpu().numpy()
    for i in range(predictions.shape[0]):
        caption = " ".join([str(token) for token in tokenizer.decode(predictions[i], skip_special_tokens=True)])
        print(f"Generated caption: {caption}")
        wandb.log({"generated_caption": caption})