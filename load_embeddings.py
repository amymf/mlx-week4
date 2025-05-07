import huggingface_hub
import torch

file_path = huggingface_hub.hf_hub_download(
    repo_id="amyf/flickr30k",
    filename="flickr30k_embeddings-test-test-with-pad.pt",
    repo_type="dataset",
    revision="main"
)
test_data = torch.load(file_path)
torch.save(test_data, "flickr30k_embeddings-test-with-pad.pt")

file_path = huggingface_hub.hf_hub_download(
    repo_id="amyf/flickr30k",
    filename="flickr30k_embeddings-train-with-pad.pt",
    repo_type="dataset",
    revision="main"
)
test_data = torch.load(file_path)
torch.save(test_data, "flickr30k_embeddings-train-with-pad.pt")

file_path = huggingface_hub.hf_hub_download(
    repo_id="amyf/flickr30k",
    filename="clip_text_embedding_layer_with_pad.pt",
    repo_type="dataset",
    revision="main"
)
test_data = torch.load(file_path, weights_only=False)
torch.save(test_data, "clip_text_embedding_layer_with_pad.pt")

