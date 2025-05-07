import torch
import torch.nn as nn


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super(DecoderLayer, self).__init__()
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, padding_mask, attn_mask):
        x = self.layer_norm1(x)
        attn_output, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=padding_mask
        )
        x = x + attn_output
        x = self.layer_norm2(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, max_len, d_model, nhead, num_decoder_layers):
        super(TransformerDecoder, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)]
        )

    def forward(self, x, padding_mask, attn_mask):
        x = x + self.positional_encoding[:, : x.size(1)]
        for layer in self.layers:
            x = layer(x, padding_mask, attn_mask)
        return x  # [batch_size, seq_len + num_patches, d_model]


class TransformerDecoderFlickr(torch.nn.Module):
    def __init__(self, vocab_size, max_len, d_model, nhead, num_decoder_layers):
        super(TransformerDecoderFlickr, self).__init__()
        self.img_projector = torch.nn.Linear(768, d_model)
        self.embedding = torch.load("clip_text_embedding_layer.pt", weights_only=False)
        self.decoder = TransformerDecoder(max_len, d_model, nhead, num_decoder_layers)
        self.fc_out = torch.nn.Linear(d_model, vocab_size)

    def create_causal_mask(self, x, num_patches):
        # sequence length is the sum of image patches and token ids
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        attn_mask[:, num_patches:] = 0
        return attn_mask

    def forward(self, image_embed, token_ids, token_padding_mask):
        image_embed = self.img_projector(image_embed)
        token_embed = self.embedding(token_ids)

        # prefix injection
        x = torch.cat((image_embed, token_embed), dim=1)

        # resize padding mask - do not mask image patches
        num_patches = image_embed.size(1)
        token_padding_mask = ~token_padding_mask.bool() # Flip to True for padding tokens
        patch_padding_mask = torch.zeros(
            (token_padding_mask.size(0), num_patches),
            dtype=torch.bool,
            device=token_padding_mask.device,
        ) # False for image patches - do not mask
        padding_mask = torch.cat((patch_padding_mask, token_padding_mask), dim=1)
        
        attn_mask = self.create_causal_mask(x, num_patches).to(padding_mask.device)
        x = self.decoder(x, padding_mask=padding_mask, attn_mask=attn_mask)
        x = self.fc_out(x)
        return x # [batch_size, seq_len + num_patches, vocab_size]
