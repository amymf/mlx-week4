import torch
import torch.nn as nn


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super(DecoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(d_model * 4, d_model),
        )
        self.dropout2 = torch.nn.Dropout(0.1)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, padding_mask, attn_mask):
        x = self.layer_norm1(x)
        attn_output, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=padding_mask
        )
        x = x + self.dropout1(attn_output)
        x = self.layer_norm2(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, max_len, d_model, nhead, num_decoder_layers):
        super(TransformerDecoder, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(0.1)
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)]
        )
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, padding_mask, attn_mask):
        # x: [batch_size, seq_len + num_patches, d_model]
        x = x + self.positional_encoding[:, : x.size(1)] # slice pos_enc to match the caption length
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, padding_mask, attn_mask)
        x = self.layer_norm(x)
        return x  # [batch_size, seq_len + num_patches, d_model]


class TransformerDecoderFlickr(torch.nn.Module):
    def __init__(self, vocab_size, max_len, d_model, nhead, num_decoder_layers):
        super(TransformerDecoderFlickr, self).__init__()
        self.img_projector = torch.nn.Linear(768, d_model)
        self.embedding = torch.load("clip_text_embedding_layer_with_pad.pt", weights_only=False)
        self.decoder = TransformerDecoder(max_len, d_model, nhead, num_decoder_layers)
        self.fc_out = torch.nn.Linear(d_model, vocab_size)

    def create_causal_mask(self, total_len, num_patches):
        attn_mask = torch.zeros(total_len, total_len)

        # Apply causal mask only to the caption part
        caption_len = total_len - num_patches
        causal_part = torch.triu(torch.ones(caption_len, caption_len) * float('-inf'), diagonal=1)
        
        attn_mask[num_patches:, num_patches:] = causal_part
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
        
        attn_mask = self.create_causal_mask(x.size(1), num_patches).to(padding_mask.device)
        x = self.decoder(x, padding_mask=padding_mask, attn_mask=attn_mask)
        x = self.fc_out(x)
        return x # [batch_size, seq_len + num_patches, vocab_size]
