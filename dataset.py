import torch


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        caption = item["caption"][0]
        processed = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
            # return_special_tokens_mask=True,
        )
        # return input_ids, attention_mask, pixel_values
        return {
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "pixel_values": processed["pixel_values"].squeeze(0),
            "img_id": item["img_id"],
        }


class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.image_embeddings = data["image_embeddings"]
        self.input_ids = data["input_ids"]
        self.attention_masks = data["attention_masks"]
        # self.image_ids = data["image_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "image_embeddings": self.image_embeddings[idx],
            "input_ids": self.input_ids[idx],
            "attention_masks": self.attention_masks[idx],
            # "image_ids": self.image_ids[idx],
        }
