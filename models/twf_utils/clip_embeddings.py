import torch
import clip

class ClipEmbeddings():
    def __init__(self, config, device, class_names_to_idx, embedding_dim) -> None:
        self.config = config
        self.device = device
        self.class_names_to_idx = class_names_to_idx
        self.embedding_dim = embedding_dim

        clip_model, preprocess = clip.load('ViT-B/32', self.device)
        class_names = [x.replace('_', ' ') for x in class_names_to_idx.keys()]
        text_inputs = torch.cat([clip.tokenize(f"something that looks like {c}") for c in class_names]).to(self.device)
        with torch.no_grad():
            self.text_features = clip_model.encode_text(text_inputs).float()
    
    def get_embeddings(self, class_idx):
        return self.text_features[class_idx]
    
    def get_embeddings_extended(self, class_idx):
        x = torch.full((class_idx.shape[0], self.embedding_dim), -torch.inf, device=self.device, dtype=torch.float32)
        x[:, :self.text_features.shape[1]] = self.text_features[class_idx]
        return x

