import torch
import torch.nn as nn
import timm

class IrisVeinViT(nn.Module):
    """Vision Transformer for Iris-Vein Multimodal Biometrics"""
    
    def __init__(self, num_classes=512, pretrained=True):
        super(IrisVeinViT, self).__init__()
        
        # Load pretrained Vision Transformer
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        
        # Feature dimensions
        vit_features = 768
        
        # Multimodal fusion layers
        self.iris_projection = nn.Linear(vit_features, 256)
        self.vein_projection = nn.Linear(vit_features, 256)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, iris_images, vein_images):
        # Extract features from both modalities
        iris_features = self.vit.forward_features(iris_images)
        vein_features = self.vit.forward_features(vein_images)
        
        # Project to common space
        iris_proj = self.iris_projection(iris_features)
        vein_proj = self.vein_projection(vein_features)
        
        # Concatenate and fuse
        combined = torch.cat([iris_proj, vein_proj], dim=1)
        fused = self.fusion(combined)
        
        return fused, iris_features, vein_features

class BiometricMatcher:
    """Matching module for biometric verification"""
    
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        
    def cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        emb1_norm = emb1 / emb1.norm(dim=1, keepdim=True)
        emb2_norm = emb2 / emb2.norm(dim=1, keepdim=True)
        return torch.mm(emb1_norm, emb2_norm.t()).diag()
    
    def verify(self, query_emb, template_emb):
        """Verify if embeddings belong to same person"""
        similarity = self.cosine_similarity(query_emb, template_emb)
        match = similarity > self.threshold
        return similarity.item(), match.item()
