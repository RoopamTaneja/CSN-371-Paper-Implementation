import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        batch_size, num_tokens, _, num_heads = *x.shape, self.num_heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=num_heads)

        attention_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == attention_scores.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            attention_scores.masked_fill_(~mask, float("-inf"))
            del mask

        attention_weights = attention_scores.softmax(dim=-1)
        attention_output = torch.einsum("bhij,bhjd->bhid", attention_weights, v)
        attention_output = rearrange(attention_output, "b h n d -> b n (h d)")
        output = self.to_out(attention_output)
        return output


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Residual(PreNorm(dim, Attention(dim, num_heads=num_heads))), Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))]) for _ in range(depth)])

    def forward(self, x, mask=None):
        for attention_layer, feedforward_layer in self.layers:
            x = attention_layer(x, mask=mask)
            x = feedforward_layer(x)
        return x


class CViT(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 7, num_classes: int = 2, cnn_channels: int = 512, transformer_dim: int = 1024, transformer_depth: int = 6, transformer_heads: int = 8, transformer_mlp_dim: int = 2048):
        super().__init__()
        assert image_size % patch_size == 0, "image dimensions must be divisible by the patch size"

        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.patch_size = patch_size
        patch_dim = cnn_channels * patch_size**2
        num_patches = (image_size // patch_size) ** 2

        # Vision Transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, transformer_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, transformer_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.transformer = Transformer(transformer_dim, transformer_depth, transformer_heads, transformer_mlp_dim)
        self.to_cls_token = nn.Identity()

        # Classification head
        self.mlp_head = nn.Sequential(nn.Linear(transformer_dim, transformer_mlp_dim), nn.ReLU(), nn.Linear(transformer_mlp_dim, num_classes))  # was self.classifier

    def forward(self, images: torch.Tensor, mask=None) -> torch.Tensor:
        patch_size = self.patch_size
        features = self.features(images)
        patches = rearrange(features, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
        patch_embeddings = self.patch_to_embedding(patches)
        batch_size = features.shape[0]
        class_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((class_tokens, patch_embeddings), 1)
        tokens += self.pos_embedding[:, : tokens.size(1)]
        transformer_output = self.transformer(tokens, mask)
        cls_output = self.to_cls_token(transformer_output[:, 0])
        return self.mlp_head(cls_output)
