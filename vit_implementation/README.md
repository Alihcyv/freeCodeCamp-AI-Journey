# Vision Transformer (ViT) from Scratch

This project is a complete implementation of the Vision Transformer (ViT) architecture. The main idea of ViT is to treat an image as a sequence of patches, allowing us to apply the Transformer architecture (originally designed for Natural Language Processing) to Computer Vision tasks.

## Architecture Breakdown

### 1. Patch Embedding
Transformers were originally designed for NLP, where words are converted into vector embeddings. To apply this to images, we need to convert a 2D image into a 1D sequence of vectors.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, image_size, channels, embed_dim) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x) 
        x = x.flatten(2).transpose(1, 2) 
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1) 
        x = x + self.pos_embed
        return x
```

1. **`self.proj` (Convolutional Projection):**
    - We create filters of size `patch_size` $\times$ `patch_size` for the input `channels`.
    - The number of filters equals `out_channels` (`embed_dim`).
    - By setting `stride=patch_size`, we ensure that the patches do not overlap, effectively slicing the image.
    - **Output Shape:** `[B, embed_dim, image_size/patch_size, image_size/patch_size]`

2. **`flatten(2).transpose(1, 2)` (Reshaping):**
    - `flatten(2)` merges the spatial dimensions (height and width of the patch grid) into one.
    - `transpose(1, 2)` swaps the dimensions to move the embedding vector to the end.
    - **Output Shape:** `[B, num_patches, embed_dim]`

3. **`self.cls_token.expand(B, -1, -1)` (Batch Alignment):**
    - The `cls_token` is initialized as `[1, 1, embed_dim]`. To match the batch size `B`, we use `.expand()`.
    - The value `-1` tells PyTorch to keep the dimension as it is.
    - **Output Shape:** `[B, 1, embed_dim]`

4. **`torch.cat((cls_token, x), dim=1)` (The Aggregator):**
    - We concatenate the `cls_token` to the start of the patch sequence.
    - This additional patch acts as a "global representative" that collects information from all other patches through the attention mechanism.
    - At the end of the model, only this token's output is used for the loss function and classification.
    - **Output Shape:** `[B, 1 + num_patches, embed_dim]`

5. **`x + self.pos_embed` (Positional Embedding):**
    - We add positional weights because spatial information is critical for the model to understand the structure of an image.
    - These embeddings also help the model differentiate the global classification token from the actual image patches.
    - **Observation:** Based on my experiments, disabling the positional embeddings led to significantly lower performance, which was especially noticeable when training on more complex images.
    - **Output Shape:** `[B, 1 + num_patches, embed_dim]`



### 2. MLP (Multi-Layer Perceptron / Feed-Forward Network)
In Transformer architectures, this block is often referred to as the **FFN (Feed-Forward Network)**. While the Attention mechanism allows tokens to interact with each other, the MLP allows each token to process its own features independently.

- **Purpose:** It acts as a local feature processor, mapping the attention-weighted representations into a higher-dimensional space to extract more complex patterns and then projecting them back.
- **GELU Activation:** Instead of the standard ReLU, I used the **GELU (Gaussian Error Linear Unit)** activation function. 
    - **Why GELU?** Unlike ReLU, which abruptly zeros out all negative values (creating "dead neurons"), GELU provides a smooth, stochastic transition. It allows a small amount of negative information to pass through, which helps the model learn more complex functions and generally leads to better convergence in Transformers.

<p align="center">
  <img src="images/gelu_activation.png" width="500"/>
</p>

- **Regularization:** I implemented **Dropout** after both linear transformations to prevent the model from overfitting on the training data.
