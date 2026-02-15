import os
import sys
import numpy as np
import torch
import open_clip
import faiss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torchvision import datasets, transforms
from PIL import Image

NUM_IMAGES   = 1500
BATCH_SIZE   = 64
TOP_K        = 10
CLIP_MODEL   = "ViT-B-32"
PRETRAINED   = "openai"
DATA_DIR     = "./cifar_data"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_cifar10(num_images=NUM_IMAGES):
    print(f"\n[1/5] Loading CIFAR-10 ({num_images} images)...")
    dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True,
        transform=transforms.ToTensor()
    )
    images_tensor, labels = [], []
    for idx in range(num_images):
        img, label = dataset[idx]
        images_tensor.append(img)
        labels.append(label)
    print(f"    Loaded {len(images_tensor)} images.")
    return images_tensor, labels


def load_clip_model():
    print(f"\n[2/5] Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Using device: {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    return model, preprocess, device


def tensor_to_pil(img_tensor):
    np_img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def generate_embeddings(images_tensor, model, preprocess, device):
    print(f"\n[3/5] Generating embeddings...")
    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(images_tensor), BATCH_SIZE):
            batch = images_tensor[start:start + BATCH_SIZE]
            clip_input = torch.stack(
                [preprocess(tensor_to_pil(t)) for t in batch]
            ).to(device)
            emb = model.encode_image(clip_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu().numpy())
            done = min(start + BATCH_SIZE, len(images_tensor))
            print(f"    {done}/{len(images_tensor)}", end="\r")
    print()
    embeddings_np = np.vstack(all_embeddings).astype(np.float32)
    print(f"    Shape: {embeddings_np.shape}")
    return embeddings_np


def build_faiss_index(embeddings):
    print(f"\n[4/5] Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"    Vectors stored: {index.ntotal}")
    return index


def query_from_image_file(image_path, model, preprocess, device, index, top_k=TOP_K):
    """
    Load any image from disk, embed it with CLIP, search FAISS.
    Supports JPG, PNG, BMP, WEBP — anything PIL can open.
    """
    print(f"\n[5/5] Querying with image: '{image_path}'...")

    if not os.path.exists(image_path):
        print(f"    ERROR: File not found — '{image_path}'")
        sys.exit(1)

    pil_img    = Image.open(image_path).convert("RGB")   
    clip_input = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(clip_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().astype(np.float32)

    distances, indices = index.search(emb, top_k)
    return distances[0], indices[0], pil_img


def display_results(query_pil, result_tensors, result_labels, distances):
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle(
        f"Visual RAG — Input Image vs Top {TOP_K} Similar CIFAR-10 Images",
        fontsize=14, fontweight="bold"
    )

    axes[0][0].imshow(query_pil)
    axes[0][0].set_title("INPUT\n(your image)", fontsize=10, fontweight="bold", color="navy")
    axes[0][0].axis("off")
    axes[1][0].axis("off")

    for rank, (tensor, label, dist) in enumerate(zip(result_tensors, result_labels, distances)):
        row = rank // 5
        col = (rank %  5) + 1
        axes[row][col].imshow(tensor_to_pil(tensor))
        axes[row][col].set_title(
            f"#{rank+1}  {CLASSES[label]}\nscore: {dist:.3f}",
            fontsize=8.5
        )
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig("rag_results.png", bbox_inches="tight", dpi=130)
    print("    Saved → 'rag_results.png'")
    plt.show()


def main():
    QUERY_IMAGE_PATH = "download.jpg" #change the image for different querries.

    print("=" * 50)
    print("  Visual RAG  —  CIFAR-10 + CLIP + FAISS")
    print("=" * 50)

    images_tensor, labels        = load_cifar10()
    model, preprocess, device    = load_clip_model()
    embeddings                   = generate_embeddings(images_tensor, model, preprocess, device)
    index                        = build_faiss_index(embeddings)
    distances, indices, query_pil = query_from_image_file(
        QUERY_IMAGE_PATH, model, preprocess, device, index
    )

    result_tensors = [images_tensor[i] for i in indices]
    result_labels  = [labels[i]         for i in indices]

    print(f"    Top {TOP_K} results:")
    for rank, (idx, dist, lbl) in enumerate(zip(indices, distances, result_labels)):
        print(f"      #{rank+1:>2}  idx={idx:>4}  score={dist:.4f}  class={CLASSES[lbl]}")

    display_results(query_pil, result_tensors, result_labels, distances)
    print("\n Done!\n")


if __name__ == "__main__":
    main()