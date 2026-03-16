import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

def visualize_graph(data, output_dir="img", num_nodes=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"📊 Visualizing a subgraph of {num_nodes} nodes.")
    
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    
    subset_nodes = range(num_nodes)
    G.add_nodes_from(subset_nodes)
    
    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edges = edge_index[:, mask].T
    G.add_edges_from(edges)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    if hasattr(data, 'y'):
        colors = data.y[:num_nodes].cpu().numpy()
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors, cmap=plt.cm.coolwarm)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
        
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    plt.title(f"Graph Visualization (first {num_nodes} nodes)")
    plt.axis('off')
    
    filepath = os.path.join(output_dir, "graph_visualization.png")
    plt.savefig(filepath)
    plt.close()
    print(f"✅ Visualization saved to {filepath}")


# NEW CODE =====================================================
# Added t-SNE visualization for final hidden embeddings.
# This expects the model to support:
#   model(data.x, data.edge_index, return_embeddings=True)
# and returns:
#   logits, embeddings
def visualize_tsne_embeddings(model, data, mask, output_dir="img", file_name="tsne_embeddings.png"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    with torch.no_grad():
        # OLD CODE COMMENTED OUT BELOW:
        # out = model(data.x, data.edge_index)

        logits, embeddings = model(data.x, data.edge_index, return_embeddings=True)

    # Keep only nodes in the provided split and exclude unknown labels.
    valid_mask = mask & (data.y != -1)

    emb = embeddings[valid_mask].detach().cpu().numpy()
    labels = data.y[valid_mask].detach().cpu().numpy()

    if emb.shape[0] < 5:
        print("⚠️ Not enough labeled nodes available for t-SNE.")
        return

    perplexity = min(30, max(5, emb.shape[0] // 10))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(8, 6))

    for cls, label_name in [(0, "licit"), (1, "illicit")]:
        cls_mask = (labels == cls)
        if cls_mask.sum() > 0:
            plt.scatter(
                emb_2d[cls_mask, 0],
                emb_2d[cls_mask, 1],
                s=18,
                alpha=0.65,
                label=label_name
            )

    plt.title("t-SNE of Final Layer Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()

    filepath = os.path.join(output_dir, file_name)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ t-SNE visualization saved to {filepath}")
# =============================================================