import torch
import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_graph(data, output_dir="img", num_nodes=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"📊 Visualizing a subgraph of {num_nodes} nodes...")
    
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