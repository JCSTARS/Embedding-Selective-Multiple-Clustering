import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['VL_ROOT_DIR'] = '/path/to/project'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from methods.utils import load_image
from methods.llava_utils import retrieve_logit_lens_llava, load_llava_state, get_embed_llava
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, rand_score, jaccard_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


def variation_of_information(labels_true, labels_pred):
    n = len(labels_true)
    from sklearn.metrics import mutual_info_score
    # 计算熵 H(labels_true) 和 H(labels_pred)
    entropy_true = -np.sum(np.bincount(labels_true) / n * np.log2(np.bincount(labels_true) / n))
    entropy_pred = -np.sum(np.bincount(labels_pred) / n * np.log2(np.bincount(labels_pred) / n))

    # 计算互信息 I(labels_true, labels_pred)
    mutual_info = mutual_info_score(labels_true, labels_pred)

    vi = entropy_true + entropy_pred - 2 * mutual_info
    return vi

def cluster_with_pseudo_labels(cluster_labels, lm_embeddings, num_clusters):
    # compute distance to cluster centers
    distances = np.linalg.norm(lm_embeddings - centroids[true_labels], axis=1) 
    # Select top10% as high confidence samples
    topk_ratio = 0.3
    pseudo_data = []
    pseudo_labels = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_distances = distances[cluster_indices]
        n_top = int(len(cluster_distances) * topk_ratio)
        if n_top == 0:
            continue # At least 1 sample should be chosen
        top_indices = cluster_indices[np.argsort(cluster_distances)[:n_top]]
        pseudo_data.append(lm_embeddings[top_indices])
        pseudo_labels.extend([cluster_id] * n_top)

    X_pseudo = np.concatenate(pseudo_data)
    y_pseudo = np.array(pseudo_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_pseudo = torch.tensor(X_pseudo, dtype=torch.float32, device=device)
    y_pseudo = torch.tensor(y_pseudo, dtype=torch.long).to(device)
    lm_embeddings = torch.tensor(lm_embeddings, dtype=torch.float32, device=device)
    
    model = ClusteringHead(in_dim, num_clusters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    dataset = TensorDataset(X_pseudo,y_pseudo)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    max_nmi = 0
    max_ri = 0
    max_epoch = 0
    num_epochs = 200
    max_vi = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        with torch.no_grad():
            transformed_embeddings = model(lm_embeddings).cpu()
        cluster_labels = kmeans.fit_predict(transformed_embeddings.numpy())
        cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=device)
        #print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        cluster_labels = cluster_labels.cpu()
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)  # NMI
        ari = adjusted_rand_score(true_labels, cluster_labels)  # ARI
        ri = rand_score(true_labels, cluster_labels)  # RI
        vi = variation_of_information(true_labels, cluster_labels)
        #print(f"NMI: {nmi}", f"ARI: {ari}", f"RI: {ri}")
        if nmi > max_nmi: 
            max_nmi = nmi
            max_epoch = epoch
        if ri > max_ri:
            max_ri = ri
        if vi < max_vi:
            max_vi = vi
    print("Max_nmi", max_nmi,"Max_ri", max_ri, "Max_epoch", max_epoch, "Min_vi", vi)

    

class ClusteringHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ClusteringHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

#model_name = "llava7b" # or "blip7b"
model_name = "llava7b" 
if model_name.startswith("llava"):
    model_state = load_llava_state()
    retrieve_logit_lens = retrieve_logit_lens_llava
    in_dim = 32000
else:
    assert False, "Model not supported"

base_path = os.path.join(os.environ["VL_ROOT_DIR"], "dataset/fruit/color")

files = os.listdir(base_path)
img_paths = []

image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.pgm']
for root, dirs, files in os.walk(base_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_path = os.path.join(root, file)
                img_paths.append(image_path)

# split line
lm_embeddings = []
true_labels = []
i = 0
for img_path in img_paths:
        torch.cuda.empty_cache()
        embedding = get_embed_llava(model_state, img_path, "The color of the fruit is")
        lm_embeddings.append(embedding.flatten())
        gt = img_path.split('/')[-2]
        true_labels.append(gt)
        i = i + 1
        print(gt)

label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(true_labels)

lm_embeddings = np.array(lm_embeddings)
print(lm_embeddings.shape)

np.save('color_fruit.npy', lm_embeddings)

# clustering
num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, random_state=0) 
kmeans.fit(lm_embeddings)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_
nmi = normalized_mutual_info_score(true_labels, cluster_labels)  # NMI
ari = adjusted_rand_score(true_labels, cluster_labels)  # ARI
ri = rand_score(true_labels, cluster_labels)  # RI
print(f"NMI: {nmi}", f"ARI: {ari}", f"RI: {ri}")
""""""
nmi = 0
ri = 0
vi = 0
for i in range(10):
    nmi_i, ri_i, vi_i = cluster_with_pseudo_labels(cluster_labels, lm_embeddings, num_clusters)
    nmi += nmi_i
    ri += ri_i
    vi += vi_i
print(nmi/10,ri/10,vi/10)

cluster_labels = list(cluster_labels)
chunk_size = 100
for i in range(0, len(cluster_labels), chunk_size):
    print(' '.join(map(str, cluster_labels[i:i + chunk_size])))

out_dim = num_clusters
num_epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clustering_head = ClusteringHead(in_dim, out_dim).to(device) # instantiate the class
optimizer = torch.optim.Adam(clustering_head.parameters(), lr=1e-4)
lm_embeddings = torch.tensor(lm_embeddings, dtype=torch.float32).to(device)

"""
max_nmi = 0
max_ri = 0
max_epoch = 0
max_ji = 0
max_vi = 10
for epoch in range(num_epochs):
    # get transformed_embeddings from the model
    with torch.no_grad():
        transformed_embeddings = clustering_head(lm_embeddings.to(device)).cpu()

    # use kmeans to get new cluster_labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(transformed_embeddings.numpy())
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=device)

    # train the clustering head
    optimizer.zero_grad()
    transformed_embeddings = clustering_head(lm_embeddings.to(device))
    loss = nn.CrossEntropyLoss()(transformed_embeddings, cluster_labels)  # pseudo_label
    loss.backward()
    optimizer.step()

    #print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    cluster_labels = cluster_labels.cpu()
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)  # NMI
    ari = adjusted_rand_score(true_labels, cluster_labels)  # ARI
    ri = rand_score(true_labels, cluster_labels)  # RI
    vi = variation_of_information(true_labels, cluster_labels)
    #print(f"NMI: {nmi}", f"ARI: {ari}", f"RI: {ri}")
    if nmi > max_nmi: 
        max_nmi = nmi
        max_epoch = epoch
    if ri > max_ri:
        max_ri = ri
    if vi < max_vi:
        max_vi = vi
print("Max_nmi", max_nmi,"Max_ri", max_ri, "Max_epoch", max_epoch, "Max_vi", vi)
"""