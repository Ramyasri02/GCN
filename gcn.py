import pandas as pd
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F

def convert_to_numeric(df):
    for column in df.columns:
        if column != "Antibiotic_Name" and column != "kmer_pairs":
            df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.fillna(0)
    return df

df_antibiotic_node_features = pd.read_csv("antibiotic_node_features.csv", dtype={"Antibiotic_Name": str})
df_antibiotic_edge_features = pd.read_csv("printed_data.csv", sep=',', dtype={"Antibiotics_Name": str})

df_antibiotic_node_features = convert_to_numeric(df_antibiotic_node_features)
df_antibiotic_edge_features = convert_to_numeric(df_antibiotic_edge_features)

df_kmer_node_features = pd.read_csv("kmer_node_features_with_names.csv", dtype={"kmer_name": str})
df_kmer_edge_features = pd.read_csv("edge_data_modified.csv", dtype={"kmer_pairs": str})
df_additional_edge_features = pd.read_csv("additional_edge_features.csv", sep='\t', dtype={"coocurrence": str}, encoding='latin1')

df_kmer_node_features = convert_to_numeric(df_kmer_node_features)
df_kmer_edge_features = convert_to_numeric(df_kmer_edge_features)
df_additional_edge_features = convert_to_numeric(df_additional_edge_features)

antibiotic_node_features = torch.from_numpy(df_antibiotic_node_features.values[:, 1:].astype('float32'))
antibiotic_name_to_index = {name: i for i, name in enumerate(df_antibiotic_node_features["Antibiotic_Name"])}

kmer_node_features = torch.tensor(df_kmer_node_features.values[:, 1:], dtype=torch.float)
kmer_name_to_index = {name: i for i, name in enumerate(df_kmer_node_features["kmer_name"])}

src_antibiotic = []
dst_antibiotic = []

for i, row in df_antibiotic_edge_features.iterrows():
    names = row["Antibiotic_Name"].split(" - ")
    if all(name in antibiotic_name_to_index for name in names):
        src_antibiotic.append(antibiotic_name_to_index[names[0]])
        dst_antibiotic.append(antibiotic_name_to_index[names[1]])

antibiotic_edge_index = torch.tensor([src_antibiotic, dst_antibiotic], dtype=torch.long)

src_kmer = []
dst_kmer = []

for i, row in df_kmer_edge_features.iterrows():
    names = row["kmer_pairs"].split(" - ")
    if all(name in kmer_name_to_index for name in names):
        src_kmer.append(kmer_name_to_index[names[0]])
        dst_kmer.append(kmer_name_to_index[names[1]])

kmer_edge_index = torch.tensor([src_kmer, dst_kmer], dtype=torch.long)
additional_edge_features = torch.tensor(df_additional_edge_features.values[:, 1:], dtype=torch.float)
data_kmer = Data(x=kmer_node_features, edge_index=kmer_edge_index, edge_attr=additional_edge_features)

class MultiGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_size):
        super(MultiGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels*2, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)

    def forward(self, x, edge_index, edge_attr):
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, training=self.training)
        x2 = self.conv2(x1, edge_index, edge_attr)

        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class MultiGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_size):
        super(MultiGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels*2, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_size = antibiotic_node_features.shape[1] + additional_edge_features.shape[1]
hidden_size = 160
output_size = 1

num_features_antibiotic = antibiotic_node_features.shape[1]
num_features_kmer = kmer_node_features.shape[1]

model_antibiotic_1 = MultiGCN(num_features_antibiotic, 80, hidden_size)
model_antibiotic_2 = MultiGCN(160, 160, hidden_size)  # hidden_channels changed to 160

model_kmer_1 = MultiGCN(num_features_kmer, 80, hidden_size)
model_kmer_2 = MultiGCN(160, 160, hidden_size)  # hidden_channels changed to 160

data_antibiotic = Data(x=antibiotic_node_features, edge_index=antibiotic_edge_index)
data_kmer = Data(x=kmer_node_features, edge_index=kmer_edge_index, edge_attr=additional_edge_features)

output_antibiotic_1 = model_antibiotic_1(data_antibiotic.x, data_antibiotic.edge_index, data_antibiotic.edge_attr)
output_antibiotic_2 = model_antibiotic_2(output_antibiotic_1, data_antibiotic.edge_index, data_antibiotic.edge_attr)

output_kmer_1 = model_kmer_1(data_kmer.x, data_kmer.edge_index, data_kmer.edge_attr)
output_kmer_2 = model_kmer_2(output_kmer_1, data_kmer.edge_index, data_kmer.edge_attr)

output_antibiotic_avg = output_antibiotic_2.mean(dim=0)
output_kmer_avg = output_kmer_2.mean(dim=0)

output_combined = torch.cat((output_antibiotic_avg, output_kmer_avg), dim=0)

model_mlp = MLP(input_size, hidden_size, output_size)
output_final = model_mlp(output_combined)


