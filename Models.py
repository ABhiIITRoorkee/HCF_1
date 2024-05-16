import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from utility.load_data import *
from utility.helper import *
from utility.batch_test import *
from utility.parser import *

#This source file is based on the GRec published by Bo Li et al.
#We would like to thank and offer our appreciation to them.
#Original algorithm can be found in paper: Embedding App-Library Graph for Neural Third Party Library Recommendation. ESEC/FSE ’21


data_loader = Data(args.data_path + args.dataset, batch_size=args.batch_size)
user_features_df = data_loader.load_userss_features()
user_features = user_features_df[[f'feature{i}' for i in range(20)]].values
print("shape of User features",user_features.shape)

def create_user_adjacency_matrix(features_path, n_components, device):
            
            #node_features = np.random.randn(31421, 20)  # Assume features have been loaded here
            # node_features_df = data_loader.load_node_features()
            # node_features = node_features_df[[f'feature{i}' for i in range(20)]].values
            #print("shape of node features",node_features.shape)
            #print("Starting GMM clustering with n_components:", n_components)
            gmm = GaussianMixture(n_components=n_components, random_state=142)
            labels = gmm.fit_predict(user_features)
            #print("GMM clustering completed. Labels:", np.unique(labels, return_counts=True))

            n_nodes = user_features.shape[0]
            adj_cat_user = np.zeros((n_nodes, n_nodes), dtype=float)

            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):  # This ensures we only compute half and mirror it to save computation
                    adj_cat_user[i, j] = adj_cat_user[j, i] = 1 if labels[i] == labels[j] else 0

            adj_cat_user = torch.FloatTensor(adj_cat_user).to(device)  # Ensure it's on the correct device
            #print("adj_cat_user  matrix shape:", adj_cat_user.shape)
            return adj_cat_user


class HCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, layer_num, dropout_list):
        super(HCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.n_layers = layer_num
        self.dropout_list = nn.ModuleList([nn.Dropout(p) for p in dropout_list])

        torch.manual_seed(50)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()



    def _init_weight_(self):
        torch.manual_seed(50)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    

    def forward(self, adj_u1, adj_u2, adj_i1, adj_i2, adj_cat):

        device = self.user_embedding.weight.device  # Get the device of the model parameters
        # print("shape of adj_u1", adj_u1.shape)
        # print("shape of adj_u2", adj_u2.shape)
        # print("shape of adj_i1", adj_i1.shape)
        # print("shape of adj_i2", adj_i2.shape)

        # Create the adjacency matrix using the modified function

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Example loading of node_features; replace with actual data
        #node_features = np.random.randn(31421, 20) 
        node_features_df = data_loader.load_node_features()
        node_features = node_features_df[[f'feature{i}' for i in range(20)]].values
        adj_cat_user = create_user_adjacency_matrix(node_features, n_components=10, device=device)
        #print("shape of adj_cat_user",adj_cat_user.shape)

        # Ensure all tensors are on the same device as model parameters
        adj_u1 = adj_u1.to(device)
        adj_u2 = adj_u2.to(device)
        adj_i1 = adj_i1.to(device)
        adj_i2 = adj_i2.to(device)
        adj_cat = adj_cat.to(device) 

        hu = self.user_embedding.weight
        hi = self.item_embedding.weight

        # print("Size of user embeddings (hu):", hu.shape)
        # print("Size of item embeddings (hi):", hi.shape)

        # User embeddings update
        user_embeddings = [hu]
        print("user_embeddings [-1] shape ",user_embeddings[-1].shape )  # (31421, 64)
        for i in range(self.n_layers):
            t = torch.sparse.mm(adj_u2, user_embeddings[-1])
            #print("Iteration adj_u2 :", i)
            #print("adj_cat shape:", adj_cat.shape)
            #print("t shape:", t.shape)
            t = torch.sparse.mm(adj_u1, t)

            #print("t shape:", t.shape)
        
            # Adding the new categorical scale
            #t_cat = torch.sparse.mm(adj_cat, t) 
            t_cat = torch.sparse.mm(adj_cat_user, t)
            # print("t_cat shape:", t_cat.shape)   
            t_cat = self.dropout_list[i](t_cat)  # Applying dropout to the updated embeddings
            user_embeddings.append(t_cat)
        u_emb = torch.mean(torch.stack(user_embeddings, dim=1), dim=1)

        # Item embeddings update
        item_embeddings = [hi]

        #print("item_embeddings [-1] shape ",item_embeddings[-1].shape )  #[727,64]

        for i in range(self.n_layers):
            t = torch.sparse.mm(adj_i2, item_embeddings[-1])    #(727,727) * (727,64)
            t = torch.sparse.mm(adj_i1, t)        #([727, 727]) * (727,64) 

            #print("Iteration adj_i1 :", i)
            # print("adj_cat shape:", adj_cat.shape)  #[31421, 31421])
            # print("t :", t.shape)                #[727, 64])

            # Adding the new categorical scale
            t_cat = torch.sparse.mm(adj_cat, t)          
            t_cat = self.dropout_list[i](t_cat)  # Applying dropout to the updated embeddings
            item_embeddings.append(t_cat)
        i_emb = torch.mean(torch.stack(item_embeddings, dim=1), dim=1)

        print("shape of u_emb", u_emb.shape)
        print("shape of i_emb", i_emb.shape)

        return u_emb, i_emb
    





# shape of adj_u1 torch.Size([31421, 31421])
# shape of adj_u2 torch.Size([31421, 31421])

# shape of adj_i1 torch.Size([727, 727])
# shape of adj_i2 torch.Size([727, 727])


# shape of u_emb torch.Size([31421, 64])
# shape of i_emb torch.Size([727, 64])


