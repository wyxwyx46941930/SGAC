import argparse
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from network import SGAC
import os
from gnn import GNN
import random
import pickle
from data import ProteinTUDataset, convert_to_tudata
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
def test(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(config['device'])
            output = model.embed(data)
            
            # Get predicted probabilities and predicted labels
            probs = F.softmax(output, dim=-1)[:, 1]  # Probability of positive class
            preds = F.log_softmax(output, dim=-1).max(1)[1]
            
            # Collect predictions and true labels
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    # Convert predictions and labels to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate F1 Score
    f1 = f1_score(all_labels, all_preds.round())  # Round probabilities to 0 or 1 for binary classification
    
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)
    
    return f1, roc_auc, pr_auc

def train(config,args):
    dset_loaders = {}
    dset_loaders["train"] = DataLoader(config["train_dataset"], batch_size=args.batch_size,
            shuffle=True)
    dset_loaders["test"] = DataLoader(config["test_dataset"], batch_size=args.batch_size,
            shuffle=True)
    dset_loaders["val"] = DataLoader(config["val_dataset"], batch_size=args.batch_size,
            shuffle=True)

    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](model.parameters(), **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    best_val = 0
    best_data = None
    for i in tqdm(range(config["num_iterations"])):
        loss_params = config["loss"]
        model.train(True)
        optimizer.zero_grad()
        for inputs_train, inputs_test in zip(dset_loaders["train"], dset_loaders["test"]):
            inputs_train, inputs_test = inputs_train.to(config['device']), inputs_test.to(config['device'])
            classifier_loss, label_loss, contrastive_loss = model(inputs_train, inputs_test, device=config['device'])
            total_loss = classifier_loss + args.label_loss * label_loss + args.contrastive_loss * contrastive_loss
            # total_loss = args.label_loss * label_loss + args.contrastive_loss * contrastive_loss
            total_loss.backward()
            optimizer.step()
        val_f1, _, _ = test(model, dset_loaders["val"])
        test_f1, test_auc, test_pr_auc = test(model, dset_loaders["test"])
        if best_val < val_f1:
            best_val = val_f1 
            best_model = model.state_dict()
            best_data = [test_f1, test_auc, test_pr_auc]
            print(test_f1)
    # save_path = "model.pth"
    # torch.save(best_model, save_path)
    return best_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--test_interval', type=int, default=5, help="interval of two continuous test phase")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, sage or gcn (default: gin)')
    parser.add_argument('--drop_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0.0)')
    parser.add_argument('--label_loss', type=float, default=0.1,
                        help='Pseudo label loss')
    parser.add_argument('--contrastive_loss', type=float, default=0.1,
                        help='contrastive loss')
    parser.add_argument('--num_layer', type=int, default=4,# main feature 2
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--projection_size', type=int, default=256)
    parser.add_argument('--prediction_size', type=int, default=2)
    parser.add_argument('--projection_hidden_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=3407)

    args = parser.parse_args()
    print(args)
    config = {}
    # read data
    config['num_class'] = args.num_class = 2
    with open('protein_data.pkl', 'rb') as f:
        protein_data = pickle.load(f)
    data_list = [convert_to_tudata(protein_name, protein_data) for protein_name, protein_data in protein_data.items()]
    random.shuffle(data_list)
    total_length = len(data_list)
    train_size = int(total_length * 0.7)
    val_size = int(total_length * 0.2)
    test_size = total_length - train_size - val_size
    train_data = data_list[:train_size]
    val_data = data_list[train_size:train_size + val_size]
    test_data = data_list[train_size + val_size:]
    train_dataset = ProteinTUDataset(root=".", data_list=train_data)
    val_dataset = ProteinTUDataset(root=".", data_list=val_data)
    test_dataset = ProteinTUDataset(root=".", data_list=test_data)
    config['feature_dim'] = max(test_dataset.num_features, train_dataset.num_features)

    if torch.cuda.is_available():
        config['device'] = 'cuda:' + str(args.gpu_id)
    else:
        config['device'] = 'cpu'
    result = {}
    for seed in [3405,3406,3407,3408,3409]:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if args.gnn == 'gin':
            gnnmodel = GNN(gnn_type='gin', num_layer=args.num_layer, emb_dim=args.emb_dim,
                            drop_ratio=args.drop_ratio, feat_dim=config['feature_dim']).to(config['device'])
        elif args.gnn == 'sage':
            gnnmodel = GNN(gnn_type='sage', num_layer=args.num_layer, emb_dim=args.emb_dim,
                            drop_ratio=args.drop_ratio, feat_dim=config['feature_dim']).to(config['device'])
        
        elif args.gnn == 'gcn':
            gnnmodel = GNN(gnn_type='gcn', num_layer=args.num_layer, emb_dim=args.emb_dim,
                            drop_ratio=args.drop_ratio, feat_dim=config['feature_dim']).to(config['device'])
        else:
            raise ValueError('Invalid GNN type') 

        model = SGAC(gnnmodel, emb_dim=args.emb_dim, projection_size=args.projection_size,
                        prediction_size=args.prediction_size, projection_hidden_size=args.projection_hidden_size)
        model.to(config['device'])
        config["gpu"] = args.gpu_id
        config["num_iterations"] = args.epochs
        config["test_interval"] = args.test_interval

        config["loss"] = {"trade_off":1.0}
        config["loss"]["random"] = args.random
        config["loss"]["random_dim"] = 1024

        config["optimizer"] = {"type":optim.Adam, "optim_params":{'lr':args.lr}}#, 'class_num':train_dataset.num_classes}}

        config["train_dataset"] = train_dataset
        config["test_dataset"] = test_dataset
        config["val_dataset"] = val_dataset
        acc = train(config,args)
        result[seed] = acc
    print(result)
                