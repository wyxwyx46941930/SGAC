import numpy as np
import math
from torch_scatter import scatter
import torch
from torch import nn
import torch.nn.functional as F
import copy
from functools import wraps
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class MLP(nn.Module):
    def __init__(self, dim, hidden_size, projection_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn
    
class SGAC(nn.Module):
    def __init__(self, net, emb_dim=300, projection_hidden_size=2048, projection_size=512,
                 prediction_size=2, pos_weight=1.0, neg_weight=0.2, initial_temperature=2.0,
                 confidence_threshold=0.7, soft_label_weight=0.1, contrastive_weight=0.3):
        super().__init__()
        self.online_encoder = net
        self.prediction_size = prediction_size
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))  # Learnable temperature
        self.online_projector = MLP(emb_dim, projection_hidden_size, projection_size)
        self.predictor = MLP(projection_size, projection_hidden_size, prediction_size)
        self.confidence_threshold = confidence_threshold
        self.soft_label_weight = soft_label_weight
        self.contrastive_weight = contrastive_weight
        self.cluster_centers = None  
        
    def apply_temperature_scaling(self, logits):
        return logits / self.temperature

    def generate_pseudo_labels_dynamic(self, features, device):
        with torch.no_grad():
            if self.cluster_centers is None:
                estimator = KMeans(n_clusters=self.prediction_size, n_init=10)
                estimator.fit(features.clone().detach().cpu())
                cluster_centers = torch.tensor(estimator.cluster_centers_, device=device, dtype=features.dtype)
            else:
                cluster_centers = self.cluster_centers

            distances = torch.cdist(features.clone().detach(), cluster_centers, p=2)
            probabilities = torch.nn.functional.softmax(-distances / self.temperature, dim=1)
            confidences, pseudo_labels = probabilities.max(dim=1)
            confident_mask = confidences > self.confidence_threshold

            confident_features = features[confident_mask]
            confident_labels = pseudo_labels[confident_mask]

            updated_cluster_centers = []
            for cluster_idx in range(self.prediction_size):
                cluster_mask = confident_labels == cluster_idx
                if cluster_mask.sum() > 0:
                    new_center = confident_features[cluster_mask].mean(dim=0)
                else:
                    new_center = cluster_centers[cluster_idx]
                updated_cluster_centers.append(new_center)
            updated_cluster_centers = torch.stack(updated_cluster_centers)

            self.cluster_centers = updated_cluster_centers
            return pseudo_labels, confident_mask, probabilities

    def compute_pseudo_label_loss(self, scaled_outputs_test, vice_features_test, device):
        pseudo_labels, confident_mask, probabilities = self.generate_pseudo_labels_dynamic(vice_features_test, device)

        test_confident_outputs = scaled_outputs_test[confident_mask].detach()
        test_confident_labels = pseudo_labels[confident_mask]
        test_confident_probabilities = probabilities[confident_mask]

        positive_mask = (test_confident_labels == 1)
        negative_mask = (test_confident_labels == 0)

        positive_outputs = test_confident_outputs[positive_mask]
        positive_probs = test_confident_probabilities[positive_mask]
        negative_outputs = test_confident_outputs[negative_mask]
        negative_probs = test_confident_probabilities[negative_mask]

        positive_loss = nn.CrossEntropyLoss()(positive_outputs, positive_probs.argmax(dim=1)) if positive_outputs.size(0) > 0 else 0.0
        negative_loss = nn.CrossEntropyLoss()(negative_outputs, negative_probs.argmax(dim=1)) if negative_outputs.size(0) > 0 else 0.0

        pseudo_label_loss = self.pos_weight * positive_loss + self.neg_weight * negative_loss
        confidence_mean = test_confident_probabilities.mean().item()
        dynamic_weight = self.soft_label_weight * (1 - confidence_mean)
        soft_loss = dynamic_weight * F.kl_div(F.log_softmax(test_confident_outputs, dim=1),
                                              test_confident_probabilities, reduction="batchmean")

        pseudo_label_loss += soft_loss
        return pseudo_label_loss

    def contrastive_loss_fn(self, embeddings, labels, margin=1.0):
        embeddings = F.normalize(embeddings, dim=1)
        distances = torch.cdist(embeddings, embeddings, p=2) / self.temperature
        label_equal = labels.unsqueeze(1) == labels.unsqueeze(0)

        positive_loss = distances.pow(2) * label_equal * self.pos_weight
        negative_loss = F.relu(margin - distances).pow(2) * ~label_equal * self.neg_weight
        num_pairs = embeddings.size(0) * (embeddings.size(0) - 1)
        contrastive_loss = (positive_loss.sum() + negative_loss.sum()) / num_pairs
        return contrastive_loss

    def forward(self, train_batch, test_batch, device='cpu'):
        train_vice_features, train_features = self.online_encoder(train_batch)
        test_vice_features, test_features = self.online_encoder(test_batch)

        train_outputs = self.predictor(self.online_projector(train_features))
        classifier_loss = nn.CrossEntropyLoss()(train_outputs, train_batch.y)

        scaled_test_outputs = self.apply_temperature_scaling(
            self.predictor(self.online_projector(test_features))
        )
        pseudo_label_loss = self.compute_pseudo_label_loss(scaled_test_outputs, test_vice_features, device)

        combined_features = torch.cat([train_features, test_features.detach()], dim=0)
        combined_labels = torch.cat([train_batch.y, -torch.ones_like(test_batch.y)], dim=0)  # Test data: -1
        contrastive_loss = self.contrastive_weight * self.contrastive_loss_fn(combined_features, combined_labels)

        return classifier_loss, pseudo_label_loss, contrastive_loss

    def embed(self, test):
        features_test, _ = self.online_encoder(test)
        online_pred_two = self.online_projector(features_test)
        outputs_test = self.predictor(online_pred_two)
        return outputs_test.detach()