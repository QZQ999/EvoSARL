"""
Style Embedding Network for behavioral style representation.
Maps (state, action) pairs to a d-dimensional unit hypersphere.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StyleEncoder(nn.Module):
    """
    Encodes (state, action) pairs into a normalized d-dimensional embedding space.
    Trained with contrastive learning (InfoNCE loss).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, embedding_dim=64):
        super(StyleEncoder, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action encoder (for discrete actions, use embedding)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Joint encoder
        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, state, action):
        """
        Args:
            state: (batch_size, state_dim) or (batch_size, n_agents, state_dim)
            action: (batch_size, action_dim) or (batch_size, n_agents, action_dim) one-hot
        Returns:
            embedding: (batch_size, embedding_dim) or (batch_size, n_agents, embedding_dim)
                      normalized to unit sphere
        """
        # Handle multi-agent case
        original_shape = state.shape
        if len(state.shape) == 3:  # (bs, n_agents, dim)
            bs, n_agents, _ = state.shape
            state = state.reshape(bs * n_agents, -1)
            action = action.reshape(bs * n_agents, -1)

        # Encode state and action
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)

        # Concatenate and encode jointly
        joint_feat = torch.cat([state_feat, action_feat], dim=-1)
        embedding = self.joint_encoder(joint_feat)

        # L2 normalize to unit hypersphere
        embedding = F.normalize(embedding, p=2, dim=-1)

        # Restore shape if multi-agent
        if len(original_shape) == 3:
            embedding = embedding.reshape(bs, n_agents, -1)

        return embedding

    def encode_trajectory(self, states, actions, normalize=True):
        """
        Encode a trajectory by temporal averaging.

        Args:
            states: (T, state_dim) or (T, n_agents, state_dim)
            actions: (T, action_dim) or (T, n_agents, action_dim)
            normalize: Whether to L2 normalize the result
        Returns:
            z_tau: (embedding_dim,) or (n_agents, embedding_dim) averaged trajectory embedding
        """
        # Encode each timestep
        embeddings = self.forward(states, actions)  # (T, embedding_dim) or (T, n_agents, embedding_dim)

        # Temporal averaging
        z_tau = embeddings.mean(dim=0)  # (embedding_dim,) or (n_agents, embedding_dim)

        if normalize:
            z_tau = F.normalize(z_tau, p=2, dim=-1)

        return z_tau


class ContrastiveLearner:
    """
    Trains the StyleEncoder using InfoNCE contrastive learning.
    Positive pairs: temporally adjacent (s_t, a_t) and (s_{t+1}, a_{t+1})
    Negative pairs: state-action pairs from different trajectories
    """

    def __init__(self, encoder, lr=1e-3, temperature=0.07, device='cpu'):
        self.encoder = encoder
        self.temperature = temperature
        self.device = device

        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        self.encoder.to(device)

    def compute_infonce_loss(self, anchor, positive, negatives):
        """
        Compute InfoNCE loss.

        Args:
            anchor: (batch_size, embedding_dim)
            positive: (batch_size, embedding_dim)
            negatives: (batch_size, n_negatives, embedding_dim)
        Returns:
            loss: scalar
        """
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (batch_size,)

        # Negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # (batch_size, n_negatives)

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (batch_size, 1+n_negatives)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def train_batch(self, trajectories):
        """
        Train on a batch of trajectories.

        Args:
            trajectories: List of (states, actions) tuples
                         states: (T, state_dim), actions: (T, action_dim)
        Returns:
            loss: scalar loss value
        """
        self.encoder.train()
        self.optimizer.zero_grad()

        all_embeddings = []
        trajectory_indices = []

        # Encode all state-action pairs
        for traj_idx, (states, actions) in enumerate(trajectories):
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)

            embeddings = self.encoder(states, actions)  # (T, embedding_dim)
            all_embeddings.append(embeddings)
            trajectory_indices.extend([traj_idx] * len(embeddings))

        # Flatten all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, embedding_dim)
        trajectory_indices = torch.LongTensor(trajectory_indices).to(self.device)

        # Sample positive and negative pairs
        batch_size = min(256, len(all_embeddings) - 1)
        anchor_indices = torch.randint(0, len(all_embeddings) - 1, (batch_size,))

        anchors = all_embeddings[anchor_indices]
        positives = all_embeddings[anchor_indices + 1]  # Next timestep as positive

        # Sample negatives from different trajectories
        n_negatives = 128
        negatives_list = []
        for i in anchor_indices:
            traj_id = trajectory_indices[i]
            # Get indices from different trajectories
            diff_traj_mask = trajectory_indices != traj_id
            diff_traj_indices = torch.where(diff_traj_mask)[0]

            if len(diff_traj_indices) >= n_negatives:
                neg_idx = diff_traj_indices[torch.randperm(len(diff_traj_indices))[:n_negatives]]
            else:
                # If not enough, sample with replacement
                neg_idx = diff_traj_indices[torch.randint(0, len(diff_traj_indices), (n_negatives,))]

            negatives_list.append(all_embeddings[neg_idx])

        negatives = torch.stack(negatives_list, dim=0)  # (batch_size, n_negatives, embedding_dim)

        # Compute loss
        loss = self.compute_infonce_loss(anchors, positives, negatives)

        # Backward
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def sample_prototypes_on_sphere(n_prototypes, embedding_dim, method='uniform'):
    """
    Sample prototype vectors uniformly on the unit hypersphere.

    Args:
        n_prototypes: Number of prototypes to sample
        embedding_dim: Dimension of the embedding space
        method: 'uniform' or 'fibonacci' for sampling strategy
    Returns:
        prototypes: (n_prototypes, embedding_dim) normalized vectors
    """
    if method == 'uniform':
        # Sample from Gaussian and normalize (generates uniform distribution on sphere)
        prototypes = np.random.randn(n_prototypes, embedding_dim)
        prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

    elif method == 'fibonacci':
        # Fibonacci sphere sampling (more uniform for low dimensions)
        # For high dimensions, fall back to Gaussian sampling
        if embedding_dim > 3:
            prototypes = np.random.randn(n_prototypes, embedding_dim)
            prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
        else:
            raise NotImplementedError("Fibonacci sphere only implemented for d>3 via Gaussian fallback")

    return torch.FloatTensor(prototypes)


def compute_trajectory_embedding(encoder, states, actions, device='cpu'):
    """
    Compute normalized trajectory embedding z_tau.

    Args:
        encoder: Trained StyleEncoder
        states: (T, state_dim) numpy array
        actions: (T, action_dim) numpy array (one-hot)
        device: torch device
    Returns:
        z_tau: (embedding_dim,) normalized trajectory embedding
    """
    encoder.eval()
    with torch.no_grad():
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        z_tau = encoder.encode_trajectory(states, actions, normalize=True)
    return z_tau
