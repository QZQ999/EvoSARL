"""
Prototype Space Management for style-controlled teammate evolution.
Handles prototype initialization, elimination, merging, and reference policy tracking.
"""
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from collections import defaultdict


class PrototypeManager:
    """
    Manages the prototype space and associated teammate populations.
    Each prototype λ_k maintains:
    - A population of teammates P_k
    - A reference policy π_ref^(k) (best performer)
    - A historical experience buffer H_k for exploration rewards
    """

    def __init__(self, n_initial_prototypes, embedding_dim, population_size_per_prototype,
                 elimination_threshold=0.0, merging_threshold=0.9, evaluation_window=5):
        """
        Args:
            n_initial_prototypes: K_0, initial number of prototypes
            embedding_dim: Dimension of style embedding space
            population_size_per_prototype: n_k for each prototype
            elimination_threshold: θ_elim for removing weak prototypes
            merging_threshold: θ_merge for merging similar prototypes
            evaluation_window: W, window for performance evaluation
        """
        self.embedding_dim = embedding_dim
        self.population_size = population_size_per_prototype
        self.elimination_threshold = elimination_threshold
        self.merging_threshold = merging_threshold
        self.evaluation_window = evaluation_window

        # Prototype vectors (λ_k)
        self.prototypes = {}  # {prototype_id: torch.Tensor(embedding_dim)}
        self.next_prototype_id = 0

        # Initialize K_0 prototypes uniformly on sphere
        from modules.style_encoder import sample_prototypes_on_sphere
        initial_prototypes = sample_prototypes_on_sphere(n_initial_prototypes, embedding_dim)
        for proto in initial_prototypes:
            self.prototypes[self.next_prototype_id] = proto
            self.next_prototype_id += 1

        # Population for each prototype {prototype_id: {individual_id: mac}}
        self.populations = {pid: {} for pid in self.prototypes.keys()}
        self.next_individual_ids = {pid: 0 for pid in self.prototypes.keys()}

        # Reference policies {prototype_id: (mac, performance, z_ref)}
        self.reference_policies = {pid: None for pid in self.prototypes.keys()}

        # Historical experience buffers {prototype_id: [(embedding, (s, a))]}
        self.history_buffers = {pid: [] for pid in self.prototypes.keys()}
        self.max_history_size = 10000

        # Performance tracking {prototype_id: [perf_1, perf_2, ...]}
        self.performance_history = {pid: [] for pid in self.prototypes.keys()}

    def assign_to_prototype(self, z_tau):
        """
        Assign a trajectory embedding to the nearest prototype.

        Args:
            z_tau: (embedding_dim,) normalized trajectory embedding
        Returns:
            prototype_id: ID of the assigned prototype
            similarity: Cosine similarity score
        """
        max_sim = -1
        assigned_id = None

        for pid, proto in self.prototypes.items():
            sim = torch.dot(z_tau, proto).item()
            if sim > max_sim:
                max_sim = sim
                assigned_id = pid

        return assigned_id, max_sim

    def add_individual(self, prototype_id, mac, performance=None):
        """
        Add a new individual to a prototype's population.

        Args:
            prototype_id: Target prototype
            mac: Multi-agent controller (teammate strategy)
            performance: Optional performance score
        Returns:
            individual_id: Unique ID for this individual
        """
        individual_id = self.next_individual_ids[prototype_id]
        self.populations[prototype_id][individual_id] = {
            'mac': mac,
            'performance': performance
        }
        self.next_individual_ids[prototype_id] += 1
        return individual_id

    def update_reference_policy(self, prototype_id, mac, performance, z_ref):
        """
        Update the reference policy for a prototype if this one performs better.

        Args:
            prototype_id: Target prototype
            mac: Candidate reference policy
            performance: Performance score (higher is better)
            z_ref: Trajectory embedding of this policy
        """
        current_ref = self.reference_policies[prototype_id]

        if current_ref is None or performance > current_ref['performance']:
            self.reference_policies[prototype_id] = {
                'mac': deepcopy(mac),
                'performance': performance,
                'z_ref': z_ref.clone()
            }

    def get_reference_embedding(self, prototype_id):
        """
        Get the reference trajectory embedding z_ref^(k) for a prototype.

        Args:
            prototype_id: Target prototype
        Returns:
            z_ref: (embedding_dim,) reference embedding, or None if no reference exists
        """
        ref = self.reference_policies[prototype_id]
        return ref['z_ref'] if ref is not None else None

    def add_to_history(self, prototype_id, embedding, state_action_pair):
        """
        Add a state-action embedding to the historical buffer for exploration reward.

        Args:
            prototype_id: Target prototype
            embedding: (embedding_dim,) embedding of (s, a)
            state_action_pair: (state, action) tuple
        """
        self.history_buffers[prototype_id].append({
            'embedding': embedding.clone(),
            'state_action': state_action_pair
        })

        # Limit buffer size
        if len(self.history_buffers[prototype_id]) > self.max_history_size:
            self.history_buffers[prototype_id].pop(0)

    def compute_knn_exploration_reward(self, prototype_id, embedding, k=5):
        """
        Compute k-NN based exploration reward.
        r_explore = ||φ(s',a') - NN_k(φ(s',a'), H_k)||_2

        Args:
            prototype_id: Target prototype
            embedding: (embedding_dim,) current state-action embedding
            k: k-th nearest neighbor to use
        Returns:
            reward: Exploration reward (larger = more novel)
        """
        history = self.history_buffers[prototype_id]

        if len(history) < k:
            # Not enough history, encourage exploration
            return 1.0

        # Compute distances to all historical embeddings
        historical_embeddings = torch.stack([h['embedding'] for h in history])  # (N, embedding_dim)
        distances = torch.norm(historical_embeddings - embedding.unsqueeze(0), dim=1)  # (N,)

        # Get k-th nearest neighbor distance
        kth_distance = torch.kthvalue(distances, k).values.item()

        return kth_distance

    def record_performance(self, prototype_id, performance):
        """
        Record performance for a prototype's best individual.

        Args:
            prototype_id: Target prototype
            performance: Performance score
        """
        self.performance_history[prototype_id].append(performance)

        # Keep only recent history
        if len(self.performance_history[prototype_id]) > self.evaluation_window:
            self.performance_history[prototype_id].pop(0)

    def should_eliminate(self, prototype_id):
        """
        Check if a prototype should be eliminated based on poor performance.

        Args:
            prototype_id: Target prototype
        Returns:
            should_eliminate: Boolean
        """
        history = self.performance_history[prototype_id]

        if len(history) < self.evaluation_window:
            return False

        avg_performance = np.mean(history)
        return avg_performance < self.elimination_threshold

    def compute_population_similarity(self, pid1, pid2, encoder):
        """
        Compute behavioral similarity between two prototype populations.
        sim(P_i, P_j) = (1 / n_i * n_j) * Σ Σ z_πa^T z_πb

        Args:
            pid1, pid2: Prototype IDs
            encoder: StyleEncoder for computing embeddings
        Returns:
            similarity: Scalar similarity score
        """
        # This requires trajectory embeddings for all individuals
        # For efficiency, we can approximate using reference embeddings
        ref1 = self.reference_policies[pid1]
        ref2 = self.reference_policies[pid2]

        if ref1 is None or ref2 is None:
            return 0.0

        similarity = torch.dot(ref1['z_ref'], ref2['z_ref']).item()
        return similarity

    def find_mergeable_pairs(self, encoder):
        """
        Find prototype pairs that should be merged based on similarity.

        Args:
            encoder: StyleEncoder for computing embeddings
        Returns:
            pairs: List of (pid1, pid2) tuples to merge
        """
        pairs = []
        prototype_ids = list(self.prototypes.keys())

        for i, pid1 in enumerate(prototype_ids):
            for pid2 in prototype_ids[i + 1:]:
                similarity = self.compute_population_similarity(pid1, pid2, encoder)
                if similarity > self.merging_threshold:
                    pairs.append((pid1, pid2))

        return pairs

    def merge_prototypes(self, pid1, pid2):
        """
        Merge two prototypes: keep pid1, remove pid2, merge populations.

        Args:
            pid1: Prototype to keep
            pid2: Prototype to remove
        """
        # Merge populations
        for ind_id, ind_data in self.populations[pid2].items():
            new_id = self.next_individual_ids[pid1]
            self.populations[pid1][new_id] = ind_data
            self.next_individual_ids[pid1] += 1

        # Merge history buffers
        self.history_buffers[pid1].extend(self.history_buffers[pid2])
        if len(self.history_buffers[pid1]) > self.max_history_size:
            self.history_buffers[pid1] = self.history_buffers[pid1][-self.max_history_size:]

        # Update reference if pid2's is better
        if self.reference_policies[pid2] is not None:
            if self.reference_policies[pid1] is None or \
                    self.reference_policies[pid2]['performance'] > self.reference_policies[pid1]['performance']:
                self.reference_policies[pid1] = self.reference_policies[pid2]

        # Remove pid2
        del self.prototypes[pid2]
        del self.populations[pid2]
        del self.reference_policies[pid2]
        del self.history_buffers[pid2]
        del self.performance_history[pid2]
        del self.next_individual_ids[pid2]

    def eliminate_prototype(self, prototype_id):
        """
        Remove a prototype and all associated data.

        Args:
            prototype_id: Prototype to remove
        """
        del self.prototypes[prototype_id]
        del self.populations[prototype_id]
        del self.reference_policies[prototype_id]
        del self.history_buffers[prototype_id]
        del self.performance_history[prototype_id]
        del self.next_individual_ids[prototype_id]

    def adaptive_management(self, encoder, generation):
        """
        Perform elimination and merging every W generations.

        Args:
            encoder: StyleEncoder
            generation: Current generation number
        Returns:
            eliminated: List of eliminated prototype IDs
            merged: List of (pid1, pid2) merged pairs
        """
        eliminated = []
        merged = []

        # Elimination
        for pid in list(self.prototypes.keys()):
            if self.should_eliminate(pid):
                self.eliminate_prototype(pid)
                eliminated.append(pid)

        # Merging
        mergeable_pairs = self.find_mergeable_pairs(encoder)
        for pid1, pid2 in mergeable_pairs:
            if pid1 in self.prototypes and pid2 in self.prototypes:  # Check both still exist
                self.merge_prototypes(pid1, pid2)
                merged.append((pid1, pid2))

        return eliminated, merged

    def get_active_prototypes(self):
        """
        Get list of currently active prototype IDs.
        """
        return list(self.prototypes.keys())

    def get_population(self, prototype_id):
        """
        Get the population for a prototype.

        Args:
            prototype_id: Target prototype
        Returns:
            population: Dict of {individual_id: {'mac': mac, 'performance': perf}}
        """
        return self.populations[prototype_id]

    def get_population_size(self, prototype_id):
        """
        Get current population size for a prototype.
        """
        return len(self.populations[prototype_id])

    def summary(self):
        """
        Get a summary of the current prototype space.
        """
        summary = {
            'n_prototypes': len(self.prototypes),
            'prototype_ids': list(self.prototypes.keys()),
            'population_sizes': {pid: len(pop) for pid, pop in self.populations.items()},
            'has_reference': {pid: self.reference_policies[pid] is not None
                              for pid in self.prototypes.keys()},
            'avg_performances': {pid: np.mean(hist) if hist else 0.0
                                 for pid, hist in self.performance_history.items()}
        }
        return summary
