"""
Reward shaping utilities for style-controlled teammate training.
Implements r_ref (style control) and r_explore (novelty-based exploration).
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_style_control_reward(batch, style_encoder, z_ref, device='cpu'):
    """
    Compute style control reward: r_ref(s,a|z_ref) = φ(s,a)^T z_ref

    Encourages alignment with reference policy's behavioral style.

    Args:
        batch: EpisodeBatch containing states and actions
        style_encoder: Trained StyleEncoder network
        z_ref: (embedding_dim,) reference style embedding
        device: torch device
    Returns:
        rewards: (batch_size, seq_len, 1) style control rewards
    """
    style_encoder.eval()

    # Extract states and actions from batch
    # batch["obs"]: (batch_size, seq_len, n_agents, obs_dim)
    # batch["actions_onehot"]: (batch_size, seq_len, n_agents, n_actions)

    obs = batch["obs"][:,  :-1]  # Remove last timestep
    actions = batch["actions_onehot"][:, :-1]  # Remove last timestep

    batch_size, seq_len, n_agents, _ = obs.shape

    # Reshape for encoding
    obs_flat = obs.reshape(batch_size * seq_len * n_agents, -1)
    actions_flat = actions.reshape(batch_size * seq_len * n_agents, -1)

    with torch.no_grad():
        # Get embeddings: (batch_size * seq_len * n_agents, embedding_dim)
        embeddings = style_encoder(obs_flat, actions_flat)

        # Reshape back
        embeddings = embeddings.reshape(batch_size, seq_len, n_agents, -1)

        # Compute dot product with reference embedding
        # Average over agents to get per-timestep reward
        z_ref = z_ref.to(device)
        rewards = torch.einsum('bsae,e->bs', embeddings, z_ref)  # (batch_size, seq_len)
        rewards = rewards / n_agents  # Average over agents
        rewards = rewards.unsqueeze(-1)  # (batch_size, seq_len, 1)

    return rewards


def compute_exploration_reward(batch, style_encoder, prototype_manager, prototype_id,
                                 device='cpu', k=5):
    """
    Compute k-NN based exploration reward: r_explore = ||φ(s',a') - NN_k(φ(s',a'), H_k)||_2

    Encourages visiting novel state-action regions within the prototype's style space.

    Args:
        batch: EpisodeBatch containing states and actions
        style_encoder: Trained StyleEncoder network
        prototype_manager: PrototypeManager instance
        prototype_id: ID of the current prototype
        device: torch device
        k: k-th nearest neighbor to use
    Returns:
        rewards: (batch_size, seq_len, 1) exploration rewards
    """
    style_encoder.eval()

    # Extract observations and actions (use next state s')
    obs = batch["obs"][:, 1:]  # Use next states (s')
    actions_onehot = batch["actions_onehot"][:, 1:]  # Use actions in next states (a')

    batch_size, seq_len, n_agents, _ = obs.shape

    # Reshape for encoding
    obs_flat = obs.reshape(batch_size * seq_len * n_agents, -1)
    actions_flat = actions_onehot.reshape(batch_size * seq_len * n_agents, -1)

    rewards = []

    with torch.no_grad():
        # Get embeddings
        embeddings = style_encoder(obs_flat, actions_flat)
        embeddings = embeddings.reshape(batch_size, seq_len, n_agents, -1)

        # Average over agents for each timestep
        embeddings_avg = embeddings.mean(dim=2)  # (batch_size, seq_len, embedding_dim)

        # Compute k-NN distance for each timestep
        for b in range(batch_size):
            batch_rewards = []
            for t in range(seq_len):
                embedding_t = embeddings_avg[b, t]  # (embedding_dim,)

                # Compute k-NN exploration reward
                reward = prototype_manager.compute_knn_exploration_reward(
                    prototype_id, embedding_t, k=k
                )
                batch_rewards.append(reward)

                # Add to history buffer (use original state-action, not next)
                if t > 0:  # Skip first timestep as we don't have previous state
                    obs_t = batch["obs"][b, t]  # Original state at time t
                    action_t = batch["actions_onehot"][b, t]
                    # Average over agents
                    obs_t_avg = obs_t.mean(dim=0) if len(obs_t.shape) > 1 else obs_t
                    action_t_avg = action_t.mean(dim=0) if len(action_t.shape) > 1 else action_t
                    prototype_manager.add_to_history(
                        prototype_id,
                        embeddings_avg[b, t-1],
                        (obs_t_avg.cpu().numpy(), action_t_avg.cpu().numpy())
                    )

            rewards.append(batch_rewards)

    rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)  # (batch_size, seq_len, 1)
    return rewards


def add_shaped_rewards_to_batch(batch, style_encoder, prototype_manager, prototype_id,
                                  z_ref, alpha_ref=1.0, alpha_explore=0.1, device='cpu'):
    """
    Add style control and exploration rewards to the batch.

    Modified reward: r' = r + alpha_ref * r_ref + alpha_explore * r_explore

    Args:
        batch: EpisodeBatch to modify (in-place)
        style_encoder: Trained StyleEncoder
        prototype_manager: PrototypeManager instance
        prototype_id: Current prototype ID
        z_ref: Reference style embedding
        alpha_ref: Weight for style control reward
        alpha_explore: Weight for exploration reward
        device: torch device
    Returns:
        batch: Modified batch with shaped rewards
        r_ref_mean: Average style control reward (for logging)
        r_explore_mean: Average exploration reward (for logging)
    """
    # Get original rewards
    original_rewards = batch["reward"][:, :-1].clone()  # (batch_size, seq_len, 1)

    # Compute shaped rewards
    if z_ref is not None:
        r_ref = compute_style_control_reward(batch, style_encoder, z_ref, device)
        r_ref_mean = r_ref.mean().item()
    else:
        r_ref = torch.zeros_like(original_rewards)
        r_ref_mean = 0.0

    r_explore = compute_exploration_reward(batch, style_encoder, prototype_manager,
                                            prototype_id, device)
    r_explore_mean = r_explore.mean().item()

    # Add shaped rewards
    shaped_rewards = original_rewards + alpha_ref * r_ref + alpha_explore * r_explore

    # Update batch rewards (in-place)
    batch["reward"][:, :-1] = shaped_rewards

    return batch, r_ref_mean, r_explore_mean


def collect_trajectories_for_embedding_training(runner, mac_list, n_episodes=10):
    """
    Collect diverse trajectories for training the style embedding function φ.

    Args:
        runner: Episode runner
        mac_list: List of MAC controllers (teammates)
        n_episodes: Number of episodes per MAC
    Returns:
        trajectories: List of (states, actions) tuples for contrastive learning
    """
    trajectories = []

    for mac in mac_list:
        for ep_idx in range(n_episodes):
            # Use train mode to avoid test stats accumulation that triggers early return
            # test_mode=True would accumulate test_returns and return mean after test_nepisode
            episode_batch = runner.run(
                mac1=mac,
                mac2=None,  # Self-play
                test_mode=False,  # Use train mode to avoid test episode counter
                test_mode_1=True,  # But still use greedy action selection for mac1
                test_mode_2=True,  # And greedy for mac2 (None in this case)
                tm_id=-1
            )

            # Check if we got a batch (not a scalar from test mode averaging)
            if not isinstance(episode_batch, (int, float)):
                # Extract states and actions
                obs = episode_batch["obs"][:, :-1]  # (1, seq_len, n_agents, obs_dim)
                actions_onehot = episode_batch["actions_onehot"][:, :-1]  # (1, seq_len, n_agents, n_actions)

                # Squeeze batch dimension and average over agents
                obs = obs.squeeze(0).mean(dim=1).cpu().numpy()  # (seq_len, obs_dim)
                actions = actions_onehot.squeeze(0).mean(dim=1).cpu().numpy()  # (seq_len, n_actions)

                trajectories.append((obs, actions))

        # Reset statistics after each MAC to keep runner clean
        if hasattr(runner, 'test_returns'):
            runner.test_returns = []
            runner.test_stats = {}
        if hasattr(runner, 'train_returns'):
            runner.train_returns = []
            runner.train_stats = {}

    return trajectories


def compute_trajectory_style_embedding(batch, style_encoder, device='cpu'):
    """
    Compute the style embedding z_tau for a complete trajectory.

    z_tau = N(1/T * Σ_t φ(s_t, a_t))

    Args:
        batch: EpisodeBatch
        style_encoder: Trained StyleEncoder
        device: torch device
    Returns:
        z_tau: (embedding_dim,) normalized trajectory embedding
    """
    style_encoder.eval()

    with torch.no_grad():
        obs = batch["obs"][:, :-1]  # (batch_size, seq_len, n_agents, obs_dim)
        actions = batch["actions_onehot"][:, :-1]

        batch_size, seq_len, n_agents, _ = obs.shape

        # Reshape
        obs_flat = obs.reshape(batch_size * seq_len * n_agents, -1)
        actions_flat = actions.reshape(batch_size * seq_len * n_agents, -1)

        # Get embeddings
        embeddings = style_encoder(obs_flat, actions_flat)
        embeddings = embeddings.reshape(batch_size, seq_len, n_agents, -1)

        # Temporal and agent averaging
        z_tau = embeddings.mean(dim=(0, 1, 2))  # Average over batch, time, agents

        # Normalize
        z_tau = F.normalize(z_tau, p=2, dim=0)

    return z_tau
