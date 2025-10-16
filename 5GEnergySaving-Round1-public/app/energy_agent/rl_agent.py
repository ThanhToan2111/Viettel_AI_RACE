## rl_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
import os
from datetime import datetime

from .transition import Transition, TransitionBuffer
from .models import Actor
from .models import Critic
from .state_normalizer import StateNormalizer


class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
        """
        Initialize PPO agent for 5G energy saving

        Args:
            n_cells (int): Number of cells to control
            n_ues (int): Number of UEs in network
            max_time (int): Maximum simulation time steps
            log_file (str): Path to log file
            use_gpu (bool): Whether to use GPU acceleration
        """
        print("Initializing RL Agent")
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # State dimensions: 17 simulation features + 14 network features + (n_cells * 12) cell features
        self.state_dim = 17 + 14 + (n_cells * 12)
        self.action_dim = n_cells  # Power ratio for each cell

        # Normalization parameters - learned from data
        self.state_normalizer = StateNormalizer(self.state_dim, n_cells=n_cells)

        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=256).to(self.device)
        self.critic = Critic(self.state_dim, hidden_dim=256).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, eps=1e-5)

        # PPO hyperparameters (optimized for energy saving task)
        self.gamma = 0.99  # Discount factor
        self.lambda_gae = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clipping parameter
        self.ppo_epochs = 10  # Number of PPO update epochs
        self.batch_size = 64
        self.buffer_size = 2048
        self.entropy_coef = 0.01  # Entropy coefficient for exploration
        self.value_loss_coef = 0.5  # Value loss coefficient

        # Experience buffer
        self.buffer = TransitionBuffer(self.buffer_size)

        self.training_mode = True
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0

        # E_opt values from opts.txt (optimal energy consumption)
        # These are reference values when all cells operate at minimum power
        self.e_opts = {
            12: 0.181,   # indoor_hotspot (12 cells)
            21: None,    # 21 cells could be dense_urban (1.708) or urban_macro (1.929)
            57: 5.799    # rural (57 cells)
        }
        # For 21 cells, we'll detect scenario dynamically based on other features
        self.e_opt_21_dense = 1.708    # Dense Urban
        self.e_opt_21_urban = 1.929    # Urban Macro

        # Load E_opt from file if available
        self._load_e_opts()

        self.setup_logging(log_file)

        self.logger.info(f"PPO Agent initialized: {n_cells} cells, {n_ues} UEs")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"E_opt loaded: {self.e_opts}")
    
    def _load_e_opts(self):
        """Load E_opt values from opts.txt file"""
        try:
            # Try to find opts.txt in common locations
            possible_paths = [
                'opts.txt',
                '../opts.txt',
                '../../opts.txt',
                '/media/tan/F/Viettel_AI_Challenge/Viettel_AI_RACE/5GEnergySaving-Round1-public/opts.txt'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                parts = line.split()
                                if len(parts) == 2:
                                    scenario, e_opt = parts[0], float(parts[1])
                                    if scenario == 'indoor_hotspot':
                                        self.e_opts[12] = e_opt
                                    elif scenario == 'dense_urban':
                                        self.e_opt_21_dense = e_opt
                                    elif scenario == 'urban_macro':
                                        self.e_opt_21_urban = e_opt
                                    elif scenario == 'rural':
                                        self.e_opts[57] = e_opt
                    print(f"E_opt values loaded from {path}")
                    return
            print("Warning: opts.txt not found, using default E_opt values")
        except Exception as e:
            print(f"Warning: Could not load E_opt values: {e}")

    def get_e_opt(self, state):
        """
        Get E_opt value for current scenario based on state

        Args:
            state: Current state vector

        Returns:
            e_opt: Optimal energy consumption for this scenario (kWh)
        """
        state = np.array(state).flatten()

        # Determine number of cells
        n_cells = int(state[0]) if len(state) > 0 else self.n_cells

        # For 12 and 57 cells, E_opt is unambiguous
        if n_cells in self.e_opts and self.e_opts[n_cells] is not None:
            return self.e_opts[n_cells]

        # For 21 cells, distinguish between Dense Urban and Urban Macro
        if n_cells == 21:
            # Use n_ues to distinguish:
            # Dense Urban: 300 UEs, Urban Macro: 210 UEs
            n_ues = int(state[1]) if len(state) > 1 else self.n_ues
            if n_ues > 250:  # Dense Urban
                return self.e_opt_21_dense
            else:  # Urban Macro
                return self.e_opt_21_urban

        # Fallback: use current n_cells setting
        if self.n_cells == 12:
            return 0.181
        elif self.n_cells == 21:
            return self.e_opt_21_urban  # Default to urban_macro (harder scenario)
        elif self.n_cells == 57:
            return 5.799
        else:
            return 1.0  # Fallback value

    def normalize_state(self, state):
        """Normalize state vector to [0, 1] range"""
        return self.state_normalizer.normalize(state)

    def setup_logging(self, log_file):
        """Setup logging configuration"""
        self.logger = logging.getLogger('PPOAgent')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_scenario(self):
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.logger.info(f"Starting episode {self.total_episodes}")
    
    def end_scenario(self):
        self.episode_rewards.append(self.current_episode_reward)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        self.logger.info(f"Episode {self.total_episodes} ended: "
                        f"Steps={self.episode_steps}, "
                        f"Reward={self.current_episode_reward:.2f}, "
                        f"Avg100={avg_reward:.2f}")
        
        # Train if buffer has enough experiences
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()
    
    def get_baseline_action(self, state):
        """
        Conservative baseline policy for initialization
        Dynamically adjusts power based on load and QoS metrics

        Args:
            state: State vector

        Returns:
            action: Power ratios for each cell [0, 1]
        """
        state = np.array(state).flatten()

        # Extract key metrics
        network_start = 17
        avg_drop_rate = state[network_start + 2]
        avg_latency = state[network_start + 3]
        max_cpu = state[network_start + 8]
        max_prb = state[network_start + 9]

        # Extract thresholds
        drop_threshold = state[11]
        latency_threshold = state[12]
        cpu_threshold = state[13]
        prb_threshold = state[14]

        # Determine number of cells
        start_idx = 17 + 12
        remaining_features = len(state) - start_idx
        n_cells = remaining_features // 12 if remaining_features > 0 else 1

        # Base power ratio (conservative - start higher to avoid violations)
        base_power = 0.7

        # Adaptive adjustment based on QoS metrics
        if avg_drop_rate > drop_threshold * 0.8 or avg_latency > latency_threshold * 0.8:
            # QoS at risk - increase power
            base_power = 0.85
        elif max_cpu > cpu_threshold * 0.9 or max_prb > prb_threshold * 0.9:
            # Resource constraints - increase power
            base_power = 0.80
        elif avg_drop_rate < drop_threshold * 0.5 and avg_latency < latency_threshold * 0.7:
            # QoS is good - can reduce power for energy saving
            base_power = 0.6

        # Create action array with slight per-cell variation for diversity
        actions = np.ones(n_cells) * base_power
        actions += np.random.uniform(-0.05, 0.05, n_cells)  # Small noise

        # Clamp to valid range
        actions = np.clip(actions, 0.0, 1.0)

        return actions

    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def get_action(self, state):
        """
        Get action from policy network

        Args:
            state: State vector from MATLAB interface

        Returns:
            action: Power ratios for each cell [0, 1]
        """
        # Use baseline policy for first few episodes or if actor not trained
        if self.total_episodes < 3 and self.training_mode:
            action = self.get_baseline_action(state)

            # Store for potential experience replay
            state_normalized = self.normalize_state(np.array(state).flatten())
            self.last_state = state_normalized
            self.last_action = action
            self.last_log_prob = np.zeros(1)

            return action

        state = self.normalize_state(np.array(state).flatten())  # make sure it's 1D
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, action_logstd = self.actor(state_tensor)

            if self.training_mode:
                # Sample from policy during training
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                # Use mean during evaluation
                action = action_mean
                log_prob = torch.zeros(1).to(self.device)

        # Clamp actions to [0, 1] range
        action = torch.clamp(action, 0.0, 1.0)

        # Store for experience replay
        self.last_state = state_tensor.cpu().numpy().flatten()
        self.last_action = action.cpu().numpy().flatten()
        self.last_log_prob = log_prob.cpu().numpy().flatten()

        return action.cpu().numpy().flatten()
    
    ## OPTIONAL: Modify reward calculation as needed
    def calculate_reward(self, prev_state, action, current_state):
        """
        IMPROVED: Calculate reward with E_opt awareness for better score optimization

        Reward structure:
        1. E_opt-aware energy reward: Guide agent to approach E_opt
        2. QoS constraint penalties: CRITICAL - zero tolerance for violations
        3. Energy efficiency bonus: Reward being close to E_opt
        4. Operational stability: Smooth transitions and load-awareness
        """
        if prev_state is None:
            return 0.0

        # Convert to numpy arrays for consistent indexing
        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()
        action = np.array(action).flatten()

        # Extract state components (17 simulation + 12 network features)
        network_start = 17  # After simulation features

        # Current state metrics - Network features
        current_energy = current_state[network_start + 0]  # totalEnergy (kWh accumulated)
        active_cells = current_state[network_start + 1]     # activeCells
        avg_drop_rate = current_state[network_start + 2]    # avgDropRate
        avg_latency = current_state[network_start + 3]      # avgLatency
        total_traffic = current_state[network_start + 4]    # totalTraffic
        connected_ues = current_state[network_start + 5]    # connectedUEs
        cpu_violations = current_state[network_start + 6]   # cpuViolations
        prb_violations = current_state[network_start + 7]   # prbViolations
        max_cpu = current_state[network_start + 8]          # maxCpuUsage
        max_prb = current_state[network_start + 9]          # maxPrbUsage

        # Extract thresholds from simulation features
        drop_threshold = current_state[11]    # dropCallThreshold
        latency_threshold = current_state[12] # latencyThreshold
        cpu_threshold = current_state[13]     # cpuThreshold
        prb_threshold = current_state[14]     # prbThreshold

        # Previous state metrics
        prev_energy = prev_state[network_start + 0]
        prev_drop_rate = prev_state[network_start + 2]
        prev_connected_ues = prev_state[network_start + 5]

        # Get E_opt for current scenario
        e_opt = self.get_e_opt(current_state)

        # Current simulation progress
        current_time = current_state[2]  # simTime
        max_time = current_state[3]      # maxSimTime
        progress = current_time / max(max_time, 1)

        # ========== 1. E_OPT-AWARE ENERGY REWARD ==========
        # Calculate instantaneous energy consumption (delta)
        energy_delta = current_energy - prev_energy
        energy_delta = max(energy_delta, 0.0001)  # Avoid division by zero

        # Project final energy based on current rate
        if progress > 0.1:  # After 10% of simulation
            projected_final_energy = current_energy / progress
        else:
            projected_final_energy = current_energy * 10  # Rough estimate

        # Calculate how close we are to E_opt (MAPE-like metric)
        # Target: Get as close to E_opt as possible
        energy_ratio = projected_final_energy / e_opt

        # Reward structure based on energy_ratio:
        # - ratio < 1.0: Below E_opt (impossible without QoS violations, but reward if achieved)
        # - ratio 1.0-1.2: Excellent (close to optimal)
        # - ratio 1.2-1.5: Good (acceptable range)
        # - ratio 1.5-2.0: Moderate (needs improvement)
        # - ratio > 2.0: Poor (significant penalty)

        if energy_ratio < 1.2:
            # Excellent zone: Strong positive reward
            energy_reward = 20.0 * (1.2 - energy_ratio)  # Max +20 when ratio=1.0
        elif energy_ratio < 1.5:
            # Good zone: Mild positive reward
            energy_reward = 10.0 * (1.5 - energy_ratio)  # +0 to +3
        elif energy_ratio < 2.0:
            # Moderate zone: Small negative
            energy_reward = -5.0 * (energy_ratio - 1.5)  # 0 to -2.5
        else:
            # Poor zone: Strong negative
            energy_reward = -10.0 * (energy_ratio - 1.5)  # < -5

        # Immediate energy penalty (encourage reducing power at each step)
        energy_reward -= energy_delta * 2.0

        # ========== 2. QoS CONSTRAINT PENALTIES (CRITICAL - NO COMPROMISE) ==========
        constraint_penalty = 0.0
        qos_violation = False

        # Drop rate constraint - EXTREMELY STRICT
        if avg_drop_rate > drop_threshold:
            constraint_penalty -= 100.0 * (avg_drop_rate - drop_threshold)
            qos_violation = True
        elif avg_drop_rate > drop_threshold * 0.9:  # Danger zone
            constraint_penalty -= 20.0 * (avg_drop_rate - drop_threshold * 0.9)
        elif avg_drop_rate > drop_threshold * 0.7:  # Warning zone
            constraint_penalty -= 5.0 * (avg_drop_rate - drop_threshold * 0.7)

        # Latency constraint - VERY STRICT
        if avg_latency > latency_threshold:
            constraint_penalty -= 50.0 * (avg_latency - latency_threshold)
            qos_violation = True
        elif avg_latency > latency_threshold * 0.95:  # Danger zone
            constraint_penalty -= 10.0 * (avg_latency - latency_threshold * 0.95)
        elif avg_latency > latency_threshold * 0.8:  # Warning zone
            constraint_penalty -= 3.0 * (avg_latency - latency_threshold * 0.8)

        # CPU constraint - STRICT
        if max_cpu > cpu_threshold:
            constraint_penalty -= 40.0 * (max_cpu - cpu_threshold)
            qos_violation = True
        elif max_cpu > cpu_threshold * 0.98:  # Danger zone
            constraint_penalty -= 8.0 * (max_cpu - cpu_threshold * 0.98)

        # PRB constraint - STRICT
        if max_prb > prb_threshold:
            constraint_penalty -= 40.0 * (max_prb - prb_threshold)
            qos_violation = True
        elif max_prb > prb_threshold * 0.98:  # Danger zone
            constraint_penalty -= 8.0 * (max_prb - prb_threshold * 0.98)

        # ========== 3. E_OPT PROXIMITY BONUS ==========
        # Extra reward for maintaining good energy-QoS balance
        e_opt_bonus = 0.0

        if not qos_violation and energy_ratio < 1.3:
            # Excellent: Close to E_opt without QoS violations
            e_opt_bonus = 15.0 * (1.3 - energy_ratio)
        elif not qos_violation and energy_ratio < 1.5:
            # Good: Reasonable energy with QoS compliance
            e_opt_bonus = 5.0 * (1.5 - energy_ratio)

        # ========== 4. OPERATIONAL STABILITY REWARDS ==========
        stability_reward = 0.0

        # Reward maintaining service quality
        if connected_ues >= prev_connected_ues:
            stability_reward += 0.5
        else:
            # Penalty for losing UEs (might indicate coverage issues)
            stability_reward -= (prev_connected_ues - connected_ues) * 0.5

        # Reward improving drop rate
        if avg_drop_rate < prev_drop_rate and prev_drop_rate > 0:
            stability_reward += (prev_drop_rate - avg_drop_rate) * 3.0

        # Load-adaptive penalty: Encourage matching power to load
        avg_prb = current_state[network_start + 7] if network_start + 7 < len(current_state) else 0.5
        avg_action = np.mean(action)
        load_mismatch = abs(avg_prb - avg_action)
        if load_mismatch > 0.3:  # Power and load significantly mismatched
            stability_reward -= load_mismatch * 3.0

        # Penalty for extreme action variance (all cells should be somewhat coordinated)
        action_variance = np.std(action)
        if action_variance > 0.35:  # Too much variation
            stability_reward -= action_variance * 5.0

        # Bonus for keeping most cells active (avoid coverage holes)
        total_cells = current_state[0]  # totalCells from simulation features
        active_ratio = active_cells / max(total_cells, 1)
        if active_ratio >= 0.8:  # At least 80% cells active
            stability_reward += 2.0
        elif active_ratio < 0.5:  # Too many cells off
            stability_reward -= 5.0 * (0.5 - active_ratio)

        # ========== 5. TOTAL REWARD ==========
        total_reward = (
            energy_reward +
            constraint_penalty +
            e_opt_bonus +
            stability_reward
        )

        # Clip to reasonable range (wider range to allow strong penalties)
        total_reward = float(np.clip(total_reward, -500, 100))

        # Log reward components for debugging (periodically)
        if self.total_steps % 100 == 0:
            self.logger.info(
                f"Reward breakdown - Energy: {energy_reward:.2f}, "
                f"QoS_penalty: {constraint_penalty:.2f}, "
                f"E_opt_bonus: {e_opt_bonus:.2f}, "
                f"Stability: {stability_reward:.2f}, "
                f"Total: {total_reward:.2f}, "
                f"E_ratio: {energy_ratio:.3f}, "
                f"E_projected: {projected_final_energy:.3f}, "
                f"E_opt: {e_opt:.3f}"
            )

        return total_reward
    
    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def update(self, state, action, next_state, done):
        """
        Update agent with experience
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Next state
            done: Whether episode is done
        """
        if not self.training_mode:
            return
        
        # Calculate actual reward using state as prev_state and next_state as current
        actual_reward = self.calculate_reward(state, action, next_state)

        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += actual_reward
        
        # Convert inputs to numpy if needed
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()
        
        # Ensure proper shapes
        state = self.normalize_state(np.array(state).flatten())
        action = np.array(action).flatten()
        next_state = self.normalize_state(np.array(next_state).flatten())
        
        # Get value estimates
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor).cpu().numpy().flatten()[0]
            next_value = self.critic(next_state_tensor).cpu().numpy().flatten()[0]
        
        # Create transition
        transition = Transition(
            state=state,
            action=action,
            reward=actual_reward,
            next_state=next_state,
            done=done,
            log_prob=getattr(self, 'last_log_prob', np.array([0.0]))[0],
            value=value
        )
        
        self.buffer.add(transition)
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lambda_gae * next_non_terminal * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def train(self):
        """Train the PPO agent"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Get all transitions
        transitions = self.buffer.get_all()
        
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions])
        old_log_probs = np.array([t.log_prob for t in transitions])
        values = np.array([t.value for t in transitions])
        
        # Compute next values
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            next_values = self.critic(next_states_tensor).cpu().numpy().flatten()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO training loop
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Compute current policy
                action_mean, action_logstd = self.actor(batch_states)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)

                # Compute entropy for exploration bonus
                entropy = dist.entropy().sum(-1).mean()

                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Actor loss with entropy bonus
                actor_loss = policy_loss - self.entropy_coef * entropy

                # Critic loss
                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)

                # Total loss
                total_loss = actor_loss + self.value_loss_coef * critic_loss

                # Update both networks together
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # Clear buffer after training
        self.buffer.clear()
        
        self.logger.info(f"Training completed: Actor loss={actor_loss:.4f}, "
                        f"Critic loss={critic_loss:.4f}")
    
    def save_model(self, filepath=None):
        """Save model parameters"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_model_{timestamp}.pth"
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training):
        """Set training mode"""
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)
        self.logger.info(f"Training mode set to {training}")
    
    def get_stats(self):
        """Get training statistics"""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_reward': avg_reward,
            'buffer_size': len(self.buffer),
            'training_mode': self.training_mode,
            'episode_steps': self.episode_steps,
            'current_episode_reward': self.current_episode_reward
        }