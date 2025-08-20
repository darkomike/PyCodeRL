#!/usr/bin/env python3
"""
PyCodeRL Training Framework
Trains the RL agents for optimal Python compilation decisions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import logging
from pathlib import Path

from pycode_rl_implementation import (
    PyCodeRLCompiler, CompilationState, Register, 
    InstructionType, RewardCalculator
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingEpisode:
    """Stores data from a single training episode"""
    states: List[torch.Tensor]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    done: bool

class ExperienceBuffer:
    """Buffer for storing training experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        
    def add_episode(self, episode: TrainingEpisode):
        self.episodes.append(episode)
        
    def sample_batch(self, batch_size: int) -> List[TrainingEpisode]:
        return random.sample(list(self.episodes), min(batch_size, len(self.episodes)))
    
    def __len__(self):
        return len(self.episodes)

class PPOTrainer:
    """Proximal Policy Optimization trainer for PyCodeRL agents"""
    
    def __init__(self, 
                 compiler: PyCodeRLCompiler,
                 learning_rate: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        
        self.compiler = compiler
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Optimizers for each agent
        self.instruction_optimizer = optim.Adam(
            compiler.instruction_agent.parameters(), lr=learning_rate
        )
        self.register_optimizer = optim.Adam(
            compiler.register_agent.parameters(), lr=learning_rate
        )
        self.memory_optimizer = optim.Adam(
            compiler.memory_agent.parameters(), lr=learning_rate
        )
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer()
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'compilation_success_rate': [],
            'average_performance': [],
            'policy_loss': [],
            'value_loss': []
        }
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value_step = next_value
            else:
                next_value_step = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value_step - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            
        return advantages, returns
    
    def collect_episode(self, python_code: str, 
                       reference_metrics: Dict[str, float]) -> TrainingEpisode:
        """Collect a single training episode"""
        
        try:
            # Parse and compile
            ast_tree = self.compiler.parse_python_code(python_code)
            semantic_info = self.compiler.semantic_analyzer.analyze(ast_tree)
            
            # Initialize episode data
            episode = TrainingEpisode([], [], [], [], [], False)
            
            # Compilation state
            state = CompilationState(
                ast_tree, list(Register), 0, {}, 0,
                {'semantic_info': semantic_info}
            )
            
            # Compile with RL decisions
            total_reward = 0
            instructions = []
            
            for node in ast_tree.body:
                # Encode current state
                state_encoding = self.compiler.state_encoder(state)
                episode.states.append(state_encoding.detach())
                
                # Get agent decisions
                action_probs, state_value = self.compiler.instruction_agent(state_encoding)
                register_probs = self.compiler.register_agent(state_encoding)
                memory_probs = self.compiler.memory_agent(state_encoding)
                
                # Sample actions
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                register_dist = torch.distributions.Categorical(register_probs)
                register_action = register_dist.sample()
                
                # Store episode data
                episode.actions.append(action.item())
                episode.values.append(state_value.item())
                episode.log_probs.append(action_dist.log_prob(action).item())
                
                # Compile current node
                node_instructions = self.compiler.compile_ast_node(node, state)
                instructions.extend(node_instructions)
                
                # Calculate immediate reward (simplified)
                immediate_reward = self._calculate_immediate_reward(
                    node, node_instructions, state
                )
                episode.rewards.append(immediate_reward)
                total_reward += immediate_reward
            
            # Execute and get final reward
            assembly = self.compiler.generate_assembly(instructions)
            exec_metrics = self.compiler.execute_and_evaluate(assembly)
            
            # Calculate final reward
            final_reward = self.compiler.reward_calculator.calculate_reward(
                exec_metrics['execution_time'],
                exec_metrics['correctness_score'],
                0.1,  # compilation_time (placeholder)
                len(instructions),
                reference_metrics
            )
            
            # Update episode with final reward
            if episode.rewards:
                episode.rewards[-1] += final_reward
            
            episode.done = True
            return episode
            
        except Exception as e:
            logger.error(f"Episode collection failed: {e}")
            # Return empty episode
            return TrainingEpisode([], [], [], [], [], True)
    
    def _calculate_immediate_reward(self, node, instructions, state) -> float:
        """Calculate immediate reward for a compilation decision"""
        base_reward = 0.1  # Base reward for successful compilation
        
        # Bonus for efficient instruction selection
        if len(instructions) <= 3:  # Efficient compilation
            base_reward += 0.2
        
        # Bonus for register efficiency
        if len(state.available_registers) > 8:  # Good register usage
            base_reward += 0.1
            
        # Penalty for complex constructs without optimization
        if isinstance(node, (ast.For, ast.While)) and len(instructions) > 10:
            base_reward -= 0.1
            
        return base_reward
    
    def update_agents(self, episodes: List[TrainingEpisode]):
        """Update agent networks using PPO"""
        
        if not episodes:
            return
        
        # Combine all episode data
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        for episode in episodes:
            if not episode.states:
                continue
                
            # Compute advantages and returns
            advantages, returns = self.compute_gae(
                episode.rewards, episode.values
            )
            
            all_states.extend(episode.states)
            all_actions.extend(episode.actions)
            all_old_log_probs.extend(episode.log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        if not all_states:
            return
        
        # Convert to tensors
        states_tensor = torch.stack(all_states)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long)
        old_log_probs_tensor = torch.tensor(all_old_log_probs)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )
        
        # PPO update loop
        for _ in range(4):  # Multiple epochs
            # Forward pass
            action_probs, state_values = self.compiler.instruction_agent(states_tensor)
            
            # Calculate new log probabilities
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions_tensor)
            
            # Calculate probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # Calculate policy loss
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = nn.MSELoss()(state_values.squeeze(), returns_tensor)
            
            # Calculate entropy bonus
            entropy = action_dist.entropy().mean()
            
            # Total loss
            total_loss = (
                policy_loss + 
                self.value_coef * value_loss - 
                self.entropy_coef * entropy
            )
            
            # Update instruction agent
            self.instruction_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.compiler.instruction_agent.parameters(), 0.5)
            self.instruction_optimizer.step()
            
            # Store metrics
            self.training_metrics['policy_loss'].append(policy_loss.item())
            self.training_metrics['value_loss'].append(value_loss.item())
    
    def train(self, training_programs: List[str], 
              num_episodes: int = 1000,
              batch_size: int = 32,
              eval_interval: int = 100):
        """Main training loop"""
        
        logger.info(f"Starting training for {num_episodes} episodes")
        
        # Reference metrics for reward calculation
        reference_metrics = {
            'execution_time': 0.1,
            'compilation_time': 0.05,
            'code_size': 100
        }
        
        for episode in range(num_episodes):
            # Select random training program
            python_code = random.choice(training_programs)
            
            # Collect episode
            episode_data = self.collect_episode(python_code, reference_metrics)
            
            if episode_data.states:  # Valid episode
                self.experience_buffer.add_episode(episode_data)
                
                episode_reward = sum(episode_data.rewards)
                self.training_metrics['episode_rewards'].append(episode_reward)
                
                # Update agents periodically
                if len(self.experience_buffer) >= batch_size and episode % 10 == 0:
                    batch_episodes = self.experience_buffer.sample_batch(batch_size)
                    self.update_agents(batch_episodes)
            
            # Evaluation and logging
            if episode % eval_interval == 0:
                self._evaluate_and_log(episode, training_programs[:5])
                
        logger.info("Training completed")
    
    def _evaluate_and_log(self, episode: int, eval_programs: List[str]):
        """Evaluate current performance and log metrics"""
        
        logger.info(f"Episode {episode} - Evaluation")
        
        total_rewards = []
        success_count = 0
        
        for program in eval_programs:
            try:
                assembly, metrics = self.compiler.compile_to_machine_code(program)
                exec_metrics = self.compiler.execute_and_evaluate(assembly)
                
                if exec_metrics['correctness_score'] > 0.5:
                    success_count += 1
                
                # Calculate reward
                reward = self.compiler.reward_calculator.calculate_reward(
                    exec_metrics['execution_time'],
                    exec_metrics['correctness_score'],
                    metrics['compilation_time'],
                    metrics['instruction_count'],
                    {'execution_time': 0.1, 'compilation_time': 0.05, 'code_size': 100}
                )
                total_rewards.append(reward)
                
            except Exception as e:
                logger.warning(f"Evaluation failed for program: {e}")
                total_rewards.append(0.0)
        
        # Store metrics
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        success_rate = success_count / len(eval_programs)
        
        self.training_metrics['average_performance'].append(avg_reward)
        self.training_metrics['compilation_success_rate'].append(success_rate)
        
        logger.info(f"  Average reward: {avg_reward:.3f}")
        logger.info(f"  Success rate: {success_rate:.3f}")
        
        # Save model checkpoint
        if episode % 500 == 0:
            self.save_checkpoint(f"checkpoint_episode_{episode}.pt")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'instruction_agent': self.compiler.instruction_agent.state_dict(),
            'register_agent': self.compiler.register_agent.state_dict(),
            'memory_agent': self.compiler.memory_agent.state_dict(),
            'state_encoder': self.compiler.state_encoder.state_dict(),
            'training_metrics': self.training_metrics
        }
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        
        self.compiler.instruction_agent.load_state_dict(checkpoint['instruction_agent'])
        self.compiler.register_agent.load_state_dict(checkpoint['register_agent'])
        self.compiler.memory_agent.load_state_dict(checkpoint['memory_agent'])
        self.compiler.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.training_metrics = checkpoint['training_metrics']
        
        logger.info(f"Checkpoint loaded: {filename}")
    
    def plot_training_metrics(self):
        """Plot training progress"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        if self.training_metrics['episode_rewards']:
            axes[0, 0].plot(self.training_metrics['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
        
        # Success rate
        if self.training_metrics['compilation_success_rate']:
            axes[0, 1].plot(self.training_metrics['compilation_success_rate'])
            axes[0, 1].set_title('Compilation Success Rate')
            axes[0, 1].set_xlabel('Evaluation')
            axes[0, 1].set_ylabel('Success Rate')
        
        # Policy loss
        if self.training_metrics['policy_loss']:
            axes[1, 0].plot(self.training_metrics['policy_loss'])
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
        
        # Average performance
        if self.training_metrics['average_performance']:
            axes[1, 1].plot(self.training_metrics['average_performance'])
            axes[1, 1].set_title('Average Performance')
            axes[1, 1].set_xlabel('Evaluation')
            axes[1, 1].set_ylabel('Performance')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()

class PythonProgramGenerator:
    """Generate diverse Python programs for training"""
    
    def __init__(self):
        self.templates = [
            # Simple arithmetic
            """
def compute():
    a = {a}
    b = {b}
    result = a {op} b
    return result
""",
            # Control flow
            """
def conditional(x):
    if x > {threshold}:
        return x * 2
    else:
        return x + 1
""",
            # Loops
            """
def loop_sum(n):
    total = 0
    for i in range(n):
        total = total + i
    return total
""",
            # Function calls
            """
def helper(x):
    return x * 2

def main():
    result = helper({value})
    return result
"""
        ]
    
    def generate_program(self) -> str:
        """Generate a random Python program"""
        template = random.choice(self.templates)
        
        # Fill template parameters
        params = {
            'a': random.randint(1, 100),
            'b': random.randint(1, 100),
            'op': random.choice(['+', '-', '*']),
            'threshold': random.randint(1, 50),
            'value': random.randint(1, 20)
        }
        
        return template.format(**params)
    
    def generate_dataset(self, size: int) -> List[str]:
        """Generate a dataset of Python programs"""
        return [self.generate_program() for _ in range(size)]

def main():
    """Main training script"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize compiler and trainer
    compiler = PyCodeRLCompiler()
    trainer = PPOTrainer(compiler)
    
    # Generate training dataset
    program_generator = PythonProgramGenerator()
    training_programs = program_generator.generate_dataset(1000)
    
    print("PyCodeRL Training Framework")
    print("=" * 40)
    print(f"Generated {len(training_programs)} training programs")
    
    # Example training programs
    example_programs = [
        """
def add(a, b):
    return a + b

result = add(5, 3)
""",
        """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
""",
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(8)
""",
        """
def sum_list(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total

result = sum_list([1, 2, 3, 4, 5])
"""
    ]
    
    # Add example programs to training set
    training_programs.extend(example_programs)
    
    print(f"Total training programs: {len(training_programs)}")
    
    # Start training
    try:
        trainer.train(
            training_programs=training_programs,
            num_episodes=2000,
            batch_size=16,
            eval_interval=50
        )
        
        # Plot results
        trainer.plot_training_metrics()
        
        # Save final model
        trainer.save_checkpoint("final_model.pt")
        
        # Test trained model
        print("\nTesting trained model:")
        print("-" * 30)
        
        for i, program in enumerate(example_programs[:3]):
            print(f"\nTest {i+1}:")
            print(program)
            
            try:
                assembly, metrics = compiler.compile_to_machine_code(program)
                exec_metrics = compiler.execute_and_evaluate(assembly)
                
                print(f"Compilation time: {metrics['compilation_time']:.4f}s")
                print(f"Execution time: {exec_metrics['execution_time']:.4f}s") 
                print(f"Correctness: {exec_metrics['correctness_score']:.2f}")
                
            except Exception as e:
                print(f"Failed: {e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted_model.pt")
    
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()