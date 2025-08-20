#!/usr/bin/env python3
"""
PyCodeRL Training Example
"""

from pycode_rl_training import PPOTrainer, PythonProgramGenerator
from pycode_rl_implementation import PyCodeRLCompiler

def main():
    # Initialize compiler and trainer
    compiler = PyCodeRLCompiler()
    trainer = PPOTrainer(compiler)
    
    # Generate training programs
    generator = PythonProgramGenerator()
    training_programs = generator.generate_dataset(100)
    
    print(f"Generated {len(training_programs)} training programs")
    
    # Start training
    trainer.train(
        training_programs=training_programs,
        num_episodes=500,
        batch_size=16,
        eval_interval=50
    )
    
    # Save trained model
    trainer.save_checkpoint("trained_model.pt")
    
    # Plot training progress
    trainer.plot_training_metrics()

if __name__ == "__main__":
    main()
