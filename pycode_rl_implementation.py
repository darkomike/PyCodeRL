#!/usr/bin/env python3
"""
PyCodeRL: Direct Python to ARM64 Machine Code Generator
A reinforcement learning framework for direct Python AST to ARM64 machine code compilation.
"""

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import subprocess
import tempfile
import os
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstructionType(Enum):
    """ARM64 (AArch64) instruction types for our simplified instruction set"""
    MOV = "mov"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    UDIV = "udiv"  # ARM64 uses udiv instead of idiv
    CMP = "cmp"
    B = "b"        # Branch (unconditional jump)
    BEQ = "b.eq"   # Branch if equal
    BNE = "b.ne"   # Branch if not equal
    BLT = "b.lt"   # Branch if less than
    BGT = "b.gt"   # Branch if greater than
    BL = "bl"      # Branch with link (call)
    RET = "ret"
    STP = "stp"    # Store pair (ARM64 push equivalent)
    LDP = "ldp"    # Load pair (ARM64 pop equivalent)
    NOP = "nop"

class Register(Enum):
    """ARM64 (AArch64) registers"""
    X0 = "x0"      # General purpose register 0 (return value, first argument)
    X1 = "x1"      # General purpose register 1 (second argument)
    X2 = "x2"      # General purpose register 2 (third argument)
    X3 = "x3"      # General purpose register 3 (fourth argument)
    X4 = "x4"      # General purpose register 4
    X5 = "x5"      # General purpose register 5
    X6 = "x6"      # General purpose register 6
    X7 = "x7"      # General purpose register 7
    X8 = "x8"      # General purpose register 8
    X9 = "x9"      # General purpose register 9
    X10 = "x10"    # General purpose register 10
    X11 = "x11"    # General purpose register 11
    X12 = "x12"    # General purpose register 12
    X13 = "x13"    # General purpose register 13
    X14 = "x14"    # General purpose register 14
    X15 = "x15"    # General purpose register 15
    X29 = "x29"    # Frame pointer
    X30 = "x30"    # Link register
    SP = "sp"      # Stack pointer

@dataclass
class CompilationState:
    """Current state during compilation"""
    ast_node: ast.AST
    available_registers: List[Register]
    stack_offset: int
    variable_map: Dict[str, str]  # variable name -> register/memory location
    control_flow_depth: int
    optimization_context: Dict[str, Any]

@dataclass 
class ARM64Instruction:
    """Represents a single ARM64 instruction"""
    opcode: InstructionType
    operands: List[str]
    comment: str = ""
    
    def __str__(self) -> str:
        operands_str = ", ".join(self.operands)
        base = f"    {self.opcode.value}"
        if operands_str:
            base += f" {operands_str}"
        if self.comment:
            base += f"  // {self.comment}"
        return base

class PythonSemanticAnalyzer:
    """Analyzes Python AST for semantic information needed for compilation"""
    
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.control_flow_depth = 0
        
    def analyze(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze Python AST and extract compilation-relevant information"""
        analysis = {
            'variables': {},
            'functions': {},
            'complexity': 0,
            'dynamic_features': [],
            'optimization_hints': []
        }
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                analysis['variables'][child.id] = {
                    'type': 'dynamic',  # Python's dynamic typing
                    'usage_count': analysis['variables'].get(child.id, {}).get('usage_count', 0) + 1
                }
            elif isinstance(child, ast.FunctionDef):
                analysis['functions'][child.name] = {
                    'args': [arg.arg for arg in child.args.args],
                    'complexity': len(list(ast.walk(child)))
                }
            elif isinstance(child, (ast.For, ast.While)):
                analysis['complexity'] += 5  # Loops add complexity
            elif isinstance(child, ast.ListComp):
                analysis['dynamic_features'].append('list_comprehension')
                analysis['optimization_hints'].append('vectorizable')
                
        return analysis

class StateEncoder(nn.Module):
    """Neural network to encode compilation state"""
    
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        self.ast_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.context_encoder = nn.Linear(64, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def encode_ast_node(self, node: ast.AST) -> torch.Tensor:
        """Encode AST node into tensor representation"""
        # Simplified AST encoding - in practice would be more sophisticated
        node_features = torch.zeros(1, 1, 256)
        
        if isinstance(node, ast.Assign):
            node_features[0, 0, 0] = 1.0
        elif isinstance(node, ast.BinOp):
            node_features[0, 0, 1] = 1.0
            if isinstance(node.op, ast.Add):
                node_features[0, 0, 10] = 1.0
            elif isinstance(node.op, ast.Sub):
                node_features[0, 0, 11] = 1.0
        elif isinstance(node, ast.Call):
            node_features[0, 0, 2] = 1.0
        elif isinstance(node, ast.Return):
            node_features[0, 0, 3] = 1.0
            
        return node_features
    
    def encode_context(self, state: CompilationState) -> torch.Tensor:
        """Encode compilation context"""
        context_features = torch.zeros(64)
        context_features[0] = len(state.available_registers) / 12.0  # Normalize
        context_features[1] = state.stack_offset / 1000.0  # Normalize
        context_features[2] = state.control_flow_depth / 10.0  # Normalize
        return context_features
    
    def forward(self, state: CompilationState) -> torch.Tensor:
        """Encode complete compilation state"""
        ast_features = self.encode_ast_node(state.ast_node)
        ast_encoded, _ = self.ast_encoder(ast_features)
        
        context_features = self.encode_context(state)
        context_encoded = self.context_encoder(context_features)
        
        fused = torch.cat([ast_encoded.squeeze(), context_encoded], dim=0)
        return self.fusion(fused)

class InstructionSelectionAgent(nn.Module):
    """RL agent for selecting x86 instructions"""
    
    def __init__(self, state_dim=512, action_dim=16):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities and state value"""
        action_logits = self.policy_net(state_encoding)
        state_value = self.value_net(state_encoding)
        return F.softmax(action_logits, dim=-1), state_value

class RegisterAllocationAgent(nn.Module):
    """RL agent for register allocation decisions"""
    
    def __init__(self, state_dim=512, num_registers=12):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, num_registers)
        )
        
    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """Select register for current operation"""
        register_logits = self.policy_net(state_encoding)
        return F.softmax(register_logits, dim=-1)

class MemoryManagementAgent(nn.Module):
    """RL agent for memory layout decisions"""
    
    def __init__(self, state_dim=512, action_dim=8):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """Make memory management decisions"""
        memory_logits = self.policy_net(state_encoding)
        return F.softmax(memory_logits, dim=-1)

class RewardCalculator:
    """Calculate rewards for RL training"""
    
    def __init__(self):
        self.performance_weight = 0.4
        self.correctness_weight = 0.3
        self.compilation_speed_weight = 0.2
        self.code_size_weight = 0.1
        
    def calculate_reward(self, 
                        execution_time: float,
                        correctness_score: float,
                        compilation_time: float,
                        code_size: int,
                        reference_metrics: Dict[str, float]) -> float:
        """Calculate composite reward for compilation decisions"""
        
        # Performance reward (inverse of execution time)
        perf_reward = (reference_metrics['execution_time'] / max(execution_time, 0.001))
        
        # Correctness reward
        correctness_reward = correctness_score
        
        # Compilation speed reward
        compile_reward = (reference_metrics['compilation_time'] / max(compilation_time, 0.001))
        
        # Code size reward (smaller is better)
        size_reward = (reference_metrics['code_size'] / max(code_size, 1))
        
        total_reward = (
            self.performance_weight * perf_reward +
            self.correctness_weight * correctness_reward +
            self.compilation_speed_weight * compile_reward +
            self.code_size_weight * size_reward
        )
        
        return total_reward

class PyCodeRLCompiler:
    """Main PyCodeRL compiler class"""
    
    def __init__(self):
        self.semantic_analyzer = PythonSemanticAnalyzer()
        self.state_encoder = StateEncoder()
        self.instruction_agent = InstructionSelectionAgent()
        self.register_agent = RegisterAllocationAgent()
        self.memory_agent = MemoryManagementAgent()
        self.reward_calculator = RewardCalculator()
        
        # Training mode flag
        self.training_mode = False
        self.episode_data = []
        
    def parse_python_code(self, code: str) -> ast.AST:
        """Parse Python code into AST"""
        try:
            tree = ast.parse(code)
            return tree
        except SyntaxError as e:
            logger.error(f"Python syntax error: {e}")
            raise
    
    def compile_ast_node(self, node: ast.AST, state: CompilationState) -> List[ARM64Instruction]:
        """Compile a single AST node to ARM64 instructions"""
        instructions = []
        
        if isinstance(node, ast.Assign):
            instructions.extend(self._compile_assignment(node, state))
        elif isinstance(node, ast.BinOp):
            instructions.extend(self._compile_binary_operation(node, state))
        elif isinstance(node, ast.Return):
            instructions.extend(self._compile_return(node, state))
        elif isinstance(node, ast.Call):
            instructions.extend(self._compile_function_call(node, state))
        elif isinstance(node, ast.If):
            instructions.extend(self._compile_conditional(node, state))
        elif isinstance(node, ast.For):
            instructions.extend(self._compile_loop(node, state))
        else:
            # Default fallback
            instructions.append(ARM64Instruction(InstructionType.NOP, [], f"Unsupported node: {type(node).__name__}"))
            
        return instructions
    
    def _compile_assignment(self, node: ast.Assign, state: CompilationState) -> List[ARM64Instruction]:
        """Compile Python assignment to ARM64"""
        instructions = []
        
        # Get RL agent decisions
        state_encoding = self.state_encoder(state)
        
        # Select register using RL agent
        register_probs = self.register_agent(state_encoding)
        selected_register_idx = torch.argmax(register_probs).item()
        selected_register = list(Register)[selected_register_idx]
        
        # Simple assignment compilation
        target = node.targets[0]
        if isinstance(target, ast.Name):
            variable_name = target.id
            
            # Compile the value expression
            if isinstance(node.value, ast.Constant):
                # Direct constant assignment using ARM64 syntax
                instructions.append(ARM64Instruction(
                    InstructionType.MOV,
                    [selected_register.value, f"#{node.value.value}"],  # ARM64 immediate syntax
                    f"Assign {node.value.value} to {variable_name}"
                ))
                state.variable_map[variable_name] = selected_register.value
                
        return instructions
    
    def _compile_binary_operation(self, node: ast.BinOp, state: CompilationState) -> List[ARM64Instruction]:
        """Compile binary operations for ARM64"""
        instructions = []
        
        # Get RL decisions
        state_encoding = self.state_encoder(state)
        _, _ = self.instruction_agent(state_encoding)  # Fixed unused variable warning
        
        # Select operation based on AST node type using ARM64 instructions
        if isinstance(node.op, ast.Add):
            instructions.append(ARM64Instruction(InstructionType.ADD, ["x0", "x0", "x1"], "Addition"))
        elif isinstance(node.op, ast.Sub):
            instructions.append(ARM64Instruction(InstructionType.SUB, ["x0", "x0", "x1"], "Subtraction"))
        elif isinstance(node.op, ast.Mult):
            instructions.append(ARM64Instruction(InstructionType.MUL, ["x0", "x0", "x1"], "Multiplication"))
            
        return instructions
    
    def _compile_return(self, node: ast.Return, state: CompilationState) -> List[ARM64Instruction]:
        """Compile return statement for ARM64"""
        instructions = []
        
        if node.value:
            # Move return value to x0 (ARM64 calling convention)
            if isinstance(node.value, ast.Name):
                var_location = state.variable_map.get(node.value.id, "x0")
                if var_location != "x0":
                    instructions.append(ARM64Instruction(
                        InstructionType.MOV,
                        [var_location, "x0"],
                        "Move return value to x0"
                    ))
        
        instructions.append(ARM64Instruction(InstructionType.RET, [], "Return"))
        return instructions
    
    def _compile_function_call(self, node: ast.Call, state: CompilationState) -> List[ARM64Instruction]:
        """Compile function calls for ARM64"""
        instructions = []
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Handle built-in functions
            if func_name == "print":
                instructions.append(ARM64Instruction(
                    InstructionType.BL,  # Branch with link for ARM64 function calls
                    ["_printf"],
                    "Call print function"
                ))
            else:
                instructions.append(ARM64Instruction(
                    InstructionType.BL,  # Branch with link for ARM64 function calls
                    [f"_{func_name}"],  # macOS requires underscore prefix
                    f"Call {func_name}"
                ))
                
        return instructions
    
    def _compile_conditional(self, node: ast.If, state: CompilationState) -> List[ARM64Instruction]:
        """Compile if statements for ARM64"""
        instructions = []
        
        # Compare condition (simplified) using ARM64 syntax
        instructions.append(ARM64Instruction(InstructionType.CMP, ["x0", "#0"], "Compare condition"))
        instructions.append(ARM64Instruction(InstructionType.BEQ, ["else_label"], "Branch if false"))
        
        # Compile body
        for stmt in node.body:
            new_state = CompilationState(
                stmt, state.available_registers, state.stack_offset,
                state.variable_map.copy(), state.control_flow_depth + 1,
                state.optimization_context
            )
            instructions.extend(self.compile_ast_node(stmt, new_state))
            
        instructions.append(ARM64Instruction(InstructionType.B, ["endif_label"], "Branch to end"))
        instructions.append(ARM64Instruction(InstructionType.NOP, [], "else_label:"))
        
        # Compile else clause if present
        if node.orelse:
            for stmt in node.orelse:
                new_state = CompilationState(
                    stmt, state.available_registers, state.stack_offset,
                    state.variable_map.copy(), state.control_flow_depth + 1,
                    state.optimization_context
                )
                instructions.extend(self.compile_ast_node(stmt, new_state))
        
        instructions.append(ARM64Instruction(InstructionType.NOP, [], "endif_label:"))
        return instructions
    
    def _compile_loop(self, node: ast.For, state: CompilationState) -> List[ARM64Instruction]:
        """Compile for loops"""
        instructions = []
        
        # Simplified loop compilation
        instructions.append(ARM64Instruction(InstructionType.NOP, [], "loop_start:"))
        
        # Loop body
        for stmt in node.body:
            new_state = CompilationState(
                stmt, state.available_registers, state.stack_offset,
                state.variable_map.copy(), state.control_flow_depth + 1,
                state.optimization_context
            )
            instructions.extend(self.compile_ast_node(stmt, new_state))
            
        instructions.append(ARM64Instruction(InstructionType.JMP, ["loop_start"], "Loop back"))
        return instructions
    
    def generate_assembly(self, instructions: List[ARM64Instruction]) -> str:
        """Generate ARM64 assembly code for macOS"""
        assembly = [
            ".section __TEXT,__text,regular,pure_instructions",
            ".globl _main",
            ".align 2",
            "_main:"
        ]
        
        for instruction in instructions:
            assembly.append(str(instruction))
            
        # Add standard ARM64 epilogue
        assembly.extend([
            "    mov x0, #0        // Return 0",
            "    ret",
            ""
        ])
        
        return "\n".join(assembly)
    
    def compile_to_machine_code(self, python_code: str) -> Tuple[str, Dict[str, float]]:
        """Compile Python code directly to ARM64 machine code"""
        start_time = time.time()
        
        try:
            # Parse Python code
            ast_tree = self.parse_python_code(python_code)
            
            # Analyze semantics
            semantic_info = self.semantic_analyzer.analyze(ast_tree)
            logger.info(f"Semantic analysis: {semantic_info}")
            
            # Initialize compilation state
            initial_state = CompilationState(
                ast_tree,
                list(Register),
                0,
                {},
                0,
                {'semantic_info': semantic_info}
            )
            
            # Compile AST to instructions
            instructions = []
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.stmt, ast.expr)):
                    instructions.extend(self.compile_ast_node(node, initial_state))
            
            # Generate assembly
            assembly_code = self.generate_assembly(instructions)
            
            compilation_time = time.time() - start_time
            
            metrics = {
                'compilation_time': compilation_time,
                'instruction_count': len(instructions),
                'code_size': len(assembly_code)
            }
            
            return assembly_code, metrics
            
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            raise
    
    def execute_and_evaluate(self, assembly_code: str, reference_output: Any = None) -> Dict[str, float]:
        """Execute compiled code and evaluate performance"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write assembly to file
            asm_file = os.path.join(tmpdir, "program.s")
            with open(asm_file, 'w') as f:
                f.write(assembly_code)
            
            # Compile with GCC
            executable = os.path.join(tmpdir, "program")
            try:
                subprocess.run([
                    "gcc", "-o", executable, asm_file
                ], check=True, capture_output=True)
                
                # Execute and measure
                start_time = time.time()
                result = subprocess.run([executable], capture_output=True, text=True, timeout=5)
                execution_time = time.time() - start_time
                
                # Calculate correctness (simplified)
                correctness_score = 1.0 if result.returncode == 0 else 0.0
                
                return {
                    'execution_time': execution_time,
                    'correctness_score': correctness_score,
                    'output': result.stdout,
                    'errors': result.stderr
                }
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Assembly compilation failed: {e}")
                return {
                    'execution_time': float('inf'),
                    'correctness_score': 0.0,
                    'output': '',
                    'errors': str(e)
                }
            except subprocess.TimeoutExpired:
                logger.error("Execution timeout")
                return {
                    'execution_time': float('inf'),
                    'correctness_score': 0.0,
                    'output': '',
                    'errors': 'Timeout'
                }

def main():
    """Example usage of PyCodeRL"""
    
    # Initialize compiler
    compiler = PyCodeRLCompiler()
    
    # Example Python programs
    test_programs = [
        """
def add_numbers(a, b):
    result = a + b
    return result

x = 5
y = 3
z = add_numbers(x, y)
""",
        """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

result = factorial(5)
""",
        """
numbers = [1, 2, 3, 4, 5]
total = 0
for num in numbers:
    total = total + num
print(total)
"""
    ]
    
    print("PyCodeRL: Direct Python to x86 Machine Code Compiler")
    print("=" * 60)
    
    for i, program in enumerate(test_programs, 1):
        print(f"\nTest Program {i}:")
        print("-" * 30)
        print(program)
        
        try:
            # Compile to assembly
            assembly, compile_metrics = compiler.compile_to_machine_code(program)
            
            print(f"\nGenerated Assembly:")
            print("-" * 20)
            print(assembly)
            
            print(f"\nCompilation Metrics:")
            print(f"  Compilation time: {compile_metrics['compilation_time']:.4f}s")
            print(f"  Instructions generated: {compile_metrics['instruction_count']}")
            print(f"  Assembly size: {compile_metrics['code_size']} bytes")
            
            # Execute and evaluate
            exec_metrics = compiler.execute_and_evaluate(assembly)
            print(f"\nExecution Results:")
            print(f"  Execution time: {exec_metrics['execution_time']:.4f}s")
            print(f"  Correctness score: {exec_metrics['correctness_score']}")
            print(f"  Output: {exec_metrics['output']}")
            if exec_metrics['errors']:
                print(f"  Errors: {exec_metrics['errors']}")
                
        except Exception as e:
            print(f"Compilation failed: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
    