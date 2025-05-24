"""
Script to validate solution files from permutation sorting experiments.

Validates that move sequences (X, L, R) in solution files actually solve the permutation
problems correctly by:
- Applying each move sequence to verify it reaches the sorted state
- Detecting suboptimal X moves (swapping when elements are already in correct order)
- Removing invalid solutions and updating the solution file
- Providing validation statistics and error reports

Used for quality control of experiment results to ensure solution correctness.
"""

import pandas as pd
import torch
import argparse
import sys
import os

# Add parent directory to path for src imports (if needed for future extensions)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def apply_move(state: torch.Tensor, move_type: str) -> torch.Tensor:
    """Apply a single move (X, L, or R) to the state"""
    if move_type == 'X':
        # Swap first two elements
        state = state.clone()
        state[0], state[1] = state[1].clone(), state[0].clone()
    elif move_type == 'L':
        # Left shift
        state = torch.roll(state, -1)
    elif move_type == 'R':
        # Right shift
        state = torch.roll(state, 1)
    return state

def apply_solution(state: torch.Tensor, solution: str) -> torch.Tensor:
    """Apply sequence of moves to the state"""
    moves = solution.split('.')
    for move in moves:
        state = apply_move(state, move)
    return state

def is_sorted(state: torch.Tensor) -> bool:
    """Check if state is sorted (0 to n-1)"""
    return torch.all(state == torch.arange(len(state)))

def check_suboptimal_x_moves(solution: str, initial_perm: torch.Tensor) -> bool:
    """Check if solution contains suboptimal X moves (X when first two elements are already in order)"""
    current = initial_perm.clone()
    moves = solution.split('.')
    
    for move in moves:
        if move == 'X':
            if current[0] < current[1]:
                return True
            current[0], current[1] = current[1].clone(), current[0].clone()
        elif move == 'L':
            current = torch.roll(current, -1)
        elif move == 'R':
            current = torch.roll(current, 1)
    
    return False

def validate_solutions(solutions_file: str):
    """Validate all solutions in the file"""
    df = pd.read_csv(solutions_file)
    original_count = len(df)

    total_moves = 0
    valid_solutions = 0
    unsolved_cases = 0
    invalid_solutions = 0
    total_time = 0
    invalid_details = []  # Track details of invalid solutions
    suboptimal_solutions = 0
    
    print(f"Validating {len(df)} solution entries...")
    print(f"Columns in file: {list(df.columns)}")
    
    for idx, row in df.iterrows():
        # Skip rows that don't have a solution (e.g., failed experiments)
        if pd.isna(row.get('solution')) or row.get('solution') == '':
            continue
            
        # Convert permutation string to tensor
        perm = torch.tensor([int(x) for x in row['permutation'].split(',')])
        solution = row['solution']
        
        # Check if solution was not found
        if solution == 'Not found':
            # Count as unsolved, not invalid
            unsolved_cases += 1
            continue

        # Count moves for valid solutions
        moves = solution.split('.')
        total_moves += len(moves)

        # Count time
        if 'solve_time' in row:
            solve_time = row['solve_time']
            total_time += solve_time
        
        # Check if solution contains suboptimal X moves
        if check_suboptimal_x_moves(solution, perm):
            suboptimal_solutions += 1
    
        # Apply solution to validate
        try:
            final_state = apply_solution(perm, solution)
            
            # Validate result
            if is_sorted(final_state):
                valid_solutions += 1
            else:
                invalid_solutions += 1
                invalid_details.append({
                    'row': idx,
                    'permutation': row['permutation'],
                    'solution': solution,
                    'final_state': final_state.tolist(),
                    'reason': 'Did not reach sorted state'
                })
                print(f"Invalid solution at row {idx}: {solution}")
                print(f"  Permutation: {row['permutation']}")
                print(f"  Final state: {final_state.tolist()}")
        except Exception as e:
            invalid_solutions += 1
            invalid_details.append({
                'row': idx,
                'permutation': row['permutation'],
                'solution': solution,
                'reason': f'Error applying solution: {str(e)}'
            })
            print(f"Error validating solution at row {idx}: {e}")
            print(f"  Permutation: {row['permutation']}")
            print(f"  Solution: {solution}")
    
    # Print summary (no file modification)
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"Total rows in file: {original_count}")
    print(f"Rows with solutions: {valid_solutions + invalid_solutions}")
    print(f"Valid solutions: {valid_solutions}")
    print(f"Invalid solutions: {invalid_solutions}")
    print(f"Unsolved cases ('Not found'): {unsolved_cases}")
    print(f"Solutions with suboptimal X moves: {suboptimal_solutions}")
    print(f"Total moves in valid solutions: {total_moves}")
    print(f"Total solve time: {total_time:.2f} seconds")
    
    if invalid_details:
        print(f"\n{'='*50}")
        print("INVALID SOLUTIONS DETAILS")
        print("="*50)
        for detail in invalid_details:
            print(f"\nRow {detail['row']}:")
            print(f"  Permutation: {detail['permutation']}")
            print(f"  Solution: {detail['solution']}")
            print(f"  Reason: {detail['reason']}")
            if 'final_state' in detail:
                print(f"  Final state: {detail['final_state']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate solution files from permutation sorting experiments.")
    parser.add_argument("solutions_file", type=str, help="Path to the solution file")
    args = parser.parse_args()

    validate_solutions(args.solutions_file) 