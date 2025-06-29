#!/usr/bin/env python3

import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluator import evaluate

def test_fixed_implementation():
    """Test the fixed HRCF implementation with proper configuration"""
    
    print("="*60)
    print("Testing Fixed HRCF Implementation")
    print("="*60)
    print()
    
    print("Key fixes implemented (Amazon-CD specific):")
    print("✓ Embedding dimension: 32 → 50")
    print("✓ Learning rate: 0.01 → 0.0015") 
    print("✓ Weight decay: 0.01 → 0.005")
    print("✓ Alpha (geo regularization): 0.1 → 25")
    print("✓ Margin: 0.1 → 0.15")
    print("✓ Num layers: 4 → 8 (critical!)")
    print("✓ Batch size: 1000 → 10000")
    print("✓ Optimizer: Adam → RiemannianSGD")
    print("✓ Negative sampling: Random → WarpSampler")
    print("✓ Adjacency normalization: Simple → HRCF style")
    print()
    
    # Test with initial program
    program_path = os.path.join(current_dir, 'initial_program.py')
    
    if not os.path.exists(program_path):
        print(f"Error: {program_path} not found")
        return
    
    print(f"Evaluating: {program_path}")
    print("Expected performance (based on original HRCF log):")
    print("- Recall@5: ~0.095 (9.5%)")
    print("- Recall@10: ~0.144 (14.4%)")  
    print("- Precision@5: ~0.074 (7.4%)")
    print("- Precision@10: ~0.090 (9.0%)")
    print("- Current performance: Recall@10 = 0.0004 (360x worse!)")
    print()
    
    try:
        results = evaluate(program_path, dataset_name='Amazon-CD')
        
        print("="*60)
        print("RESULTS")
        print("="*60)
        
        print(f"Recall@10: {results['recall_at_10']:.6f}")
        print(f"Recall@20: {results['recall_at_20']:.6f}")
        print(f"Precision@10: {results['precision_at_10']:.6f}")
        print(f"Precision@20: {results['precision_at_20']:.6f}")
        print(f"NDCG@10: {results['ndcg_at_10']:.6f}")
        print(f"NDCG@20: {results['ndcg_at_20']:.6f}")
        print(f"Hit Rate@10: {results['hit_rate_at_10']:.6f}")
        print(f"Hit Rate@20: {results['hit_rate_at_20']:.6f}")
        print(f"Combined Score: {results['combined_score']:.6f}")
        print(f"Execution Time: {results['execution_time']:.2f}s")
        
        if results['error']:
            print(f"Error: {results['error']}")
        else:
            print("✓ No errors!")
            
        print()
        print("Comparison with expected results:")
        print(f"Expected Recall@10: 0.144 (original HRCF)")
        print(f"Previous Recall@10:  0.0004")
        print(f"Current Recall@10:   {results['recall_at_10']:.6f}")
        
        if results['recall_at_10'] > 0:
            improvement_vs_previous = results['recall_at_10'] / 0.0004
            gap_vs_expected = results['recall_at_10'] / 0.144
            print(f"Improvement vs previous: {improvement_vs_previous:.1f}x")
            print(f"Gap to expected: {gap_vs_expected:.1%} of target")
        
        if results['recall_at_10'] > 0.10:  # Within 30% of expected
            print("🎉 SUCCESS: Performance matches original HRCF!")
        elif results['recall_at_10'] > 0.05:  # At least 35% of expected
            print("📈 GOOD PROGRESS: Getting close to expected performance")
        elif results['recall_at_10'] > 0.01:  # At least 7% of expected
            print("📊 SOME PROGRESS: Better but still far from target")
        else:
            print("❌ ISSUE: Performance still much lower than expected")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_implementation() 