#!/bin/bash

# Run all codebook comparison tests
# Executes configurations in order of increasing complexity

echo "Starting comprehensive codebook comparison tests"

# Change to project root directory
cd "$(dirname "$0")/.."

# Test 1: Standard (baseline)
echo "Test 1: Standard Configuration"
python3 demos/codebook_comparison.py --config test1
if [ $? -ne 0 ]; then
    echo "FAILED: Test 1"
    exit 1
fi

# Test 2: Experimental
echo "Test 2: Experimental Configuration"
python3 demos/codebook_comparison.py --config test2
if [ $? -ne 0 ]; then
    echo "FAILED: Test 2"
    exit 1
fi

# Test 3: High Resolution
echo "Test 3: High Resolution Configuration"
python3 demos/codebook_comparison.py --config test3
if [ $? -ne 0 ]; then
    echo "FAILED: Test 3"
    exit 1
fi

# Test 4: Memory Efficient
echo "Test 4: Memory Efficient Configuration"
python3 demos/codebook_comparison.py --config test4
if [ $? -ne 0 ]; then
    echo "FAILED: Test 4"
    exit 1
fi

# Test 5: Multi-Metric
echo "Test 5: Multi-Metric Configuration"
python3 demos/codebook_comparison.py --config test5
if [ $? -ne 0 ]; then
    echo "FAILED: Test 5"
    exit 1
fi

echo "All tests completed successfully"
echo "Results: demo_outputs/"
echo "Documentation: docs/codebook_comparison_tests.md"
