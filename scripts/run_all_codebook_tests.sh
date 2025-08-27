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

echo "Skipping non-existent tests (test3..test5). Available: test1, test2"

echo "All tests completed successfully"
echo "Results: demo_outputs/"
echo "Documentation: docs/codebook_comparison_tests.md"
