#!/bin/bash
# Test script to simulate GitHub Action locally

set -e

echo "🚀 Testing LLM Evaluation locally (simulating GitHub Action)"
echo "================================================"

# Check environment
echo "📋 Environment Check:"
python --version
echo "✅ Python ready"

# Check dependencies
python -c "import deepeval; print('✅ DeepEval:', deepeval.__version__)"
python -c "import openai; print('✅ OpenAI client ready')"
python -c "import anthropic; print('✅ Anthropic client ready')"

# Check API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set"
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY not set"
    exit 1
fi

echo "✅ API keys configured"

# Create results directory
mkdir -p test-results

# Run single test (like GitHub Action default)
echo ""
echo "🧪 Running single test case evaluation..."
python -m pytest test_llm_evaluation.py \
  -k "test_case0 and test_correctness and claude" \
  --junitxml=test-results/junit-local.xml \
  --html=test-results/report-local.html \
  --self-contained-html \
  -v --tb=short

# Check results
if [ -f test-results/junit-local.xml ]; then
    echo ""
    echo "📊 Test Results Summary:"

    # Extract metrics from JUnit XML
    TESTS=$(grep -o 'tests="[0-9]*"' test-results/junit-local.xml | grep -o '[0-9]*')
    FAILURES=$(grep -o 'failures="[0-9]*"' test-results/junit-local.xml | grep -o '[0-9]*')
    TIME=$(grep -o 'time="[0-9.]*"' test-results/junit-local.xml | grep -o '[0-9.]*' | head -1)

    echo "- Tests Run: $TESTS"
    echo "- Failures: $FAILURES"
    echo "- Duration: ${TIME}s"

    # Extract cost if available
    COST=$(grep -o 'api_cost_usd.*value="[^"]*"' test-results/junit-local.xml | grep -o '\$[0-9.]*' || echo "N/A")
    echo "- API Cost: $COST"

    # Check if passed
    if [ "$FAILURES" = "0" ]; then
        echo "✅ All tests passed!"
    else
        echo "❌ $FAILURES test(s) failed"
        exit 1
    fi
else
    echo "❌ No test results found"
    exit 1
fi

echo ""
echo "📁 Generated files:"
ls -la test-results/

echo ""
echo "🎉 Local test completed successfully!"
echo "📋 Check test-results/report-local.html for detailed results"