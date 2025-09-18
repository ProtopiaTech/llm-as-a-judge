"""
Pytest configuration for LLM evaluation tests
Adds custom properties to JUnit XML for LLM-specific metrics
"""

import pytest
from _pytest.reports import TestReport


def pytest_runtest_makereport(item, call):
    """Enhance test reports with LLM metrics for JUnit XML"""
    if call.when == "call":
        # Add custom properties to JUnit XML if test has metadata
        if hasattr(item, 'callspec') and hasattr(item.callspec, 'params'):
            params = item.callspec.params

            # Add model and temperature info
            if "model" in params:
                item.user_properties.append(("model", params["model"]))
            if "temperature" in params:
                item.user_properties.append(("temperature", str(params["temperature"])))
            if "test_case" in params:
                test_case = params["test_case"]
                item.user_properties.append(("question", test_case.get("question", "")))
                item.user_properties.append(("expected_answer", test_case.get("expected_answer", "")[:200] + "..."))

        # Extract LLM response and cost if stored in test
        if hasattr(item, 'llm_response'):
            item.user_properties.append(("generated_answer", item.llm_response[:300] + "..."))
        if hasattr(item, 'api_cost'):
            item.user_properties.append(("api_cost_usd", f"${item.api_cost:.4f}"))
        if hasattr(item, 'tokens_used'):
            item.user_properties.append(("tokens_used", str(item.tokens_used)))

        # Add DeepEval metrics if available in test function
        if hasattr(item, 'deepeval_metrics'):
            for metric_name, metric_data in item.deepeval_metrics.items():
                item.user_properties.append((f"{metric_name}_score", str(metric_data.get('score', 'N/A'))))
                item.user_properties.append((f"{metric_name}_threshold", str(metric_data.get('threshold', 'N/A'))))
                item.user_properties.append((f"{metric_name}_passed", str(metric_data.get('passed', False))))
                reason = metric_data.get('reason', '')
                if reason:
                    item.user_properties.append((f"{metric_name}_reason", reason[:500] + "..." if len(reason) > 500 else reason))


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add metadata"""
    for item in items:
        # Add marker for async tests
        if "async" in item.name or hasattr(item.function, "__code__") and "async" in str(item.function.__code__.co_flags):
            item.add_marker(pytest.mark.asyncio)

        # Add slow marker for tests that call APIs
        if "test_correctness_and_style" in item.name:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_metadata():
    """Fixture to provide test metadata"""
    return {
        "test_suite": "LLM-as-a-Judge Evaluation",
        "framework": "DeepEval + pytest",
        "judge_model": "gpt-5"
    }