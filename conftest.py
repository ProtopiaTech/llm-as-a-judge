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
        if hasattr(item, 'test_case_id'):
            item.user_properties.append(("test_case_id", item.test_case_id))

        # Add DeepEval metrics if available in test function
        if hasattr(item, 'deepeval_metrics'):
            for metric_name, metric_data in item.deepeval_metrics.items():
                item.user_properties.append((f"{metric_name}_score", str(metric_data.get('score', 'N/A'))))
                item.user_properties.append((f"{metric_name}_threshold", str(metric_data.get('threshold', 'N/A'))))
                item.user_properties.append((f"{metric_name}_passed", str(metric_data.get('passed', False))))
                reason = metric_data.get('reason', '')
                if reason:
                    item.user_properties.append((f"{metric_name}_reason", reason))

        # Add Agent evaluation metrics if available
        if hasattr(item, 'agent_evaluation_metrics'):
            for metric_name, metric_data in item.agent_evaluation_metrics.items():
                item.user_properties.append((f"agent_{metric_name}_score", str(metric_data.get('score', 'N/A'))))
                item.user_properties.append((f"agent_{metric_name}_threshold", str(metric_data.get('threshold', 'N/A'))))
                item.user_properties.append((f"agent_{metric_name}_passed", str(metric_data.get('passed', False))))
                reason = metric_data.get('reason', '')
                if reason:
                    item.user_properties.append((f"agent_{metric_name}_reason", reason))

        # Add Agent-specific properties
        if hasattr(item, 'agent_scenario'):
            item.user_properties.append(("agent_scenario", item.agent_scenario))
        if hasattr(item, 'expected_tools'):
            item.user_properties.append(("expected_tools", str(item.expected_tools)))
        if hasattr(item, 'actual_tools'):
            item.user_properties.append(("actual_tools", str(item.actual_tools)))
        if hasattr(item, 'expected_tool_inputs'):
            item.user_properties.append(("expected_tool_inputs", str(item.expected_tool_inputs)))
        if hasattr(item, 'actual_tool_inputs'):
            item.user_properties.append(("actual_tool_inputs", str(item.actual_tool_inputs)))
        if hasattr(item, 'safety_critical'):
            item.user_properties.append(("safety_critical", str(item.safety_critical)))
        if hasattr(item, 'final_response'):
            # Truncate response for XML
            response_short = item.final_response[:300] + "..." if len(item.final_response) > 300 else item.final_response
            item.user_properties.append(("agent_final_response", response_short))


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