"""
LLM-as-a-Judge Evaluation Tests using pytest + DeepEval
Based on input.md blueprint - native pytest integration with JUnit XML output
"""

import json
import os
import asyncio
from typing import List, Dict
import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Load environment variables
load_dotenv()


class LLMResponseGenerator:
    """Helper class to generate responses from different LLM models"""

    def __init__(self):
        self.openai_client = AsyncOpenAI()
        self.anthropic_client = AsyncAnthropic()
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()

    async def generate_response(self, model: str, question: str) -> str:
        """Generate response using specified model"""
        if "gpt" in model.lower():
            return await self._call_openai(model, question)
        elif "claude" in model.lower():
            return await self._call_anthropic(model, question)
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def _call_openai(self, model: str, question: str) -> str:
        """Call OpenAI API"""
        try:
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
            }

            # Set temperature for all models except gpt-5 which only supports default (1.0)
            if "gpt-5" not in model:
                params["temperature"] = 0.3
            # gpt-5 models use default temperature=1.0 automatically

            if "gpt-5" in model:
                params["max_completion_tokens"] = 500
            else:
                params["max_tokens"] = 500

            response = await self.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating response: {e}"

    async def _call_anthropic(self, model: str, question: str) -> str:
        """Call Anthropic API"""
        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                system=self.system_prompt,
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
                max_tokens=500
            )
            return response.content[0].text

        except Exception as e:
            return f"Error generating response: {e}"

    async def _call_openai_with_cost(self, model: str, question: str) -> tuple[str, int, float]:
        """Call OpenAI API and return response, tokens, cost"""
        try:
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
            }

            if "gpt-5" not in model:
                params["temperature"] = 0.3

            if "gpt-5" in model:
                params["max_completion_tokens"] = 500
            else:
                params["max_tokens"] = 500

            response = await self.openai_client.chat.completions.create(**params)
            content = response.choices[0].message.content

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Calculate cost based on pricing from .env
            cost = self._calculate_openai_cost(model, input_tokens, output_tokens)

            return content, total_tokens, cost

        except Exception as e:
            return f"Error generating response: {e}", 0, 0.0

    async def _call_anthropic_with_cost(self, model: str, question: str) -> tuple[str, int, float]:
        """Call Anthropic API and return response, tokens, cost"""
        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                system=self.system_prompt,
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
                max_tokens=500
            )

            content = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Calculate cost based on pricing from .env
            cost = self._calculate_anthropic_cost(model, input_tokens, output_tokens)

            return content, total_tokens, cost

        except Exception as e:
            return f"Error generating response: {e}", 0, 0.0

    def _calculate_openai_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate OpenAI API cost"""
        pricing = {
            "gpt-5": {"input": 1.25, "output": 10.0},
            "gpt-5-mini-2025-08-07": {"input": 0.25, "output": 2.0},
            "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60}
        }

        model_pricing = pricing.get(model, {"input": 1.0, "output": 3.0})
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        return input_cost + output_cost

    def _calculate_anthropic_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate Anthropic API cost"""
        pricing = {
            "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0}
        }

        model_pricing = pricing.get(model, {"input": 1.0, "output": 4.0})
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        return input_cost + output_cost


# Create global instance
response_generator = LLMResponseGenerator()


def load_test_cases() -> List[Dict]:
    """Load test cases from JSONL file"""
    test_cases = []
    with open("test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_cases.append(json.loads(line.strip()))
    return test_cases


# Load test data
TEST_CASES = load_test_cases()

# Model configurations
MODELS = [
    "gpt-5-mini-2025-08-07",
    "claude-3-5-haiku-20241022",
    "gpt-4o-mini-2024-07-18"
]

TEMPERATURES = [0.3]  # Single temperature for all models


class TestLLMQuality:
    """Main test class for LLM evaluation using DeepEval"""

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    @pytest.mark.parametrize("test_case", TEST_CASES)
    @pytest.mark.asyncio
    async def test_correctness(self, request, model: str, temperature: float, test_case: Dict):
        """Test correctness evaluation for each model/temperature/question combination"""
        await self._run_single_evaluation(request, model, temperature, test_case, "correctness")

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    @pytest.mark.parametrize("test_case", TEST_CASES)
    @pytest.mark.asyncio
    async def test_style(self, request, model: str, temperature: float, test_case: Dict):
        """Test style evaluation for each model/temperature/question combination"""
        await self._run_single_evaluation(request, model, temperature, test_case, "style")

    async def _run_single_evaluation(self, request, model: str, temperature: float, test_case: Dict, metric_type: str):
        """Helper method to run single metric evaluation with detailed tracking"""

        question = test_case["question"]
        expected_answer = test_case["expected_answer"]

        # Generate response and track cost
        generated_answer, tokens_used, api_cost = await self._generate_response_with_cost(model, question)

        # Store response and cost info in pytest item for JUnit XML
        request.node.llm_response = generated_answer
        request.node.tokens_used = tokens_used
        request.node.api_cost = api_cost

        # Create appropriate metric
        if metric_type == "correctness":
            metric = GEval(
                name="Correctness",
                criteria="""
                Evaluate if the generated answer correctly addresses the user's question
                compared to the expected answer. Consider:
                1. Factual accuracy of medical information
                2. Completeness of the response
                3. Whether key safety information is included
                4. Proper handling of off-topic questions (redirecting to KEYTRUDA)
                """,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT
                ],
                evaluation_steps=[
                    "Compare the key medical facts between generated and expected answers",
                    "Check if safety warnings are appropriately included",
                    "Verify if off-topic questions are properly redirected",
                    "Assess if the response directly addresses the user's concern",
                    "Provide correctness score from 0.0 to 1.0"
                ],
                threshold=0.7,
                model=os.getenv("JUDGE_MODEL", "gpt-5"),
                strict_mode=False
            )
        else:  # style
            metric = GEval(
                name="Style",
                criteria="""
                Evaluate if the response follows KEYTRUDA chatbot style guidelines:
                1. Uses simple, everyday language (avoids medical jargon)
                2. Maintains friendly, patient, and supportive tone
                3. Keeps responses concise and clear
                4. Is honest but reassuring when discussing side effects
                5. Always reminds users to consult their doctor
                6. Stays within scope of provided information
                """,
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.CONTEXT
                ],
                evaluation_steps=[
                    "Check if language is simple and avoids unnecessary medical jargon",
                    "Assess if tone is friendly, patient and supportive",
                    "Verify response is concise and clearly structured",
                    "Evaluate balance between honesty and reassurance for medical topics",
                    "Confirm reminder to consult doctor is included when appropriate",
                    "Check if response stays within KEYTRUDA scope",
                    "Provide style score from 0.0 to 1.0"
                ],
                threshold=0.8,
                model=os.getenv("JUDGE_MODEL", "gpt-5"),
                strict_mode=False
            )

        # Create test case for evaluation
        llm_test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            expected_output=expected_answer,
            context=[response_generator.system_prompt]
        )

        # Run evaluation and capture detailed results
        try:
            # Use assert_test to capture detailed evaluation
            try:
                assert_test(llm_test_case, [metric])
                # If we get here, test passed
                score = 1.0  # Passed means score >= threshold
                reason = f"✅ {metric_type.title()} evaluation passed"
                passed = True
            except AssertionError as ae:
                # Parse the assertion error to extract score and reason
                error_msg = str(ae)

                # Extract score from error message like "Metrics: Style [GEval] (score: 0.6, threshold: 0.8..."
                import re
                score_match = re.search(r'score:\s*([\d.]+)', error_msg)
                reason_match = re.search(r'reason:\s*([^)]+)', error_msg)

                score = float(score_match.group(1)) if score_match else 0.0
                reason = reason_match.group(1) if reason_match else error_msg
                passed = False

                # Re-raise if score is too low
                raise

            # Store detailed metrics in pytest item
            request.node.deepeval_metrics = {
                metric_type: {
                    'score': score,
                    'threshold': metric.threshold,
                    'passed': passed,
                    'reason': reason[:500] + "..." if len(reason) > 500 else reason,
                    'judge_model': os.getenv("JUDGE_MODEL", "gpt-5")
                }
            }

        except Exception as e:
            # Store error info
            request.node.deepeval_metrics = {
                metric_type: {
                    'score': 0.0,
                    'threshold': metric.threshold,
                    'passed': False,
                    'reason': f"Evaluation error: {str(e)[:300]}",
                    'judge_model': os.getenv("JUDGE_MODEL", "gpt-5")
                }
            }
            raise

    async def _generate_response_with_cost(self, model: str, question: str) -> tuple[str, int, float]:
        """Generate response and track token usage and cost"""
        if "gpt" in model.lower():
            return await response_generator._call_openai_with_cost(model, question)
        elif "claude" in model.lower():
            return await response_generator._call_anthropic_with_cost(model, question)
        else:
            raise ValueError(f"Unsupported model: {model}")


@pytest.mark.asyncio
async def test_model_availability():
    """Test that all models are accessible"""
    for model in MODELS:
        try:
            response = await response_generator.generate_response(model, "Test question")
            assert "Error" not in response, f"Model {model} returned error: {response}"
        except Exception as e:
            pytest.fail(f"Model {model} is not accessible: {e}")


if __name__ == "__main__":
    # Run tests with JUnit XML output
    pytest.main([
        __file__,
        "--junitxml=test-results/evaluation.xml",
        "--html=test-results/report.html",
        "--self-contained-html",
        "-v"
    ])