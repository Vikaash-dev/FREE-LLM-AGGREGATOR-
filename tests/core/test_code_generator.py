import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.core.code_generator import MultiLanguageCodeGenerator, PythonAnalyzer, CodeQualityChecker
from src.core.generation_structures import CodeSpecification, CodeGenerationResult
from src.models import ChatCompletionResponse, Choice, Message as OpenHandsMessage
from src.core.aggregator import LLMAggregator # For mocking

class TestPythonAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PythonAnalyzer()

    def test_analyze_empty_code(self):
        results = self.analyzer.analyze_code_structure("")
        self.assertIn("error", results)
        self.assertEqual(results["error"], "Input code string is empty.")

    def test_analyze_syntax_error(self):
        code = "def func(:\n  pass"
        results = self.analyzer.analyze_code_structure(code)
        self.assertIn("error", results)
        self.assertTrue(results["error"].startswith("SyntaxError:"))

    def test_analyze_imports(self):
        code = "import os\nimport sys as system\nfrom collections import defaultdict"
        results = self.analyzer.analyze_code_structure(code)
        self.assertNotIn("error", results)
        self.assertIn("os", results["imports"])
        self.assertIn("sys as system", results["imports"])
        self.assertIn("from collections import defaultdict", results["imports"])

    def test_analyze_functions(self):
        code = "def foo(a, b=1):\n  '''Docstring for foo.'''\n  pass\ndef bar():\n  pass"
        results = self.analyzer.analyze_code_structure(code)
        self.assertNotIn("error", results)
        self.assertEqual(len(results["functions"]), 2)
        foo_func = next(f for f in results["functions"] if f["name"] == "foo")
        bar_func = next(f for f in results["functions"] if f["name"] == "bar")

        self.assertEqual(foo_func["args"], ["a", "b=<default>"])
        self.assertTrue(foo_func["docstring_exists"])
        self.assertTrue(foo_func["docstring_preview"].startswith("Docstring for foo."))

        self.assertEqual(bar_func["args"], [])
        self.assertFalse(bar_func["docstring_exists"])

    def test_analyze_classes_and_methods(self):
        code = """
class MyClass:
    '''Class docstring.'''
    def __init__(self, val):
        self.val = val

    def method_one(self, x):
        '''Method one docstring.'''
        return x * self.val

    def method_two(self): # No docstring
        pass
"""
        results = self.analyzer.analyze_code_structure(code)
        self.assertNotIn("error", results)
        self.assertEqual(len(results["classes"]), 1)
        my_class = results["classes"][0]
        self.assertEqual(my_class["name"], "MyClass")
        self.assertTrue(my_class["docstring_exists"])
        self.assertTrue(my_class["docstring_preview"].startswith("Class docstring."))

        self.assertEqual(len(my_class["methods"]), 3) # __init__, method_one, method_two
        init_method = next(m for m in my_class["methods"] if m["name"] == "__init__")
        method_one = next(m for m in my_class["methods"] if m["name"] == "method_one")
        method_two = next(m for m in my_class["methods"] if m["name"] == "method_two")

        self.assertEqual(init_method["args"], ["self", "val"])
        self.assertFalse(init_method["docstring_exists"]) # By default, __init__ docstring isn't strictly checked by this basic analyzer for existence, but its presence is noted

        self.assertEqual(method_one["args"], ["self", "x"])
        self.assertTrue(method_one["docstring_exists"])

        self.assertEqual(method_two["args"], ["self"])
        self.assertFalse(method_two["docstring_exists"])


class TestCodeQualityChecker(unittest.TestCase):
    def setUp(self):
        self.checker = CodeQualityChecker(max_line_length=80)

    def test_empty_code(self):
        results = self.checker.check_python_code("")
        self.assertIn("Code is empty.", results["issues_found"])
        self.assertEqual(results["quality_score"], 0.0)

    def test_todo_fixme_comments(self):
        code = "# TODO: Fix this\n# FIXME: Another one"
        results = self.checker.check_python_code(code)
        self.assertIn("Line 1: Contains 'TODO' or 'FIXME'.", results["issues_found"])
        self.assertIn("Line 2: Contains 'TODO' or 'FIXME'.", results["issues_found"])
        self.assertLess(results["quality_score"], 1.0)

    def test_long_lines(self):
        code = "a = " + "'" * 100 # Creates a line longer than 80
        results = self.checker.check_python_code(code)
        self.assertIn("Line 1: Exceeds max line length of 80 (length: 104).", results["issues_found"])

    def test_docstring_checks_with_ast_results(self):
        code = "def my_func():\n  pass\n\nclass MyClazz:\n  def method(self):\n    pass"
        ast_results = {
            "functions": [{"name": "my_func", "docstring_exists": False}],
            "classes": [{"name": "MyClazz", "docstring_exists": False, "methods": [
                {"name": "method", "docstring_exists": False}
            ]}]
        }
        results = self.checker.check_python_code(code, ast_analysis_results=ast_results)
        self.assertIn("Function 'my_func': Missing docstring.", results["issues_found"])
        self.assertIn("Class 'MyClazz': Missing docstring.", results["issues_found"])
        self.assertIn("Method 'MyClazz.method': Missing docstring.", results["issues_found"])

    def test_clean_code(self):
        code = "def add(a, b):\n  '''Adds two numbers.'''\n  return a + b"
        ast_results = {
            "functions": [{"name": "add", "docstring_exists": True, "args": ["a", "b"]}],
            "classes": []
        }
        results = self.checker.check_python_code(code, ast_analysis_results=ast_results)
        self.assertEqual(len(results["issues_found"]), 0)
        self.assertEqual(results["quality_score"], 1.0)


class TestMultiLanguageCodeGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_llm_aggregator = AsyncMock(spec=LLMAggregator)
        # Using real PythonAnalyzer and CodeQualityChecker as they don't have external deps for these tests
        self.code_generator = MultiLanguageCodeGenerator(llm_aggregator=self.mock_llm_aggregator)

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_generate_python_code_success(self):
        spec = CodeSpecification(target_language="python", prompt_details="Create a function that adds two numbers.")
        mock_generated_code = "def add(a, b):\n  '''Adds two numbers.'''\n  return a + b"

        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_generated_code), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="cg_resp_1", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        result = self._run_async(self.code_generator.generate_code(spec))

        self.mock_llm_aggregator.chat_completion.assert_called_once()
        self.assertTrue(result.succeeded)
        self.assertEqual(result.generated_code, mock_generated_code)
        self.assertEqual(result.language, "python")
        self.assertIsNone(result.error_message)
        self.assertIsNotNone(result.quality_score) # Should have a score
        # Issues might be empty for this clean code
        self.assertIsInstance(result.issues_found, list)

    def test_generate_python_code_with_markdown_fences(self):
        spec = CodeSpecification(target_language="python", prompt_details="Simple print statement.")
        mock_generated_code_fenced = "```python\nprint('Hello')\n```"
        expected_cleaned_code = "print('Hello')"

        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_generated_code_fenced), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="cg_resp_2", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        result = self._run_async(self.code_generator.generate_code(spec))
        self.assertTrue(result.succeeded)
        self.assertEqual(result.generated_code, expected_cleaned_code)

    def test_generate_python_code_llm_returns_error(self):
        spec = CodeSpecification(target_language="python", prompt_details="Something unclear.")
        llm_error_message = "ERROR: Request is too ambiguous."

        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=llm_error_message), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="cg_resp_3", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        result = self._run_async(self.code_generator.generate_code(spec))
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.generated_code)
        self.assertEqual(result.error_message, llm_error_message)

    def test_generate_unsupported_language(self):
        spec = CodeSpecification(target_language="javascript", prompt_details="Create a JS function.")
        result = self._run_async(self.code_generator.generate_code(spec))
        self.assertFalse(result.succeeded)
        self.assertIsNotNone(result.error_message)
        self.assertTrue("not supported" in result.error_message)
        self.mock_llm_aggregator.chat_completion.assert_not_called()


if __name__ == '__main__':
    unittest.main()
