import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
import json # For creating mock JSON file content
import os

from src.core.code_generator import MultiLanguageCodeGenerator, PythonAnalyzer, CodeQualityChecker, BestPracticesDatabase
from src.core.generation_structures import CodeSpecification, CodeGenerationResult
from src.models import ChatCompletionResponse, Choice, Message as OpenHandsMessage
from src.core.aggregator import LLMAggregator # For mocking

class TestPythonAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PythonAnalyzer()
        # Initialize logger for the test class if it's used by PythonAnalyzer directly or for consistency
        # For now, assuming PythonAnalyzer uses the module-level logger from code_generator.py

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

    def test_analyze_function_details(self):
        code = """
def my_func(a, b):
    # A comment line
    try:
        x = a + b
        print(x) # Another line
    except Exception as e: # Line 6
        print(f"Error: {e}")
    # Final line of func body
"""
        # Expected body line count from line 3 to line 8 (inclusive) is 6
        # Line numbers are 1-indexed.
        # first_stmt line 3, last_stmt line 8. (8-3)+1 = 6
        analysis = self.analyzer.analyze_code_structure(code)
        self.assertIsNone(analysis.get("error"), f"Analysis failed with: {analysis.get('error')}")
        self.assertEqual(len(analysis["functions"]), 1)
        func_info = analysis["functions"][0]
        self.assertEqual(func_info["name"], "my_func")
        self.assertEqual(func_info["body_line_count"], 6)
        self.assertIn(6, func_info["generic_except_clauses"])
        self.assertEqual(func_info["start_lineno"], 2) # Corrected: def line
        self.assertEqual(func_info["end_lineno"], 8) # Corrected: last line of function

    def test_analyze_classes_and_methods(self):
        code = """
class MyClass:
    '''Class docstring.'''
    def __init__(self, val): # Line 4
        self.val = val # Line 5

    def method_one(self, x): # Line 7
        '''Method one docstring.'''
        return x * self.val # Line 9

    def method_two(self): # Line 11
        pass # Line 12
"""
        results = self.analyzer.analyze_code_structure(code)
        self.assertIsNone(results.get("error"), f"Analysis failed with: {results.get('error')}")
        self.assertEqual(len(results["classes"]), 1)
        my_class = results["classes"][0]
        self.assertEqual(my_class["name"], "MyClass")
        self.assertTrue(my_class["docstring_exists"])
        self.assertTrue(my_class["docstring_preview"].startswith("Class docstring."))
        self.assertEqual(my_class["start_lineno"], 2)
        self.assertEqual(my_class["end_lineno"], 12)
        # Class body: __init__ starts line 4, method_two ends line 12. (12-4)+1 = 9
        self.assertEqual(my_class["body_line_count"], 9)

        self.assertEqual(len(my_class["methods"]), 3)
        init_method = next(m for m in my_class["methods"] if m["name"] == "__init__")
        method_one = next(m for m in my_class["methods"] if m["name"] == "method_one")
        method_two = next(m for m in my_class["methods"] if m["name"] == "method_two")

        self.assertEqual(init_method["args"], ["self", "val"])
        self.assertFalse(init_method["docstring_exists"])
        self.assertEqual(init_method["body_line_count"], 1) # self.val = val

        self.assertEqual(method_one["args"], ["self", "x"])
        self.assertTrue(method_one["docstring_exists"])
        self.assertEqual(method_one["body_line_count"], 1) # return x * self.val

        self.assertEqual(method_two["args"], ["self"])
        self.assertFalse(method_two["docstring_exists"])
        self.assertEqual(method_two["body_line_count"], 1) # pass


class TestCodeQualityChecker(unittest.TestCase):
    def setUp(self):
        self.checker = CodeQualityChecker(max_line_length=80, max_function_lines=10) # Reduced for testing

    def test_empty_code(self):
        results = self.checker.check_python_code("")
        self.assertIn("Code is empty.", results["issues_found"])
        self.assertEqual(results["quality_score"], 0.0)

    def test_todo_fixme_comments(self): # Case-insensitive check
        code = "# todo: Fix this\n# FIXME: Another one"
        results = self.checker.check_python_code(code)
        self.assertIn("Line 1: Contains 'TODO' or 'FIXME'.", results["issues_found"])
        self.assertIn("Line 2: Contains 'TODO' or 'FIXME'.", results["issues_found"])

    def test_long_lines(self):
        code = "a_very_long_variable_name_that_will_surely_exceed_the_eighty_character_limit_set_for_this_test = 'some value'"
        results = self.checker.check_python_code(code)
        self.assertTrue(any("Exceeds max line length of 80" in issue for issue in results["issues_found"]))

    def test_docstring_checks(self): # Combined from previous test
        code = "def my_func():\n  pass\n\nclass MyClazz:\n  def method(self):\n    pass"
        ast_results = {
            "functions": [{"name": "my_func", "docstring_exists": False, "args": []}],
            "classes": [{"name": "MyClazz", "docstring_exists": False, "methods": [
                {"name": "method", "docstring_exists": False, "args": ["self"]}
            ]}]
        }
        results = self.checker.check_python_code(code, ast_analysis_results=ast_results)
        self.assertIn("Function 'my_func': Missing docstring.", results["issues_found"])
        self.assertIn("Class 'MyClazz': Missing docstring.", results["issues_found"])
        self.assertIn("Method 'MyClazz.method': Missing docstring.", results["issues_found"])

    def test_check_naming_conventions(self):
        ast_results = {
            "functions": [{"name": "MyFunction", "args": ["BadArgName", "good_arg"], "docstring_exists": True}],
            "classes": [{"name": "my_class", "docstring_exists": True, "methods": [
                {"name": "MyMethod", "args": ["self"], "docstring_exists": True}
            ]}]
        }
        results = self.checker.check_python_code("code", ast_analysis_results=ast_results)
        self.assertIn("Function Name 'MyFunction': Does not appear to follow snake_case naming convention.", results["issues_found"])
        self.assertIn("Function 'MyFunction' Argument 'BadArgName': Does not appear to follow snake_case.", results["issues_found"])
        self.assertIn("Class Name 'my_class': Does not appear to follow PascalCase naming convention.", results["issues_found"])
        self.assertIn("Method Name 'my_class.MyMethod': Does not appear to follow snake_case.", results["issues_found"])

    def test_check_function_length(self):
        # Code string is not strictly needed if AST results are mocked, but good for context
        code = "def long_func():\n" + "\n".join(["    pass"] * 11)
        mock_ast_results = {
             "functions": [{"name": "long_func", "docstring_exists": True, "body_line_count": 11, "args":[]}],
        }
        result = self.checker.check_python_code(code, ast_analysis_results=mock_ast_results)
        self.assertTrue(any(f"Function 'long_func': Exceeds max length of {self.checker.max_function_lines} lines (actual: 11)" in issue for issue in result["issues_found"]))

    def test_check_wildcard_imports(self):
        mock_ast_results = {"imports": ["from module import *"]}
        result = self.checker.check_python_code("code", ast_analysis_results=mock_ast_results)
        self.assertIn("Import Statement: Contains wildcard import: 'from module import *'.", result["issues_found"])

    def test_check_generic_exceptions(self):
        mock_ast_results = {
            "functions": [{"name": "foo", "generic_except_clauses": [3], "args":[], "docstring_exists": True}],
            "classes": [{"name": "Bar", "methods": [{"name": "baz", "generic_except_clauses": [7], "args":["self"], "docstring_exists": True}], "docstring_exists": True}]
        }
        result = self.checker.check_python_code("code", ast_analysis_results=mock_ast_results)
        self.assertIn("Function 'foo': Line 3 uses a generic 'except:' or 'except Exception:' clause. Consider using more specific exceptions.", result["issues_found"])
        self.assertIn("Method 'Bar.baz': Line 7 uses a generic 'except:' or 'except Exception:' clause. Consider using more specific exceptions.", result["issues_found"])

    def test_clean_code(self):
        code = "def add(a, b):\n  '''Adds two numbers.'''\n  return a + b"
        ast_results = {
            "functions": [{"name": "add", "docstring_exists": True, "args": ["a", "b"], "body_line_count": 1, "generic_except_clauses":[]}],
            "classes": [],
            "imports": ["math"]
        }
        results = self.checker.check_python_code(code, ast_analysis_results=ast_results)
        self.assertEqual(len(results["issues_found"]), 0)
        self.assertEqual(results["quality_score"], 1.0)


class TestMultiLanguageCodeGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_llm_aggregator = AsyncMock(spec=LLMAggregator)
        # MultiLanguageCodeGenerator now instantiates its own components.
        # We can mock these components if needed for more focused unit tests,
        # or use the real ones for more of an integration-style unit test.
        # For now, using real ones to see their interaction.
        self.code_generator = MultiLanguageCodeGenerator(llm_aggregator=self.mock_llm_aggregator)
        # Mock the BestPracticesDatabase part for predictable prompt content
        self.code_generator.best_practices_db = MagicMock(spec=BestPracticesDatabase)

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_generate_python_code_success_with_best_practices_and_checks(self):
        spec = CodeSpecification(target_language="python", prompt_details="Create a function that adds two numbers.")
        mock_generated_code = "def add(a, b):\n  # Adds two numbers.\n  # TODO: Add type hints\n  return a + b # This line is a bit long for a simple sum if max_line_length is very small"

        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_generated_code), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="cg_resp_1", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        # Configure mock for best_practices_db
        self.code_generator.best_practices_db.get_practices = MagicMock(return_value=["Use type hints.", "Keep functions small."])

        result = self._run_async(self.code_generator.generate_code(spec))

        self.mock_llm_aggregator.chat_completion.assert_called_once()
        # Check that best practices were included in the prompt to the LLM
        prompt_sent_to_llm = self.mock_llm_aggregator.chat_completion.call_args[0][0].messages[1].content
        self.assertIn("Use type hints.", prompt_sent_to_llm)
        self.assertIn("Keep functions small.", prompt_sent_to_llm)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.generated_code, mock_generated_code)
        self.assertIsNotNone(result.quality_score)
        # Based on the mock code and default CodeQualityChecker(max_line_length=100, max_function_lines=50)
        # - Missing docstring for 'add'
        # - Contains 'TODO'
        # The line length check might pass depending on actual length vs default 100.
        # "  return a + b # This line is a bit long for a simple sum if max_line_length is very small" is 88 chars.
        self.assertTrue(len(result.issues_found) >= 2)
        self.assertIn("Function 'add': Missing docstring.", result.issues_found)
        self.assertIn("Line 3: Contains 'TODO' or 'FIXME'.", result.issues_found)


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
