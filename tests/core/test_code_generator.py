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
        self.sample_ast_clean = {"functions": [{"name": "foo", "docstring_exists": True, "body_line_count": 1, "args": [], "generic_except_clauses": []}], "classes": [], "imports": []}


    @patch('src.core.code_generator.run_flake8_on_code')
    def test_empty_code(self, mock_run_flake8):
        mock_run_flake8.return_value = [] # Flake8 finds nothing in empty code
        results = self.checker.check_python_code("", ast_analysis_results={"functions": [], "classes": [], "imports": []})
        # Custom check for empty code
        self.assertTrue(any("Code is empty." in issue["message"] for issue in results["issues_found"] if issue["type"] == "custom"))
        self.assertEqual(results["quality_score"], 0.0)
        self.assertEqual(results["flake8_issue_count"], 0)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_todo_fixme_comments(self, mock_run_flake8): # Case-insensitive check
        mock_run_flake8.return_value = []
        code = "# todo: Fix this\n# FIXME: Another one"
        # Provide minimal AST results for this code
        ast_results = {"functions": [], "classes": [], "imports": []}
        results = self.checker.check_python_code(code, ast_analysis_results=ast_results)

        custom_issues = [issue for issue in results["issues_found"] if issue["type"] == "custom"]
        self.assertTrue(any("Line 1: Contains 'TODO' or 'FIXME'." in issue["message"] for issue in custom_issues))
        self.assertTrue(any("Line 2: Contains 'TODO' or 'FIXME'." in issue["message"] for issue in custom_issues))
        self.assertEqual(results["flake8_issue_count"], 0)
        self.assertGreaterEqual(results["custom_issue_count"], 2)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_long_lines_custom_check_no_flake8_e501(self, mock_run_flake8):
        mock_run_flake8.return_value = [] # Flake8 does not report E501
        code = "a_very_long_variable_name_that_will_surely_exceed_the_eighty_character_limit_set_for_this_test = 'some value'"
        ast_results = {"functions": [], "classes": [], "imports": []} # Minimal AST
        results = self.checker.check_python_code(code, ast_analysis_results=ast_results)

        custom_issues = [issue for issue in results["issues_found"] if issue["type"] == "custom"]
        self.assertTrue(any(f"Exceeds custom max line length of {self.checker.max_line_length}" in issue["message"] for issue in custom_issues))
        self.assertEqual(results["flake8_issue_count"], 0)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_custom_line_length_skipped_if_flake8_e501(self, mock_run_flake8):
        long_line_content = 'print("' + 'A'*(self.checker.max_line_length + 20) + '")' # Exceeds max_line_length
        code = f"def func():\n    {long_line_content}"

        # Simulate Flake8 reporting E501 for the long line
        mock_run_flake8.return_value = [
            {"line": 2, "col": self.checker.max_line_length + 1, "code": "E501", "message": "line too long"}
        ]

        # AST results for a function with 1 line body, docstring exists (to avoid other issues)
        mock_ast = {
            "functions": [{"name":"func", "docstring_exists":True, "body_line_count":1, "args":[], "generic_except_clauses":[]}],
            "classes":[],
            "imports":[]
        }

        result = self.checker.check_python_code(code, ast_analysis_results=mock_ast)

        # Ensure custom line length check message is NOT present
        custom_long_line_messages = [
            issue["message"] for issue in result["issues_found"]
            if issue["type"] == "custom" and f"Exceeds custom max line length of {self.checker.max_line_length}" in issue["message"]
        ]
        self.assertEqual(len(custom_long_line_messages), 0, "Custom line length check should be skipped if Flake8 E501 is present.")
        self.assertEqual(result["flake8_issue_count"], 1)
        self.assertTrue(any(issue["code"] == "E501" for issue in result["issues_found"] if issue["type"] == "flake8"))

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_docstring_checks(self, mock_run_flake8):
        mock_run_flake8.return_value = []
        code = "def my_func():\n  pass\n\nclass MyClazz:\n  def method(self):\n    pass"
        ast_results = {
            "functions": [{"name": "my_func", "docstring_exists": False, "args": [], "body_line_count":1, "generic_except_clauses":[]}],
            "classes": [{"name": "MyClazz", "docstring_exists": False, "methods": [
                {"name": "method", "docstring_exists": False, "args": ["self"], "body_line_count":1, "generic_except_clauses":[]}
            ], "body_line_count":1, "generic_except_clauses":[]}], # Added body_line_count for class
            "imports": []
        }
        results = self.checker.check_python_code(code, ast_analysis_results=ast_results)
        custom_issues = {issue["message"] for issue in results["issues_found"] if issue["type"] == "custom"}
        self.assertIn("Function 'my_func': Missing docstring.", custom_issues)
        self.assertIn("Class 'MyClazz': Missing docstring.", custom_issues)
        self.assertIn("Method 'MyClazz.method': Missing docstring.", custom_issues)
        self.assertEqual(results["flake8_issue_count"], 0)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_check_naming_conventions(self, mock_run_flake8):
        mock_run_flake8.return_value = []
        ast_results = {
            "functions": [{"name": "MyFunction", "args": ["BadArgName", "good_arg"], "docstring_exists": True, "body_line_count":1, "generic_except_clauses":[]}],
            "classes": [{"name": "my_class", "docstring_exists": True, "methods": [
                {"name": "MyMethod", "args": ["self"], "docstring_exists": True, "body_line_count":1, "generic_except_clauses":[]}
            ], "body_line_count":1, "generic_except_clauses":[]}],
            "imports": []
        }
        results = self.checker.check_python_code("code", ast_analysis_results=ast_results)
        custom_issues = {issue["message"] for issue in results["issues_found"] if issue["type"] == "custom"}
        self.assertIn("Function Name 'MyFunction': Does not appear to follow snake_case naming convention.", custom_issues)
        self.assertIn("Function 'MyFunction' Argument 'BadArgName': Does not appear to follow snake_case.", custom_issues)
        self.assertIn("Class Name 'my_class': Does not appear to follow PascalCase naming convention.", custom_issues)
        self.assertIn("Method Name 'my_class.MyMethod': Does not appear to follow snake_case.", custom_issues)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_check_function_length(self, mock_run_flake8):
        mock_run_flake8.return_value = []
        code = "def long_func():\n" + "\n".join(["    pass"] * 11)
        mock_ast_results = {
             "functions": [{"name": "long_func", "docstring_exists": True, "body_line_count": 11, "args":[], "generic_except_clauses":[]}],
             "classes": [], "imports": []
        }
        result = self.checker.check_python_code(code, ast_analysis_results=mock_ast_results)
        self.assertTrue(any(f"Function 'long_func': Exceeds max length of {self.checker.max_function_lines} lines (actual: 11)" in issue["message"]
                            for issue in result["issues_found"] if issue["type"] == "custom"))

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_check_wildcard_imports(self, mock_run_flake8):
        mock_run_flake8.return_value = [] # Assume Flake8 might also catch this (F403), but testing custom check
        mock_ast_results = {"imports": ["from module import *"], "functions": [], "classes": []}
        result = self.checker.check_python_code("code", ast_analysis_results=mock_ast_results)
        self.assertTrue(any("Import Statement: Contains wildcard import: 'from module import *'." in issue["message"]
                            for issue in result["issues_found"] if issue["type"] == "custom"))

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_check_generic_exceptions(self, mock_run_flake8):
        mock_run_flake8.return_value = [] # Assume Flake8 might also catch this (E722), but testing custom check
        mock_ast_results = {
            "functions": [{"name": "foo", "generic_except_clauses": [3], "args":[], "docstring_exists": True, "body_line_count":1}],
            "classes": [{"name": "Bar", "methods": [{"name": "baz", "generic_except_clauses": [7], "args":["self"], "docstring_exists": True, "body_line_count":1}], "docstring_exists": True, "body_line_count":1}],
            "imports": []
        }
        result = self.checker.check_python_code("code", ast_analysis_results=mock_ast_results)
        custom_issues = {issue["message"] for issue in result["issues_found"] if issue["type"] == "custom"}
        self.assertIn("Function 'foo': Line 3 uses a generic 'except:' or 'except Exception:' clause. Consider using more specific exceptions.", custom_issues)
        self.assertIn("Method 'Bar.baz': Line 7 uses a generic 'except:' or 'except Exception:' clause. Consider using more specific exceptions.", custom_issues)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_clean_code_no_issues(self, mock_run_flake8):
        mock_run_flake8.return_value = [] # Flake8 finds no issues
        code = "def add(a, b):\n  '''Adds two numbers.'''\n  return a + b"
        ast_results = {
            "functions": [{"name": "add", "docstring_exists": True, "args": ["a", "b"], "body_line_count": 1, "generic_except_clauses":[]}],
            "classes": [],
            "imports": ["import math"] # Example import, assuming Flake8 won't flag as unused if it's just 'import math'
        }
        results = self.checker.check_python_code(code, ast_analysis_results=ast_results)
        self.assertEqual(len(results["issues_found"]), 0)
        self.assertEqual(results["custom_issue_count"], 0)
        self.assertEqual(results["flake8_issue_count"], 0)
        self.assertEqual(results["quality_score"], 1.0)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_check_python_code_with_flake8_and_custom_issues(self, mock_run_flake8):
        mock_flake8_issues = [
            {"line": 1, "col": 1, "code": "F401", "message": "'sys' imported but unused"}
        ]
        mock_run_flake8.return_value = mock_flake8_issues

        code_with_todo = "import sys\ndef my_func():\n    # TODO: Implement this\n    pass"
        mock_ast_todo = {
            "functions": [{"name": "my_func", "docstring_exists": False, "body_line_count": 2, "args":[], "generic_except_clauses":[]}],
            "imports": ["import sys"], "classes": []
        }

        result = self.checker.check_python_code(code_with_todo, ast_analysis_results=mock_ast_todo)

        self.assertEqual(result["flake8_issue_count"], 1)
        # Expect TODO and missing docstring as custom issues
        self.assertGreaterEqual(result["custom_issue_count"], 2)

        found_f401 = any(issue.get("code") == "F401" and issue.get("type") == "flake8" for issue in result["issues_found"])
        found_todo = any("TODO" in issue.get("message", "") and issue.get("type") == "custom" for issue in result["issues_found"])
        found_docstring = any("Missing docstring" in issue.get("message", "") and issue.get("type") == "custom" for issue in result["issues_found"])

        self.assertTrue(found_f401, "Flake8 issue F401 not found")
        self.assertTrue(found_todo, "Custom TODO issue not found")
        self.assertTrue(found_docstring, "Custom missing docstring issue not found")

        # Verify quality score calculation based on combined issues
        # Penalties: Flake8 = 0.1 each, Custom = 0.05 each
        expected_score = 1.0 - (1 * 0.1) - (result["custom_issue_count"] * 0.05)
        self.assertAlmostEqual(result["quality_score"], max(0.0, expected_score), places=2)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_check_python_code_flake8_timeout(self, mock_run_flake8):
        mock_run_flake8.return_value = [{"line": 0, "col": 0, "code": "LNT001", "message": "Linter execution timed out."}]
        code = "def test(): pass"
        # Provide minimal AST for this code
        ast_results = {"functions": [{"name":"test", "docstring_exists":True, "body_line_count":1, "args":[], "generic_except_clauses":[]}], "classes": [], "imports": []}
        result = self.checker.check_python_code(code, ast_analysis_results=ast_results)
        self.assertEqual(result["flake8_issue_count"], 1)
        self.assertTrue(any(issue.get("code") == "LNT001" and issue.get("type") == "flake8" for issue in result["issues_found"]))
        # Check that score is penalized for the linter issue
        self.assertLess(result["quality_score"], 1.0)

    @patch('src.core.code_generator.run_flake8_on_code')
    def test_flake8_reports_error_not_file_not_found(self, mock_run_flake8):
        # Simulate Flake8 itself reporting an internal error or misconfiguration
        mock_run_flake8.return_value = [{"line": 0, "col": 0, "code": "E902", "message": "IOError: [Errno 2] No such file or directory: 'stdin'"}]
        code = "def test_func(): pass"
        ast_results = {"functions": [{"name":"test_func", "docstring_exists":True, "body_line_count":1, "args":[], "generic_except_clauses":[]}], "classes": [], "imports": []}
        result = self.checker.check_python_code(code, ast_analysis_results=ast_results)
        self.assertEqual(result["flake8_issue_count"], 1)
        self.assertTrue(any(issue.get("code") == "E902" and issue.get("type") == "flake8" for issue in result["issues_found"]))
        self.assertLess(result["quality_score"], 1.0)


class TestMultiLanguageCodeGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_llm_aggregator = AsyncMock(spec=LLMAggregator)
        self.code_generator = MultiLanguageCodeGenerator(llm_aggregator=self.mock_llm_aggregator)
        # Mock the BestPracticesDatabase for predictable prompt content
        # This mock is now on an instance created by MultiLanguageCodeGenerator
        self.code_generator.best_practices_db = MagicMock(spec=BestPracticesDatabase)
        # Also mock the CodeQualityChecker's run_flake8_on_code method for tests here
        # This is a bit deeper, might be better to mock CodeQualityChecker instance if it was a direct dependency
        # For now, let's assume we can patch run_flake8_on_code globally or on the instance if accessible.
        # If CodeQualityChecker is instantiated inside generate_code, we'd need a different approach.
        # Let's assume it's an instance variable for now: self.code_generator.quality_checker
        # Based on current MultiLanguageCodeGenerator, it instantiates CodeQualityChecker internally, so we patch 'run_flake8_on_code' at its definition.

    def _run_async(self, coro):
        return asyncio.run(coro)

    @patch('src.core.code_generator.run_flake8_on_code') # Patch for Flake8
    def test_generate_python_code_success_with_best_practices_and_checks(self, mock_run_flake8):
        mock_run_flake8.return_value = [ # Simulate some Flake8 issues
            {"line": 1, "col": 1, "code": "F401", "message": "'os' imported but unused"},
        ]

        spec = CodeSpecification(target_language="python", prompt_details="Create a function that adds two numbers and imports os.")
        # Code that will have custom issues (TODO, missing docstring) and Flake8 issue (F401)
        mock_generated_code = "import os\ndef add(a, b):\n  # TODO: Add type hints\n  return a + b"

        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_generated_code), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="cg_resp_1", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        # Configure mock for best_practices_db
        comprehensive_practices = ["Use type hints.", "Keep functions small.", "Handle exceptions gracefully.", "Write good comments."]
        self.code_generator.best_practices_db.get_practices = MagicMock(return_value=comprehensive_practices)

        result = self._run_async(self.code_generator.generate_code(spec))

        self.mock_llm_aggregator.chat_completion.assert_called_once()
        prompt_sent_to_llm = self.mock_llm_aggregator.chat_completion.call_args[0][0].messages[1].content # System prompt is usually index 0, user/task prompt index 1
        for practice in comprehensive_practices:
            self.assertIn(practice, prompt_sent_to_llm)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.generated_code, mock_generated_code)
        self.assertIsNotNone(result.quality_score)

        # Expected issues:
        # Flake8: 1 (F401 from mock_run_flake8)
        # Custom: 2 (Missing docstring for 'add', TODO on line 3)
        self.assertEqual(result.flake8_issue_count, 1)
        self.assertGreaterEqual(result.custom_issue_count, 2) # Could be more if other default checks trigger

        found_f401_flake8 = any(issue["type"] == "flake8" and issue["code"] == "F401" for issue in result.issues_found)
        found_todo_custom = any(issue["type"] == "custom" and "TODO" in issue["message"] for issue in result.issues_found)
        found_docstring_custom = any(issue["type"] == "custom" and "Missing docstring" in issue["message"] for issue in result.issues_found)

        self.assertTrue(found_f401_flake8, "Mocked Flake8 F401 issue not found in results.")
        self.assertTrue(found_todo_custom, "Custom TODO issue not found.")
        self.assertTrue(found_docstring_custom, "Custom missing docstring issue not found.")

        expected_score = 1.0 - (1 * 0.1) - (result.custom_issue_count * 0.05)
        self.assertAlmostEqual(result.quality_score, max(0.0, expected_score), places=2)


    @patch('src.core.code_generator.run_flake8_on_code') # Patch for Flake8
    def test_generate_python_code_with_markdown_fences(self, mock_run_flake8):
        mock_run_flake8.return_value = [] # No Flake8 issues for this simple code
        spec = CodeSpecification(target_language="python", prompt_details="Simple print statement.")
        mock_generated_code_fenced = "```python\nprint('Hello')\n```"
        expected_cleaned_code = "print('Hello')"

        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_generated_code_fenced), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="cg_resp_2", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)
        self.code_generator.best_practices_db.get_practices = MagicMock(return_value=[])


        result = self._run_async(self.code_generator.generate_code(spec))
        self.assertTrue(result.succeeded)
        self.assertEqual(result.generated_code, expected_cleaned_code)
        self.assertEqual(result.flake8_issue_count, 0)
        self.assertEqual(result.custom_issue_count, 0) # Expect no custom issues for "print('Hello')" if AST is minimal

    @patch('src.core.code_generator.run_flake8_on_code') # Patch for Flake8
    def test_generate_python_code_llm_returns_error(self, mock_run_flake8):
        # Flake8 check won't run if LLM fails to produce code-like output.
        # So, mock_run_flake8 doesn't need to be configured with a return value here.
        spec = CodeSpecification(target_language="python", prompt_details="Something unclear.")
        llm_error_message = "ERROR: Request is too ambiguous."

        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=llm_error_message), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="cg_resp_3", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        result = self._run_async(self.code_generator.generate_code(spec))
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.generated_code)
        self.assertEqual(result.error_message, llm_error_message)

    @patch('src.core.code_generator.run_flake8_on_code') # Added to allow mock_run_flake8.assert_not_called()
    def test_generate_unsupported_language(self, mock_run_flake8):
        spec = CodeSpecification(target_language="javascript", prompt_details="Create a JS function.")
        result = self._run_async(self.code_generator.generate_code(spec))
        self.assertFalse(result.succeeded)
        self.assertIsNotNone(result.error_message)
        self.assertTrue("not supported" in result.error_message)
        self.mock_llm_aggregator.chat_completion.assert_not_called()
        mock_run_flake8.assert_not_called() # Flake8 is Python specific


class TestBestPracticesDatabase(unittest.TestCase):
    def setUp(self):
        self.db_path = "dummy_practices.json"
        self.comprehensive_practices_data = {
            "general": [
                "Write modular code.",
                "Keep components loosely coupled."
            ],
            "python": {
                "general_python": [
                    "Follow PEP 8 styling guidelines.",
                    "Write clear and readable code."
                ],
                "error_handling": [
                    "Use specific exception types rather than generic 'Exception'.",
                    "Clean up resources using 'finally' or 'with' statements."
                ],
                "performance": [
                    "Use list comprehensions and generator expressions effectively.",
                    "Profile code to find bottlenecks before optimizing."
                ],
                "security": [
                    "Validate and sanitize user inputs, especially for web applications or external data.",
                    "Be cautious with deserialization (e.g., pickle)."
                ],
                "idiomatic_python": [
                    "Leverage generator functions for lazy evaluation.",
                    "Use 'enumerate' for index and value when iterating."
                ]
            },
            "java": [ # Example for another language
                "Use interfaces to define contracts.",
                "Prefer composition over inheritance where appropriate."
            ]
        }

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_load_practices_success_comprehensive(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.comprehensive_practices_data, f)

        db = BestPracticesDatabase(filepath=self.db_path)
        self.assertIsNotNone(db.practices)
        self.assertIn("Follow PEP 8 styling guidelines.", db.practices["python"]["general_python"])
        self.assertIn("Use list comprehensions and generator expressions effectively.", db.practices["python"]["performance"])
        self.assertIn("Validate and sanitize user inputs, especially for web applications or external data.", db.practices["python"]["security"])
        self.assertIn("Leverage generator functions for lazy evaluation.", db.practices["python"]["idiomatic_python"])
        self.assertIn("Use specific exception types rather than generic 'Exception'.", db.practices["python"]["error_handling"])

    def test_get_practices_for_python_comprehensive(self):
        mock_file_content = json.dumps(self.comprehensive_practices_data)
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            db = BestPracticesDatabase(filepath="mock_path.json")

        python_practices = db.get_practices("python")

        # Check for practices from various Python categories
        self.assertIn("Follow PEP 8 styling guidelines.", python_practices) # python.general_python
        self.assertIn("Clean up resources using 'finally' or 'with' statements.", python_practices) # python.error_handling
        self.assertIn("Profile code to find bottlenecks before optimizing.", python_practices) # python.performance
        self.assertIn("Be cautious with deserialization (e.g., pickle).", python_practices) # python.security
        self.assertIn("Use 'enumerate' for index and value when iterating.", python_practices) # python.idiomatic_python

        # Check for general practices
        self.assertIn("Write modular code.", python_practices)
        self.assertIn("Keep components loosely coupled.", python_practices)

        # Ensure Java-specific practices are not included
        self.assertNotIn("Use interfaces to define contracts.", python_practices)

    def test_get_practices_unsupported_language_with_comprehensive_data(self):
        mock_file_content = json.dumps(self.comprehensive_practices_data)
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            db = BestPracticesDatabase(filepath="mock_path.json")

        js_practices = db.get_practices("javascript") # No JS specific practices in mock data
        expected_general_practices = self.comprehensive_practices_data["general"]
        self.assertEqual(len(js_practices), len(expected_general_practices))
        for gp in expected_general_practices:
            self.assertIn(gp, js_practices)
        self.assertNotIn("Follow PEP 8 styling guidelines.", js_practices) # Python specific


if __name__ == '__main__':
    unittest.main()
