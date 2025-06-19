import structlog
from typing import List, Dict, Any, Optional
import json # For potential parsing if LLM wraps output unexpectedly, though direct code is expected
import ast # Import the ast module
import os # For path joining
import re # For regex-based naming convention checks

from .aggregator import LLMAggregator
from .generation_structures import CodeSpecification, CodeGenerationResult
from ..models import ChatCompletionRequest, Message as OpenHandsMessage # Ensure OpenHandsMessage alias

logger = structlog.get_logger(__name__) # Ensure logger is available if not already defined at module level

# --- Placeholder Components for Code Generation ---
# These are part of the conceptual design from DEVIKA_AI_INTEGRATION.md
# but will not be fully implemented in this phase beyond basic placeholders.

class LanguageAnalyzer: # Keep the base class
    '''Base class for language-specific analysis.'''
    def __init__(self, language_name: str):
        self.language_name = language_name
        # Removed the logger call from base __init__ to avoid duplicate logs if super() is called.
        # logger.info(f"{language_name}Analyzer initialized (placeholder).")

    async def analyze_project_context(self, context: Any) -> Dict[str, Any]:
        logger.warn(f"{self.language_name}Analyzer.analyze_project_context() called - PENDING IMPLEMENTATION")
        return {"language_specific_patterns": f"mock patterns for {self.language_name}"}

class PythonAnalyzer(LanguageAnalyzer):
    '''
    Analyzes Python code structure using the AST module.
    '''
    def __init__(self):
        super().__init__("Python")
        logger.info("PythonAnalyzer initialized (updated for more detailed analysis).")

    def _get_node_body_line_count(self, node: ast.AST) -> Optional[int]:
        '''Helper to calculate the line span of a node's body, approximately.'''
        if not hasattr(node, 'body') or not node.body:
            return 0

        # Body is a list of nodes. First node in body is start. Last node in body is end.
        # However, a simple line span from node.lineno to node.end_lineno includes the signature.
        # For function/class body, it's more accurate to find the start of the first statement in the body
        # and the end of the last statement in the body.

        first_stmt = node.body[0]
        last_stmt = node.body[-1]

        start_line = first_stmt.lineno
        end_line = last_stmt.end_lineno

        if end_line is not None and start_line is not None:
            # Add 1 because line numbers are 1-indexed and we want inclusive count
            # This is an approximation of lines of code in the body.
            # A more precise count would ignore comments and blank lines within the body.
            return (end_line - start_line) + 1
        return None


    def _analyze_function_node(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        '''Helper to analyze a single ast.FunctionDef or ast.AsyncFunctionDef node.'''
        args = [arg.arg for arg in func_node.args.args]
        defaults_count = len(func_node.args.defaults)
        if defaults_count > 0:
            for i in range(defaults_count):
                arg_index = len(args) - defaults_count + i
                args[arg_index] = f"{args[arg_index]}=<default>"

        docstring_node = ast.get_docstring(func_node, clean=False)
        body_line_count = self._get_node_body_line_count(func_node)

        generic_except_clauses = []
        for node_in_body in ast.walk(func_node): # Walk through function body
            if isinstance(node_in_body, ast.ExceptHandler):
                # Check if the type is None (bare except) or Name(id='Exception')
                if node_in_body.type is None or \
                   (isinstance(node_in_body.type, ast.Name) and node_in_body.type.id == 'Exception'):
                    generic_except_clauses.append(node_in_body.lineno)

        return {
            "name": func_node.name,
            "args": args,
            "docstring_exists": docstring_node is not None,
            "docstring_preview": (docstring_node.splitlines()[0][:50] + "..." if docstring_node and docstring_node.splitlines() else "") if docstring_node else "",
            "start_lineno": func_node.lineno,
            "end_lineno": func_node.end_lineno, # end_lineno is available on Python 3.8+
            "body_line_count": body_line_count,
            "generic_except_clauses": sorted(list(set(generic_except_clauses))) # Unique, sorted line numbers
        }

    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        logger.info("Analyzing Python code structure (enhanced detail)", code_length=len(code))
        results: Dict[str, Any] = {
            "imports": [],
            "functions": [],
            "classes": [],
            "error": None # Initialize error key
        }

        if not code.strip():

        Args:
            code: A string containing the Python code to analyze.

        Returns:
            A dictionary with extracted structural information, or an error message.
            Example: {
                "imports": ["module1", "module2.sub_module as alias"],
                "functions": [{"name": "func1", "args": ["a", "b=1"], "docstring": "..."}],
                "classes": [{"name": "MyClass", "methods": [{"name": "method1", "args": ["self", "x"]}]}]
            }
        '''
            logger.warn("Attempted to analyze empty Python code string.")
            results["error"] = "Input code string is empty."
            return results

        try:
            tree = ast.parse(code)

            # Extract imports (can walk the whole tree for this)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        results["imports"].append(alias.name + (f" as {alias.asname}" if alias.asname else ""))
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or "."
                    for alias in node.names:
                        results["imports"].append(f"from {module_name} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            results["imports"] = sorted(list(set(results["imports"]))) # Unique, sorted

            # Extract top-level functions and classes
            for node in tree.body: # Iterate only top-level nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    results["functions"].append(self._analyze_function_node(node)) # type: ignore

                elif isinstance(node, ast.ClassDef):
                    class_methods = []
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)): # Method
                            class_methods.append(self._analyze_function_node(item)) # type: ignore

                    docstring_node = ast.get_docstring(node, clean=False)
                    body_line_count = self._get_node_body_line_count(node)

                    results["classes"].append({
                        "name": node.name,
                        "methods": class_methods,
                        "docstring_exists": docstring_node is not None,
                        "docstring_preview": (docstring_node.splitlines()[0][:50] + "..." if docstring_node and docstring_node.splitlines() else "") if docstring_node else "",
                        "start_lineno": node.lineno,
                        "end_lineno": node.end_lineno, # Python 3.8+
                        "body_line_count": body_line_count
                    })

            logger.info("Python code structure analysis complete (enhanced detail).",
                        num_imports=len(results["imports"]),
                        num_functions=len(results["functions"]),
                        num_classes=len(results["classes"]))

        except SyntaxError as e:
            logger.error("Syntax error during Python code analysis", error_msg=e.msg, lineno=e.lineno, offset=e.offset, text_line=e.text, code_snippet=code[:100]) # exc_info=True too verbose here
            results["error"] = f"SyntaxError: {e.msg} on line {e.lineno} offset {e.offset}"
            # results["code_snippet_on_error"] = e.text # This might be too much for the dict
        except Exception as e:
            logger.error("Unexpected error during Python code analysis", error=str(e), exc_info=True)
            results["error"] = f"Unexpected error: {str(e)}"

        return results

# Add other placeholders if desired, e.g., JavaScriptAnalyzer, JavaAnalyzer...

class BestPracticesDatabase:
    '''
    Loads and provides language-specific best practices, initially for Python.
    '''
    def __init__(self, practices_file_path: str = "config/python_best_practices.json"):
        self.practices_file_path = practices_file_path
        self.practices: Dict[str, Dict[str, List[str]]] = {} # language -> category -> [practice_string]
        self._load_practices()
        logger.info("BestPracticesDatabase initialized.", practices_file=practices_file_path, loaded_languages=list(self.practices.keys()))

    def _load_practices(self):
        '''Loads practices from the specified JSON file.'''
        try:
            # Construct path relative to the project root or a known config directory
            # For simplicity, assume 'config/' is at the same level as 'src/' or accessible.
            # If this script is in src/core, config/ is ../../config relative to it.
            # A more robust way might involve getting project root from settings or env.
            # For now, let's assume the path is valid as given for now.

            # Correct path assuming execution from project root where 'config' and 'src' are siblings.
            # If running from src/core, this path needs adjustment or a more robust path resolution.
            # Let's assume the file path is valid as given for now.

            if not os.path.exists(self.practices_file_path):
                logger.error(f"Best practices file not found: {self.practices_file_path}. No practices will be loaded.")
                return

            with open(self.practices_file_path, 'r') as f:
                # For now, structure is directly Python practices under categories
                python_practices = json.load(f)
                self.practices["python"] = python_practices # Store under "python" key
            logger.info(f"Successfully loaded Python best practices from {self.practices_file_path} for 'python' language.")

        except FileNotFoundError:
            logger.error(f"Best practices file not found: {self.practices_file_path}", exc_info=True)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from best practices file: {self.practices_file_path}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading best practices from {self.practices_file_path}", error=str(e), exc_info=True)

    def get_practices(self, language: str, context_categories: Optional[List[str]] = None) -> List[str]:
        '''
        Retrieves best practices for a given language and optional categories.

        Args:
            language: The programming language (e.g., "python").
            context_categories: Optional list of categories (e.g., ["general", "functions"]).
                                If None, all practices for the language are returned.

        Returns:
            A list of best practice strings.
        '''
        lang_practices = self.practices.get(language.lower())
        if not lang_practices:
            logger.warn("No best practices found for language", language=language)
            return []

        relevant_practices: List[str] = []
        if context_categories:
            for category in context_categories:
                if category in lang_practices:
                    relevant_practices.extend(lang_practices[category])
                else:
                    logger.debug(f"Category '{category}' not found in practices for language '{language}'.")
            # Add general practices if not explicitly requested but others were, or if it's a common fallback.
            if "general" not in context_categories and "general" in lang_practices:
                # Decide if general should always be added or only if no specific categories match.
                # For now, let's add general if any specific category was requested.
                # A more refined logic could be: if relevant_practices is empty after specific categories, add general.
                pass # Not adding general by default if specific categories are given.
        else: # No categories specified, return all for the language
            for category_practices in lang_practices.values():
                relevant_practices.extend(category_practices)

        # A simple way to always include 'general' if specific categories don't yield much,
        # or if 'general' is always desired alongside specific ones.
        # For now, let's refine: if specific categories are given, only return those.
        # If specific categories result in empty, maybe then add general.
        # Or, user must explicitly ask for "general".
        # Current logic: only returns practices from specified categories. If context_categories is None, returns all.

        if not relevant_practices and not context_categories and "general" in lang_practices:
            # If no categories specified and nothing found (e.g. malformed JSON), at least try general
            relevant_practices.extend(lang_practices["general"])


        logger.debug("Retrieved best practices", language=language, categories_requested=context_categories, num_practices_retrieved=len(relevant_practices))
        return list(set(relevant_practices)) # Return unique practices

class CodeQualityChecker:
    '''
    Performs basic custom quality checks on Python code.
    '''
    def __init__(self, max_line_length: int = 100):
        self.max_line_length = max_line_length
        logger.info(f"CodeQualityChecker initialized (max_line_length={max_line_length}).")

    def check_python_code(self, code: str, ast_analysis_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        '''
        Performs basic custom quality checks on the provided Python code string.

        Args:
            code: The Python code string to check.
            ast_analysis_results: Optional output from PythonAnalyzer.analyze_code_structure.
                                  Used for checks like docstring presence.

        Returns:
            A dictionary with 'issues_found' (List[str]) and 'quality_score' (float).
        '''
        logger.info("Checking Python code quality", code_length=len(code))
        issues_found: List[str] = []
        lines = code.splitlines()
        num_lines = len(lines)

        if num_lines == 0:
            logger.warn("Attempted to check quality of empty Python code string.")
            return {"issues_found": ["Code is empty."], "quality_score": 0.0}

        # Check 1: "TODO" or "FIXME" comments
        for i, line in enumerate(lines):
            line_num = i + 1
            if "TODO" in line or "FIXME" in line:
                issues_found.append(f"Line {line_num}: Contains 'TODO' or 'FIXME'.")

        # Check 2: Overly long lines
        for i, line in enumerate(lines):
            line_num = i + 1
            if len(line) > self.max_line_length:
                issues_found.append(f"Line {line_num}: Exceeds max line length of {self.max_line_length} (length: {len(line)}).")

        # Check 3: Basic docstring checks using ast_analysis_results (if provided)
        if ast_analysis_results:
            if "functions" in ast_analysis_results:
                for func_info in ast_analysis_results["functions"]:
                    if not func_info.get("docstring_exists"):
                        issues_found.append(f"Function '{func_info.get('name', 'N/A')}': Missing docstring.")

            if "classes" in ast_analysis_results:
                for class_info in ast_analysis_results["classes"]:
                    if not class_info.get("docstring_exists"):
                        issues_found.append(f"Class '{class_info.get('name', 'N/A')}': Missing docstring.")
                    # Check methods within classes for docstrings
                    for method_info in class_info.get("methods", []):
                        # Skip __init__ for docstring check for simplicity, or apply different rule
                        if method_info.get("name") == "__init__" and not method_info.get("docstring_exists"):
                             # Could have a softer warning or ignore for __init__
                             pass # issues_found.append(f"Method '{class_info.get('name')}.__init__': Missing docstring (optional but good practice).")
                        elif method_info.get("name") != "__init__" and not method_info.get("docstring_exists"):
                             issues_found.append(f"Method '{class_info.get('name', 'N/A')}.{method_info.get('name', 'N/A')}': Missing docstring.")
        else:
            logger.debug("AST analysis results not provided, skipping docstring checks.")

        # Calculate a conceptual quality score
        # Score starts at 1.0 and decreases for each issue. Max penalty per line for multiple issues on one line is avoided by this simple count.
        # This is a very naive scoring mechanism.
        quality_penalty = len(issues_found) * 0.1 # Penalize 0.1 for each issue
        quality_score = max(0.0, 1.0 - quality_penalty)

        # Alternative scoring: based on ratio of issues to lines
        # if num_lines > 0:
        #     quality_score = max(0.0, 1.0 - (len(issues_found) / num_lines))
        # else:
        #     quality_score = 0.0 if issues_found else 1.0


        logger.info("Python code quality check complete.", num_issues=len(issues_found), quality_score=quality_score)
        return {
            "issues_found": issues_found,
            "quality_score": quality_score
        }

# --- Main Code Generator Class ---

class MultiLanguageCodeGenerator:
    '''
    Generates code in multiple languages, guided by specifications and context.
    Integrates Python code analysis, quality checking, and best practices.
    '''

    def __init__(self,
                 llm_aggregator: LLMAggregator
                 ):
        self.llm_aggregator = llm_aggregator
        self.python_analyzer = PythonAnalyzer()
        self.best_practices_db = BestPracticesDatabase() # Instantiate BestPracticesDatabase
        self.code_quality_checker = CodeQualityChecker()
        logger.info("MultiLanguageCodeGenerator initialized with PythonAnalyzer, BestPracticesDatabase, and CodeQualityChecker.")

    async def generate_code(self, spec: CodeSpecification) -> CodeGenerationResult:
        '''
        Generates code based on the provided specification using an LLM.
        For Python, it also retrieves best practices, analyzes structure, and checks basic quality.

        Args:
            spec: A CodeSpecification object detailing what code to generate.

        Returns:
            A CodeGenerationResult object.
        '''
        logger.info("Starting code generation with best practices, analysis, and quality check",
                    spec_id=spec.spec_id, target_language=spec.target_language)

        if spec.target_language.lower() != "python":
            # ... (existing non-python language handling) ...
            logger.warn("Unsupported language for detailed generation pipeline",
                        language=spec.target_language, spec_id=spec.spec_id)
            return CodeGenerationResult(
                specification_id=spec.spec_id,
                language=spec.target_language,
                error_message=f"Language '{spec.target_language}' not supported for full analysis/generation in this version."
            )

        # --- Retrieve Best Practices for Python ---
        # Determine relevant categories for best practices.
        # For now, let's request "general", "functions", and "error_handling" by default for Python.
        # This could be made more dynamic based on spec.prompt_details in the future.
        practice_categories = ["general", "functions", "classes", "error_handling", "imports"]
        retrieved_practices = self.best_practices_db.get_practices("python", context_categories=practice_categories)

        logger.debug("Retrieved best practices for Python prompt", num_practices=len(retrieved_practices), categories=practice_categories)
        # --- End Retrieve Best Practices ---

        # Construct the prompt for Python code generation
        prompt_parts = [f"Please generate Python code based on the following specification."]
        prompt_parts.append(f"Core Task/Logic: {spec.prompt_details}")

        if spec.context_summary:
            prompt_parts.append(f"Relevant Context: {spec.context_summary}")

        if spec.examples:
            example_str = "\n".join([f"- Input: {ex.get('input', 'N/A')}, Output: {ex.get('output', 'N/A')}" for ex in spec.examples])
            prompt_parts.append(f"Examples of desired behavior:\n{example_str}")

        if spec.constraints:
            constraint_str = "\n".join([f"- {c}" for c in spec.constraints])
            prompt_parts.append(f"Constraints or specific requirements:\n{constraint_str}")

        # --- Add Best Practices to Prompt ---
        if retrieved_practices:
            practices_str = "\n".join([f"- {p}" for p in retrieved_practices])
            prompt_parts.append(f"Please adhere to the following Python best practices where applicable:\n{practices_str}")
        # --- End Add Best Practices ---

        prompt_parts.append("\nEnsure the output contains ONLY the Python code. Do not include any explanations, introductions, or markdown fences like ```python ... ``` unless the fence is part of a multi-line string within the code itself.")
        prompt_parts.append("If the request is unclear or cannot be fulfilled as Python code, please respond with an error message prefixed with 'ERROR:'.")

        final_prompt = "\n\n".join(prompt_parts)

        messages = [
            OpenHandsMessage(role="system", content="You are an expert Python code generation AI. Your output should be only the raw Python code as requested, adhering to provided specifications and best practices. If you cannot fulfill the request, respond with 'ERROR: Your reason for failure.'"),
            OpenHandsMessage(role="user", content=final_prompt)
        ]

        request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.25) # type: ignore

        generated_code_content: Optional[str] = None
        error_message: Optional[str] = None
        analysis_results: Optional[Dict[str, Any]] = None
        quality_check_results: Optional[Dict[str, Any]] = None

        try:
            logger.debug("Sending Python code generation request to LLM", spec_id=spec.spec_id)
            response = await self.llm_aggregator.chat_completion(request)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                llm_output = response.choices[0].message.content.strip()
                logger.debug("Received LLM response for code generation", spec_id=spec.spec_id, response_length=len(llm_output))

                if llm_output.startswith("ERROR:"):
                    error_message = llm_output
                    logger.warn("LLM indicated an error in generating code", spec_id=spec.spec_id, llm_error=error_message)
                else:
                    # Clean markdown fences
                    if llm_output.startswith("```python"):
                        llm_output = llm_output[len("```python"):].strip()
                        if llm_output.endswith("```"):
                            llm_output = llm_output[:-3].strip()
                    elif llm_output.startswith("```"):
                        llm_output = llm_output[3:].strip()
                        if llm_output.endswith("```"):
                             llm_output = llm_output[:-3].strip()

                    generated_code_content = llm_output
                    logger.info("Successfully generated Python code (pre-analysis)", spec_id=spec.spec_id, code_length=len(generated_code_content)) # Keep this log as is or make more generic

                    if generated_code_content: # Proceed with analysis only if code was generated
                        logger.info("Analyzing generated Python code structure...", spec_id=spec.spec_id)
                        analysis_results = self.python_analyzer.analyze_code_structure(generated_code_content)
                        logger.info("Code structure analysis complete.", spec_id=spec.spec_id, analysis_keys=list(analysis_results.keys()) if analysis_results else None)
                        if analysis_results.get("error"):
                            logger.warn("Error during AST analysis of generated code", spec_id=spec.spec_id, analysis_error=analysis_results.get("error"))

                        logger.info("Checking quality of generated Python code...", spec_id=spec.spec_id)
                        quality_check_results = self.code_quality_checker.check_python_code(generated_code_content, ast_analysis_results=analysis_results)
                        logger.info("Code quality check complete.", spec_id=spec.spec_id, issues_found=len(quality_check_results.get("issues_found",[])), score=quality_check_results.get("quality_score"))
            else: # LLM response was empty or malformed
                error_message = "LLM response was empty or malformed for code generation."

        except Exception as e: # Catch-all for other errors during the process
            error_message = f"An unexpected error occurred during code generation pipeline: {str(e)}"
            logger.error("Code generation pipeline failed", spec_id=spec.spec_id, error=str(e), exc_info=True)

        return CodeGenerationResult(
            specification_id=spec.spec_id,
            generated_code=generated_code_content,
            language="python" if generated_code_content else spec.target_language,
            error_message=error_message,
            # Populate from quality_check_results
            issues_found=quality_check_results.get("issues_found", []) if quality_check_results else [],
            quality_score=quality_check_results.get("quality_score") if quality_check_results else None
            # Add analysis_output to CodeGenerationResult if desired in its definition
            # analysis_output=analysis_results
        )
