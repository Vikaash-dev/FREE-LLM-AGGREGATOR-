import structlog
from typing import List, Dict, Any, Optional
import json # For potential parsing if LLM wraps output unexpectedly, though direct code is expected
import ast # Import the ast module

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
        logger.info("PythonAnalyzer initialized.")

    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        '''
        Parses Python code using the AST module to extract structural information.

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
        logger.info("Analyzing Python code structure", code_length=len(code))
        results: Dict[str, Any] = {
            "imports": [],
            "functions": [],
            "classes": []
        }

        if not code.strip():
            logger.warn("Attempted to analyze empty Python code string.")
            results["error"] = "Input code string is empty."
            return results

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        results["imports"].append(alias.name + (f" as {alias.asname}" if alias.asname else ""))
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or "." # Handle relative imports starting with '.'
                    for alias in node.names:
                        results["imports"].append(f"from {module_name} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))

                # Consider only top-level functions and classes for now
                # To do this, iterate tree.body instead of ast.walk for top-level only

            # Reset and iterate for top-level functions/classes specifically
            results["functions"] = []
            results["classes"] = []

            for node in tree.body: # Iterate only top-level nodes
                if isinstance(node, ast.FunctionDef):
                    args = [arg.arg for arg in node.args.args]
                    # Represent default arguments simply for now
                    defaults_count = len(node.args.defaults)
                    if defaults_count > 0:
                        for i in range(defaults_count):
                            arg_index = len(args) - defaults_count + i
                            # This is a simplified representation, actual default value is complex
                            args[arg_index] = f"{args[arg_index]}=<default>"

                    docstring_node = ast.get_docstring(node, clean=False) # Get raw docstring
                    results["functions"].append({
                        "name": node.name,
                        "args": args,
                        "docstring_exists": docstring_node is not None,
                        "docstring_preview": (docstring_node.splitlines()[0] if docstring_node else "")[:50] + "..." if docstring_node else ""
                    })
                elif isinstance(node, ast.ClassDef):
                    class_methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef): # Method
                            method_args = [arg.arg for arg in item.args.args]
                            method_docstring_node = ast.get_docstring(item, clean=False)
                            class_methods.append({
                                "name": item.name,
                                "args": method_args,
                                "docstring_exists": method_docstring_node is not None
                            })
                    class_docstring_node = ast.get_docstring(node, clean=False)
                    results["classes"].append({
                        "name": node.name,
                        "methods": class_methods,
                        "docstring_exists": class_docstring_node is not None,
                        "docstring_preview": (class_docstring_node.splitlines()[0] if class_docstring_node else "")[:50] + "..." if class_docstring_node else ""
                    })

            logger.info("Python code structure analysis complete.",
                        num_imports=len(results["imports"]),
                        num_functions=len(results["functions"]),
                        num_classes=len(results["classes"]))

        except SyntaxError as e:
            logger.error("Syntax error during Python code analysis", error=str(e), code_snippet=code[:100], exc_info=True)
            results["error"] = f"SyntaxError: {e.msg} on line {e.lineno} offset {e.offset}"
            results["code_snippet_on_error"] = e.text
        except Exception as e:
            logger.error("Unexpected error during Python code analysis", error=str(e), exc_info=True)
            results["error"] = f"Unexpected error: {str(e)}"

        return results

# Add other placeholders if desired, e.g., JavaScriptAnalyzer, JavaAnalyzer...

class BestPracticesDatabase:
    '''Placeholder for a system that provides language/framework specific best practices.'''
    def __init__(self):
        logger.info("BestPracticesDatabase initialized (placeholder).")

    async def get_practices(self, language: str, project_type: Optional[str] = None, framework: Optional[str] = None) -> List[str]:
        logger.warn("BestPracticesDatabase.get_practices() called - PENDING IMPLEMENTATION")
        return [f"Mock best practice 1 for {language}", f"Mock best practice 2 for {language}"]

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
    Integrates Python code analysis and quality checking.
    '''

    def __init__(self,
                 llm_aggregator: LLMAggregator,
                 # python_analyzer: Optional[PythonAnalyzer] = None, # Replaced by direct instantiation
                 # best_practices_db: Optional[BestPracticesDatabase] = None, # Still placeholder
                 # quality_checker: Optional[CodeQualityChecker] = None # Replaced by direct instantiation
                 ):
        self.llm_aggregator = llm_aggregator
        self.python_analyzer = PythonAnalyzer() # Instantiate PythonAnalyzer
        self.best_practices_db = BestPracticesDatabase() # Keep as placeholder for now
        self.code_quality_checker = CodeQualityChecker() # Instantiate CodeQualityChecker
        logger.info("MultiLanguageCodeGenerator initialized with PythonAnalyzer and CodeQualityChecker.")

    async def generate_code(self, spec: CodeSpecification) -> CodeGenerationResult:
        '''
        Generates code based on the provided specification using an LLM.
        For Python, it also analyzes structure and checks basic quality.

        Args:
            spec: A CodeSpecification object detailing what code to generate.

        Returns:
            A CodeGenerationResult object.
        '''
        logger.info("Starting code generation with analysis and quality check",
                    spec_id=spec.spec_id, target_language=spec.target_language)

        if spec.target_language.lower() != "python":
            logger.warn("Unsupported language for detailed generation pipeline",
                        language=spec.target_language, spec_id=spec.spec_id)
            return CodeGenerationResult(
                specification_id=spec.spec_id,
                language=spec.target_language,
                error_message=f"Language '{spec.target_language}' not supported for full analysis/generation in this version."
            )

        # Construct the prompt for Python code generation (as implemented in Step 3)
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

        prompt_parts.append("\nEnsure the output contains ONLY the Python code. Do not include any explanations, introductions, or markdown fences like ```python ... ``` unless the fence is part of a multi-line string within the code itself.")
        prompt_parts.append("If the request is unclear or cannot be fulfilled as Python code, please respond with an error message prefixed with 'ERROR:'.")

        final_prompt = "\n\n".join(prompt_parts)

        messages = [
            OpenHandsMessage(role="system", content="You are an expert Python code generation AI. Your output should be only the raw Python code as requested. If you cannot fulfill the request, respond with 'ERROR: Your reason for failure.'"),
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
                    logger.info("Successfully generated Python code (pre-analysis)", spec_id=spec.spec_id, code_length=len(generated_code_content))

                    # --- Integration of Analyzer and Checker ---
                    if generated_code_content:
                        logger.info("Analyzing generated Python code structure...", spec_id=spec.spec_id)
                        analysis_results = self.python_analyzer.analyze_code_structure(generated_code_content)
                        logger.info("Code structure analysis complete.", spec_id=spec.spec_id, analysis_keys=list(analysis_results.keys()) if analysis_results else None)

                        # Log analysis errors if any, but don't stop quality check necessarily
                        if analysis_results.get("error"):
                            logger.warn("Error during AST analysis of generated code", spec_id=spec.spec_id, analysis_error=analysis_results.get("error"))

                        logger.info("Checking quality of generated Python code...", spec_id=spec.spec_id)
                        quality_check_results = self.code_quality_checker.check_python_code(generated_code_content, ast_analysis_results=analysis_results)
                        logger.info("Code quality check complete.", spec_id=spec.spec_id, issues_found=len(quality_check_results.get("issues_found",[])), score=quality_check_results.get("quality_score"))
                    # --- End Integration ---
            else:
                error_message = "LLM response was empty or malformed."
                logger.warn(error_message, spec_id=spec.spec_id, llm_response=response)

        except Exception as e:
            error_message = f"An unexpected error occurred during LLM call or post-processing for code generation: {str(e)}"
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
