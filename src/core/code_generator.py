import structlog
from typing import List, Dict, Any, Optional
import json # For potential parsing if LLM wraps output unexpectedly, though direct code is expected
import ast # Import the ast module
import os # For path joining
import re # For regex-based naming convention checks
import subprocess # For running flake8
import tempfile # For temporary file
import sys # For sys.executable
import xml.etree.ElementTree as ET # For parsing cppcheck XML output

from .aggregator import LLMAggregator
from .generation_structures import CodeSpecification, CodeGenerationResult
from ..models import ChatCompletionRequest, Message as OpenHandsMessage # Ensure OpenHandsMessage alias

logger = structlog.get_logger(__name__) # Ensure logger is available if not already defined at module level

# --- Helper function for Flake8 ---
def run_flake8_on_code(code_string: str) -> List[Dict[str, Any]]:
    '''
    Runs flake8 on a given Python code string and parses its output.

    Args:
        code_string: The Python code to check.

    Returns:
        A list of dictionaries, where each dictionary represents a flake8 issue.
        Example: [{'line': 1, 'col': 1, 'code': 'F401', 'message': "'module' imported but unused"}]
        Returns an empty list if no issues are found or if flake8 fails to run.
    '''
    issues_found: List[Dict[str, Any]] = []
    if not code_string.strip():
        logger.debug("Skipping flake8 on empty code string.")
        return issues_found

    # Flake8 needs a file to operate on.
    # We use a temporary file with a .py extension.
    # tempfile.NamedTemporaryFile can be tricky with subprocesses on Windows (file locking).
    # A more robust approach is to create a temp dir and a file within it.

    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(tmp_dir, "temp_code_to_lint.py")

        with open(tmp_file_path, "w", encoding="utf-8") as f:
            f.write(code_string)

        logger.debug(f"Running flake8 on temporary file: {tmp_file_path}")

        # Common flake8 invocation. Adjust options as needed.
        # --select=E,W,F to get errors, warnings, and fatal errors.
        # --format='%(row)d,%(col)d,%(code)s,%(text)s' for easy parsing.
        # Add --isolated to ignore project/user flake8 config for consistent behavior.
        flake8_cmd = [
            sys.executable, # Use the current Python interpreter to run flake8 module
            "-m", "flake8",
            tmp_file_path,
            "--format=%(row)d,%(col)d,%(code)s,%(text)s",
            "--isolated"
        ]

        # Set a timeout for flake8 process, e.g., 10 seconds
        process_timeout = 10

        completed_process = subprocess.run(
            flake8_cmd,
            capture_output=True,
            text=True,
            timeout=process_timeout,
            check=False # Don't raise exception for non-zero exit code (flake8 exits non-zero if issues found)
        )

        if completed_process.stderr:
            logger.warn("Flake8 produced stderr output", stderr=completed_process.stderr.strip())

        if completed_process.stdout:
            output_lines = completed_process.stdout.strip().splitlines()
            for line in output_lines:
                parts = line.split(',', 3) # Split into 4 parts: row, col, code, text
                if len(parts) == 4:
                    try:
                        issues_found.append({
                            "line": int(parts[0]),
                            "col": int(parts[1]),
                            "code": parts[2].strip(),
                            "message": parts[3].strip()
                        })
                    except ValueError:
                        logger.warn("Failed to parse flake8 output line", line=line, reason="ValueError converting line/col to int")
                elif line.strip(): # Non-empty line that doesn't match format
                    logger.warn("Unparseable flake8 output line", line=line)
            logger.debug(f"Flake8 found {len(issues_found)} issues.")
        else:
            logger.debug("Flake8 found no issues (stdout was empty).")
            if completed_process.returncode != 0 and not completed_process.stderr: # No stdout, but error code
                 logger.warn("Flake8 exited with non-zero code but no stdout/stderr.", return_code=completed_process.returncode)


    except subprocess.TimeoutExpired:
        logger.error(f"Flake8 execution timed out after {process_timeout} seconds.", file_path=tmp_file_path if 'tmp_file_path' in locals() else "N/A")
        issues_found.append({"line": 0, "col": 0, "code": "LNT001", "message": "Linter execution timed out."}) # Custom code for timeout
    except FileNotFoundError: # If flake8 or python executable is not found
        logger.error("Flake8 command not found. Ensure flake8 is installed and accessible.", exc_info=True)
        issues_found.append({"line": 0, "col": 0, "code": "LNT002", "message": "Linter (flake8) not found or not executable."})
    except Exception as e:
        logger.error("An unexpected error occurred while running flake8", error=str(e), exc_info=True)
        issues_found.append({"line": 0, "col": 0, "code": "LNT003", "message": f"Unexpected error during linting: {str(e)}"})
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                # Clean up: Remove temporary file and directory
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                os.rmdir(tmp_dir)
                logger.debug(f"Successfully cleaned up temporary directory: {tmp_dir}")
            except Exception as e_cleanup:
                logger.error(f"Error cleaning up temporary directory {tmp_dir}", error=str(e_cleanup), exc_info=True)

    return issues_found

# --- Helper function for Cppcheck ---
def run_cppcheck_on_code(code_string: str) -> List[Dict[str, Any]]:
    '''
    Runs cppcheck on a given C code string and parses its XML output.

    Args:
        code_string: The C code to check.

    Returns:
        A list of dictionaries, where each dictionary represents a cppcheck issue.
        Example: [{'file': 'temp_code.c', 'line': 5, 'id': 'unreadVariable',
                   'severity': 'style', 'message': 'Variable "x" is assigned a value that is never used.',
                   'verbose_message': 'Variable "x" is assigned a value that is never used.'}]
        Returns an empty list if no issues are found or if cppcheck fails.
    '''
    issues_found: List[Dict[str, Any]] = []
    if not code_string.strip():
        logger.debug("Skipping cppcheck on empty code string.")
        return issues_found

    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp()
        # cppcheck often works better if the file has a .c extension
        tmp_file_path = os.path.join(tmp_dir, "temp_code_to_check.c")

        with open(tmp_file_path, "w", encoding="utf-8") as f:
            f.write(code_string)

        logger.debug(f"Running cppcheck on temporary file: {tmp_file_path}")

        # cppcheck command. --enable=all for more checks. --xml for parseable output.
        # --quiet suppresses informational messages from cppcheck itself to stderr.
        cppcheck_cmd = [
            "cppcheck", # Assumes cppcheck is in PATH
            "--enable=all",
            "--xml",
            "--quiet",
            tmp_file_path
        ]

        process_timeout = 15 # Cppcheck can sometimes be slower than flake8

        completed_process = subprocess.run(
            cppcheck_cmd,
            capture_output=True, # Capture stderr for XML output
            text=True,
            timeout=process_timeout,
            check=False
        )

        # Cppcheck outputs XML to stderr when --xml is used.
        if completed_process.stderr:
            xml_output = completed_process.stderr.strip()
            logger.debug("Raw cppcheck XML output received", length=len(xml_output))
            try:
                root = ET.fromstring(xml_output)
                for error_element in root.findall(".//error"):
                    issue = {
                        "file": error_element.get("file", tmp_file_path.split(os.sep)[-1]), # Report relative file name
                        "line": int(error_element.get("line", "0")),
                        "id": error_element.get("id"),
                        "severity": error_element.get("severity"), # e.g., error, warning, style, performance, portability
                        "message": error_element.get("msg"),
                        "verbose_message": error_element.get("verbose")
                    }
                    # Location elements might provide more detail if needed
                    # for loc_element in error_element.findall("location"):
                    #    pass # process if needed
                    issues_found.append(issue)
                logger.debug(f"Cppcheck found {len(issues_found)} issues from XML.")
            except ET.ParseError as e_xml:
                logger.error("Failed to parse cppcheck XML output.", xml_snippet=xml_output[:500], error=str(e_xml), exc_info=True)
                issues_found.append({"line": 0, "id": "CPPCHECK_XML_ERROR", "severity": "error", "message": "Failed to parse cppcheck XML output."})
            except Exception as e_parse: # Catch other parsing errors
                logger.error("Unexpected error parsing cppcheck XML.", error=str(e_parse), xml_snippet=xml_output[:500], exc_info=True)
                issues_found.append({"line": 0, "id": "CPPCHECK_PARSE_UNEXPECTED", "severity": "error", "message": f"Unexpected XML parsing error: {str(e_parse)}"})

        elif completed_process.stdout: # Should be empty with --quiet, but log if not
            logger.warn("Cppcheck produced unexpected stdout output (expected XML on stderr with --quiet)", stdout=completed_process.stdout.strip())

        if completed_process.returncode != 0 and not issues_found and not completed_process.stderr.strip() : # No specific issues found via XML but cppcheck had an error
             logger.warn("Cppcheck exited with non-zero code but no XML issues parsed from stderr.", return_code=completed_process.returncode)
             issues_found.append({"line": 0, "id": f"CPPCHECK_RC_{completed_process.returncode}", "severity": "error", "message": f"Cppcheck exited with code {completed_process.returncode} without specific error details in XML."})


    except subprocess.TimeoutExpired:
        logger.error(f"Cppcheck execution timed out after {process_timeout} seconds.", file_path=tmp_file_path if 'tmp_file_path' in locals() else "N/A")
        issues_found.append({"line": 0, "id": "LNTCPP001", "severity": "error", "message": "Linter (cppcheck) execution timed out."})
    except FileNotFoundError:
        logger.error("cppcheck command not found. Ensure cppcheck is installed and in PATH.", exc_info=False) # No need for full exc_info if it's just not found
        issues_found.append({"line": 0, "id": "LNTCPP002", "severity": "error", "message": "Linter (cppcheck) not found or not executable."})
    except Exception as e:
        logger.error("An unexpected error occurred while running cppcheck", error=str(e), exc_info=True)
        issues_found.append({"line": 0, "id": "LNTCPP003", "severity": "error", "message": f"Unexpected error during C linting: {str(e)}"})
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                os.rmdir(tmp_dir)
                logger.debug(f"Successfully cleaned up temporary directory for cppcheck: {tmp_dir}")
            except Exception as e_cleanup:
                logger.error(f"Error cleaning up temporary directory for cppcheck {tmp_dir}", error=str(e_cleanup), exc_info=True)

    return issues_found

# --- Placeholder Components for Code Generation ---
# These are part of the conceptual design from DEVIKA_AI_INTEGRATION.md
# but will not be fully implemented in this phase beyond basic placeholders.

class LanguageAnalyzer: # Keep the base class
    '''Base class for language-specific analysis.'''
    def __init__(self, language_name: str):
        self.language_name = language_name
        # Removed the logger call from base __init__ to avoid duplicate logs if super() is called.
        # logger.info(f"{language_name}Analyzer initialized (placeholder).")

    async def analyze_project_context(self, context: Any) -> Dict[str, Any]: # Keep async for future
        logger.warn(f"{self.language_name}Analyzer.analyze_project_context() called - PENDING IMPLEMENTATION")
        return {"language_specific_patterns": f"mock patterns for {self.language_name}"}

    def analyze_code_structure(self, code: str) -> Dict[str, Any]: # Make base sync for now
        logger.warn(f"{self.language_name}Analyzer.analyze_code_structure() called - Base method, PENDING IMPLEMENTATION for specific languages.")
        return {"error": "Base analyzer does not implement structure analysis."}


class PythonAnalyzer(LanguageAnalyzer):
    '''
    Analyzes Python code structure using the AST module.
    '''
    # ... (existing PythonAnalyzer implementation) ...
    pass


class CAnalyzer(LanguageAnalyzer):
    '''
    Analyzes C code structure using regex and basic string parsing.
    (Note: This is a simplified approach and not a full C parser.)
    '''
    def __init__(self):
        super().__init__("C")
        logger.info("CAnalyzer initialized (using regex-based analysis).")

    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        logger.info("Analyzing C code structure (regex-based)", code_length=len(code))
        results: Dict[str, Any] = {
            "includes": [],
            "function_signatures": [], # Store as strings for simplicity
            "struct_declarations": [], # Store names
            "enum_declarations": [],   # Store names
            "typedef_declarations": [], # Store new type names
            "error": None
        }

        if not code.strip():
            logger.warn("Attempted to analyze empty C code string.")
            results["error"] = "Input code string is empty."
            return results

        try:
            # 1. Extract #include statements
            # Matches #include <stdio.h> or #include "myheader.h"
            include_pattern = r'^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]'
            for line in code.splitlines():
                match = re.search(include_pattern, line)
                if match:
                    results["includes"].append(match.group(1))
            results["includes"] = sorted(list(set(results["includes"])))

            # 2. Extract basic function signatures (very simplified)
            # This regex is basic and might not capture all valid C function signatures,
            # especially complex ones, K&R style, or those with pointers/arrays in tricky ways.
            # It tries to capture: return_type function_name(parameters)
            # Does not handle multi-line signatures well.
            # Example: int main(int argc, char *argv[])
            # Example: void print_hello(void)
            # Example: char* get_message()
            function_pattern = r'([a-zA-Z_][a-zA-Z0-9_*\s]*?\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?:\{|;)'
            # Explanation:
            # ([a-zA-Z_][a-zA-Z0-9_*\s]*?\s+) : Return type (allows spaces, pointers like char *)
            # ([a-zA-Z_][a-zA-Z0-9_]*)       : Function name
            # \s*\(([^)]*)\)\s*             : Parameters within parentheses
            # (?:\{|;)                       : Followed by an opening brace or semicolon (declaration)

            # Remove comments before regex search for functions to improve accuracy
            code_no_comments = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.MULTILINE|re.DOTALL)

            for match in re.finditer(function_pattern, code_no_comments):
                return_type = match.group(1).strip()
                func_name = match.group(2).strip()
                params = match.group(3).strip()
                # Store the full matched signature part for simplicity
                full_signature = f"{return_type} {func_name}({params})"
                # Avoid capturing function calls that might look like definitions if regex is too greedy
                # A simple check: if the line also contains typical keywords for definitions or declarations
                line_of_match = code_no_comments[:match.start()].count('\n') + 1

                # This check is heuristic, real parsing is needed for accuracy
                # For now, accept what regex finds.
                results["function_signatures"].append(full_signature)
            results["function_signatures"] = sorted(list(set(results["function_signatures"])))


            # 3. Extract struct, enum, typedef declarations (names only)
            struct_pattern = r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\{|;)' # captures struct name
            enum_pattern = r'enum\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\{|;)'     # captures enum name
            # Typedef can be complex, e.g. typedef struct {...} name; or typedef int my_int;
            # Simple typedef: typedef ... existing_type new_type_name;
            # This regex aims for `typedef ... NewTypeName;` (captures NewTypeName)
            # It's very basic and will miss complex typedefs like function pointers or struct typedefs.
            typedef_pattern = r'typedef\s+(?:[^;]*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*;'


            for match in re.finditer(struct_pattern, code_no_comments):
                results["struct_declarations"].append(match.group(1))
            results["struct_declarations"] = sorted(list(set(results["struct_declarations"])))

            for match in re.finditer(enum_pattern, code_no_comments):
                results["enum_declarations"].append(match.group(1))
            results["enum_declarations"] = sorted(list(set(results["enum_declarations"])))

            for match in re.finditer(typedef_pattern, code_no_comments):
                results["typedef_declarations"].append(match.group(1)) # The last word before ';' is assumed as new type
            results["typedef_declarations"] = sorted(list(set(results["typedef_declarations"])))


            logger.info("C code structure analysis complete (regex-based).",
                        num_includes=len(results["includes"]),
                        num_function_signatures=len(results["function_signatures"]),
                        num_structs=len(results["struct_declarations"]),
                        num_enums=len(results["enum_declarations"]),
                        num_typedefs=len(results["typedef_declarations"]))

        except Exception as e:
            logger.error("Unexpected error during C code analysis (regex)", error=str(e), exc_info=True)
            results["error"] = f"Unexpected error during C code analysis: {str(e)}"

        return results

# Add other placeholders if desired, e.g., JavaScriptAnalyzer, JavaAnalyzer...

class BestPracticesDatabase:
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
    Loads and provides language-specific best practices.
    Now supports Python and C.
    '''
    # Removed default practices_file_path from __init__ signature
    # It will now load language-specific files based on convention or a map.
    def __init__(self, languages_to_load: Optional[List[str]] = None):
        # Default to loading Python and C if no specific list is provided
        if languages_to_load is None:
            languages_to_load = ["python", "c"]

        self.practices: Dict[str, Dict[str, List[str]]] = {} # language -> category -> [practice_string]
        self.language_file_map: Dict[str, str] = { # Maps language to its practices JSON file
            "python": "config/python_best_practices.json",
            "c": "config/c_best_practices.json"
            # Add more languages and their files here in the future
        }

        # Filter map for only requested languages to load
        self.files_to_load_for_session: Dict[str, str] = {
            lang: file_path for lang, file_path in self.language_file_map.items()
            if lang in [l.lower() for l in languages_to_load]
        }

        self._load_practices()
        logger.info("BestPracticesDatabase initialized.",
                    requested_languages_to_load=languages_to_load,
                    actually_loaded_languages=list(self.practices.keys()))

    def _load_practices(self):
        '''Loads practices from the JSON files specified in self.files_to_load_for_session.'''
        for lang, file_path in self.files_to_load_for_session.items():
            logger.debug(f"Attempting to load best practices for language '{lang}' from '{file_path}'.")
            if not os.path.exists(file_path):
                logger.warn(f"Best practices file not found for language '{lang}' at path: {file_path}. No practices will be loaded for this language.")
                continue # Skip to next language if file not found

            try:
                with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
                    lang_specific_practices = json.load(f)
                    self.practices[lang] = lang_specific_practices
                logger.info(f"Successfully loaded best practices for language '{lang}' from {file_path}.")

            except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
                logger.error(f"Best practices file not found (double check) for '{lang}': {file_path}", exc_info=True)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from best practices file for '{lang}': {file_path}", exc_info=True)
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading best practices for '{lang}' from {file_path}", error=str(e), exc_info=True)

    def get_practices(self, language: str, context_categories: Optional[List[str]] = None) -> List[str]:
        # ... (get_practices method remains the same as implemented in "Enhanced Python CodeGen" Step 1) ...
        # It already handles different languages via self.practices.get(language.lower())
        lang_key = language.lower()
        lang_practices = self.practices.get(lang_key)
        if not lang_practices:
            logger.warn("No best practices found for language", language=lang_key)
            return []

        relevant_practices: List[str] = []
        requested_categories_lower = [cat.lower() for cat in context_categories] if context_categories else []

        if context_categories: # Specific categories requested
            for category in requested_categories_lower:
                if category in lang_practices:
                    relevant_practices.extend(lang_practices[category])
                else:
                    logger.debug(f"Category '{category}' not found in practices for language '{lang_key}'.")
            # Optionally, always add "general" practices if any specific category was requested and found results,
            # or if no results were found from specific categories.
            # For now, if specific categories are requested, only those are returned.
            # To also include 'general' when specific categories are asked:
            # if "general" in lang_practices and "general" not in requested_categories_lower:
            #    relevant_practices.extend(lang_practices["general"])
        else: # No categories specified, return all for the language
            for category_practices in lang_practices.values():
                relevant_practices.extend(category_practices)

        # Fallback if specific categories yielded nothing but general exists
        if context_categories and not relevant_practices and "general" in lang_practices:
            logger.debug(f"No practices found for specific categories {context_categories} in '{lang_key}', falling back to 'general' practices.")
            relevant_practices.extend(lang_practices["general"])

        # If no context_categories were given, 'general' would already be included by iterating all values.
        # If context_categories was empty list, it would also iterate all values.

        logger.debug("Retrieved best practices", language=lang_key, categories_requested=context_categories, num_practices_retrieved=len(set(relevant_practices)))
        return list(set(relevant_practices)) # Return unique practices


class CodeQualityChecker:
    '''
    Performs quality checks on Python code, integrating flake8 and custom AST-based checks.
    '''
    def __init__(self,
                 max_line_length: int = 100, # Custom check, though flake8 also has one (E501)
                 max_function_lines: int = 50
                ):
        self.max_line_length = max_line_length # Kept for custom check, can be compared with flake8
        self.max_function_lines = max_function_lines
        logger.info(f"CodeQualityChecker initialized (supports Python, C; max_line_length={max_line_length}, max_function_lines={self.max_function_lines}).")

    # ... (_is_snake_case, _is_pascal_case helper methods for Python checks - keep them) ...
    def _is_snake_case(self, name: str) -> bool: # Python specific, keep for Python checks
        return re.fullmatch(r'_?[a-z0-9_]+', name) is not None

    def _is_pascal_case(self, name: str) -> bool: # Python specific, keep for Python checks
        return re.fullmatch(r'[A-Z][a-zA-Z0-9]*', name) is not None

    def _check_python_specific(self, code: str, lines: List[str], ast_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        # Extracted Python-specific custom checks into a helper
        custom_issues_messages: List[str] = []

        # Line length (only if Flake8's E501 is not present - this logic needs flake8 results first)
        # This check might be better handled after flake8 results are known or removed if E501 is primary.
        # For now, this helper will just run it. The main check_code can decide to use it.
        # This specific conditional logic for line length based on E501 is handled in the main check_code method.
        # Here, we just provide the basic check.
        for i, line_content in enumerate(lines):
            line_num = i + 1
            if len(line_content) > self.max_line_length: # This will be re-evaluated in check_code for Python
                custom_issues_messages.append(f"Line {line_num}: Exceeds custom max line length of {self.max_line_length} (length: {len(line_content)}).")

        if ast_analysis_results:
            # Docstring checks
            if "functions" in ast_analysis_results:
                for func_info in ast_analysis_results["functions"]:
                    if not func_info.get("docstring_exists"):
                        custom_issues_messages.append(f"Function '{func_info.get('name', 'N/A')}': Missing docstring.")
            if "classes" in ast_analysis_results:
                for class_info in ast_analysis_results["classes"]:
                    if not class_info.get("docstring_exists"):
                        custom_issues_messages.append(f"Class '{class_info.get('name', 'N/A')}': Missing docstring.")
                    for method_info in class_info.get("methods", []):
                        if method_info.get("name") != "__init__" and not method_info.get("docstring_exists"):
                             custom_issues_messages.append(f"Method '{class_info.get('name', 'N/A')}.{method_info.get('name', 'N/A')}': Missing docstring.")

            # Naming Conventions
            if "functions" in ast_analysis_results:
                for func_info in ast_analysis_results["functions"]:
                    func_name = func_info.get("name")
                    if func_name and not self._is_snake_case(func_name) and not (func_name.startswith("__") and func_name.endswith("__")):
                        custom_issues_messages.append(f"Naming: Python Function '{func_name}' may not follow snake_case.")
                    for arg_name in func_info.get("args", []):
                        arg_base_name = arg_name.split('=')[0].strip()
                        if not self._is_snake_case(arg_base_name) and arg_base_name not in ["self", "cls"]:
                             custom_issues_messages.append(f"Naming: Python Function '{func_name}' arg '{arg_base_name}' may not follow snake_case.")
            if "classes" in ast_analysis_results:
                for class_info in ast_analysis_results["classes"]:
                    class_name = class_info.get("name")
                    if class_name and not self._is_pascal_case(class_name):
                        custom_issues_messages.append(f"Naming: Python Class '{class_name}' may not follow PascalCase.")

            # Function/Method Max Length
            if "functions" in ast_analysis_results:
                for func_info in ast_analysis_results["functions"]:
                    if func_info.get("body_line_count", 0) > self.max_function_lines:
                        custom_issues_messages.append(f"Length: Python Function '{func_info.get('name')}' ({func_info.get('body_line_count')} lines) exceeds max {self.max_function_lines} lines.")
            if "classes" in ast_analysis_results:
                for class_info in ast_analysis_results["classes"]:
                    for method_info in class_info.get("methods", []):
                        if method_info.get("body_line_count", 0) > self.max_function_lines:
                            custom_issues_messages.append(f"Length: Python Method '{class_info.get('name')}.{method_info.get('name')}' ({method_info.get('body_line_count')} lines) exceeds max {self.max_function_lines} lines.")

            # Wildcard Imports & Generic Exceptions (already implemented for Python)
            if "imports" in ast_analysis_results:
                for imp_statement in ast_analysis_results["imports"]:
                    if "*" in imp_statement and "from" in imp_statement:
                        custom_issues_messages.append(f"Import: Python Wildcard import found: '{imp_statement}'.")
            if "functions" in ast_analysis_results: # Generic exceptions checks
                for func_info in ast_analysis_results["functions"]:
                    for line_num in func_info.get("generic_except_clauses", []):
                        custom_issues_messages.append(f"Exception: Python Function '{func_info.get('name')}' uses generic 'except:' or 'except Exception:' on line {line_num}.")
            # ... (add for methods in classes too if not already covered by PythonAnalyzer's generic_except_clauses for methods)

        # Common "TODO" / "FIXME" check (can be considered language-agnostic but placed here for Python specific flow)
        for i, line_content in enumerate(lines):
            line_num = i + 1
            if "TODO" in line_content.upper() or "FIXME" in line_content.upper():
                custom_issues_messages.append(f"Line {line_num}: Contains 'TODO' or 'FIXME'.")
        return custom_issues_messages

    def _check_c_specific(self, code: str, lines: List[str], c_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        custom_issues_messages: List[str] = []
        # Example custom C check: gets() usage (very basic string check, cppcheck should be primary)
        for i, line_content in enumerate(lines):
            line_num = i + 1
            if "gets(" in line_content: # Naive check for gets()
                custom_issues_messages.append(f"Line {line_num}: Use of 'gets()' is highly insecure and deprecated.")

        # TODO/FIXME for C (language agnostic, but run it here for C as well)
        for i, line_content in enumerate(lines):
            line_num = i + 1
            if "TODO" in line_content.upper() or "FIXME" in line_content.upper():
                custom_issues_messages.append(f"Line {line_num}: Contains 'TODO' or 'FIXME'.")

        # Add more C-specific custom checks if desired, e.g., based on c_analysis_results
        # For instance, if CAnalyzer provided counts of `malloc` vs `free`.
        if c_analysis_results:
            # Example: check for missing includes if certain functions are used (highly heuristic)
            # if "function_signatures" in c_analysis_results and any("printf" in sig for sig in c_analysis_results["function_signatures"]):
            #     if not any("stdio.h" in inc for inc in c_analysis_results.get("includes",[])):
            #         custom_issues_messages.append("Usage of stdio functions like 'printf' suggested, but 'stdio.h' not found in includes.")
            pass

        return custom_issues_messages

    def check_code(self, code: str, language: str, analysis_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Checking {language} code quality", code_length=len(code))
        all_issues_found: List[Dict[str, Any]] = []
        lines = code.splitlines()
        num_lines = len(lines)

        linter_issue_count = 0
        custom_issue_messages: List[str] = []

        if num_lines == 0:
            logger.warn(f"Attempted to check quality of empty {language} code string.")
            return {"issues_found": [{"type": "custom", "message": "Code is empty."}], "quality_score": 0.0, f"{language}_linter_issues": 0, "custom_issues": 1}

        if language.lower() == "python":
            flake8_issues_raw = run_flake8_on_code(code)
            linter_issue_count = len(flake8_issues_raw)
            logger.debug("Flake8 analysis complete for Python", num_issues_found=linter_issue_count)
            for issue in flake8_issues_raw:
                all_issues_found.append({"type": "flake8", **issue})

            # Run Python-specific custom checks
            # Make custom line length check conditional on Flake8's E501 for Python
            # Store original max_line_length and temporarily modify if E501 is found
            original_max_line_length = self.max_line_length
            if any(issue.get("code") == "E501" for issue in flake8_issues_raw):
                logger.debug("Flake8 E501 found, disabling custom line length check for this run.")
                self.max_line_length = float('inf') # Effectively disable custom check

            custom_issue_messages = self._check_python_specific(code, lines, analysis_results)

            self.max_line_length = original_max_line_length # Restore original value

        elif language.lower() == "c":
            cppcheck_issues_raw = run_cppcheck_on_code(code)
            linter_issue_count = len(cppcheck_issues_raw)
            logger.debug("Cppcheck analysis complete for C", num_issues_found=linter_issue_count)
            for issue in cppcheck_issues_raw:
                # Ensure cppcheck issues have a consistent structure, map severity to a numeric if needed for scoring
                all_issues_found.append({"type": "cppcheck", **issue})

            # Run C-specific custom checks
            custom_issue_messages = self._check_c_specific(code, lines, analysis_results)

        else:
            logger.warn(f"Quality checks for language '{language}' are not fully implemented. Running generic custom checks only.")
            # Generic TODO/FIXME check for any language
            for i, line_content in enumerate(lines):
                if "TODO" in line_content.upper() or "FIXME" in line_content.upper():
                    custom_issue_messages.append(f"Line {i+1}: Contains 'TODO' or 'FIXME'.")


        for cust_msg in custom_issue_messages:
            all_issues_found.append({"type": "custom", "message": cust_msg}) # line, col, code might be missing for some custom

        num_custom_issues = len(custom_issue_messages)

        # Quality Score Calculation (example, can be language-specific)
        quality_score = 1.0
        # Penalties can be different for linter issues vs custom, or by severity for Cppcheck
        if language.lower() == "python":
            quality_score -= linter_issue_count * 0.1
            quality_score -= num_custom_issues * 0.05
        elif language.lower() == "c":
            # For Cppcheck, severity can be 'error', 'warning', 'style', 'performance', 'portability'
            for issue in all_issues_found:
                if issue.get("type") == "cppcheck":
                    severity = issue.get("severity", "style")
                    if severity == "error": quality_score -= 0.15
                    elif severity == "warning": quality_score -= 0.1
                    else: quality_score -= 0.05 # style, performance, portability
            quality_score -= num_custom_issues * 0.05 # Custom C checks
        else: # Generic language
            quality_score -= num_custom_issues * 0.05


        quality_score = max(0.0, round(quality_score, 2))

        logger.info(f"{language} code quality check complete.",
                    num_linter_issues=linter_issue_count,
                    num_custom_issues=num_custom_issues,
                    total_issues=len(all_issues_found),
                    quality_score=quality_score)

        return {
            "issues_found": all_issues_found,
            "quality_score": quality_score,
            f"{language.lower()}_linter_issue_count": linter_issue_count, # e.g. python_flake8_issue_count, c_cppcheck_issue_count
            "custom_issue_count": num_custom_issues
        }

# --- Main Code Generator Class ---

class MultiLanguageCodeGenerator:

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
    Performs quality checks on Python code, integrating flake8 and custom AST-based checks.
    '''
    def __init__(self,
                 max_line_length: int = 100, # Custom check, though flake8 also has one (E501)
                 max_function_lines: int = 50
                ):
        self.max_line_length = max_line_length # Kept for custom check, can be compared with flake8
        self.max_function_lines = max_function_lines
        logger.info(f"CodeQualityChecker initialized (supports Python, C; max_line_length={max_line_length}, max_function_lines={self.max_function_lines}).")

    # ... (_is_snake_case, _is_pascal_case helper methods for Python checks - keep them) ...
    def _is_snake_case(self, name: str) -> bool: # Python specific, keep for Python checks
        return re.fullmatch(r'_?[a-z0-9_]+', name) is not None

    def _is_pascal_case(self, name: str) -> bool: # Python specific, keep for Python checks
        return re.fullmatch(r'[A-Z][a-zA-Z0-9]*', name) is not None

    def _check_python_specific(self, code: str, lines: List[str], ast_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        # Extracted Python-specific custom checks into a helper
        custom_issues_messages: List[str] = []

        # Line length (only if Flake8's E501 is not present - this logic needs flake8 results first)
        # This check might be better handled after flake8 results are known or removed if E501 is primary.
        # For now, this helper will just run it. The main check_code can decide to use it.
        # This specific conditional logic for line length based on E501 is handled in the main check_code method.
        # Here, we just provide the basic check.
        for i, line_content in enumerate(lines):
            line_num = i + 1
            if len(line_content) > self.max_line_length: # This will be re-evaluated in check_code for Python
                custom_issues_messages.append(f"Line {line_num}: Exceeds custom max line length of {self.max_line_length} (length: {len(line_content)}).")

        if ast_analysis_results:
            # Docstring checks
            if "functions" in ast_analysis_results:
                for func_info in ast_analysis_results["functions"]:
                    if not func_info.get("docstring_exists"):
                        custom_issues_messages.append(f"Function '{func_info.get('name', 'N/A')}': Missing docstring.")
            if "classes" in ast_analysis_results:
                for class_info in ast_analysis_results["classes"]:
                    if not class_info.get("docstring_exists"):
                        custom_issues_messages.append(f"Class '{class_info.get('name', 'N/A')}': Missing docstring.")
                    for method_info in class_info.get("methods", []):
                        if method_info.get("name") != "__init__" and not method_info.get("docstring_exists"):
                             custom_issues_messages.append(f"Method '{class_info.get('name', 'N/A')}.{method_info.get('name', 'N/A')}': Missing docstring.")

            # Naming Conventions
            if "functions" in ast_analysis_results:
                for func_info in ast_analysis_results["functions"]:
                    func_name = func_info.get("name")
                    if func_name and not self._is_snake_case(func_name) and not (func_name.startswith("__") and func_name.endswith("__")):
                        custom_issues_messages.append(f"Naming: Python Function '{func_name}' may not follow snake_case.")
                    for arg_name in func_info.get("args", []):
                        arg_base_name = arg_name.split('=')[0].strip()
                        if not self._is_snake_case(arg_base_name) and arg_base_name not in ["self", "cls"]:
                             custom_issues_messages.append(f"Naming: Python Function '{func_name}' arg '{arg_base_name}' may not follow snake_case.")
            if "classes" in ast_analysis_results:
                for class_info in ast_analysis_results["classes"]:
                    class_name = class_info.get("name")
                    if class_name and not self._is_pascal_case(class_name):
                        custom_issues_messages.append(f"Naming: Python Class '{class_name}' may not follow PascalCase.")

            # Function/Method Max Length
            if "functions" in ast_analysis_results:
                for func_info in ast_analysis_results["functions"]:
                    if func_info.get("body_line_count", 0) > self.max_function_lines:
                        custom_issues_messages.append(f"Length: Python Function '{func_info.get('name')}' ({func_info.get('body_line_count')} lines) exceeds max {self.max_function_lines} lines.")
            if "classes" in ast_analysis_results:
                for class_info in ast_analysis_results["classes"]:
                    for method_info in class_info.get("methods", []):
                        if method_info.get("body_line_count", 0) > self.max_function_lines:
                            custom_issues_messages.append(f"Length: Python Method '{class_info.get('name')}.{method_info.get('name')}' ({method_info.get('body_line_count')} lines) exceeds max {self.max_function_lines} lines.")

            # Wildcard Imports & Generic Exceptions (already implemented for Python)
            if "imports" in ast_analysis_results:
                for imp_statement in ast_analysis_results["imports"]:
                    if "*" in imp_statement and "from" in imp_statement:
                        custom_issues_messages.append(f"Import: Python Wildcard import found: '{imp_statement}'.")
            if "functions" in ast_analysis_results: # Generic exceptions checks
                for func_info in ast_analysis_results["functions"]:
                    for line_num in func_info.get("generic_except_clauses", []):
                        custom_issues_messages.append(f"Exception: Python Function '{func_info.get('name')}' uses generic 'except:' or 'except Exception:' on line {line_num}.")
            # ... (add for methods in classes too if not already covered by PythonAnalyzer's generic_except_clauses for methods)

        # Common "TODO" / "FIXME" check (can be considered language-agnostic but placed here for Python specific flow)
        for i, line_content in enumerate(lines):
            line_num = i + 1
            if "TODO" in line_content.upper() or "FIXME" in line_content.upper():
                custom_issues_messages.append(f"Line {line_num}: Contains 'TODO' or 'FIXME'.")
        return custom_issues_messages

    def _check_c_specific(self, code: str, lines: List[str], c_analysis_results: Optional[Dict[str, Any]]) -> List[str]:
        custom_issues_messages: List[str] = []
        # Example custom C check: gets() usage (very basic string check, cppcheck should be primary)
        for i, line_content in enumerate(lines):
            line_num = i + 1
            if "gets(" in line_content: # Naive check for gets()
                custom_issues_messages.append(f"Line {line_num}: Use of 'gets()' is highly insecure and deprecated.")

        # TODO/FIXME for C (language agnostic, but run it here for C as well)
        for i, line_content in enumerate(lines):
            line_num = i + 1
            if "TODO" in line_content.upper() or "FIXME" in line_content.upper():
                custom_issues_messages.append(f"Line {line_num}: Contains 'TODO' or 'FIXME'.")

        # Add more C-specific custom checks if desired, e.g., based on c_analysis_results
        # For instance, if CAnalyzer provided counts of `malloc` vs `free`.
        if c_analysis_results:
            # Example: check for missing includes if certain functions are used (highly heuristic)
            # if "function_signatures" in c_analysis_results and any("printf" in sig for sig in c_analysis_results["function_signatures"]):
            #     if not any("stdio.h" in inc for inc in c_analysis_results.get("includes",[])):
            #         custom_issues_messages.append("Usage of stdio functions like 'printf' suggested, but 'stdio.h' not found in includes.")
            pass

        return custom_issues_messages

    def check_code(self, code: str, language: str, analysis_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Checking {language} code quality", code_length=len(code))
        all_issues_found: List[Dict[str, Any]] = []
        lines = code.splitlines()
        num_lines = len(lines)

        linter_issue_count = 0
        custom_issue_messages: List[str] = []

        if num_lines == 0:
            logger.warn(f"Attempted to check quality of empty {language} code string.")
            return {"issues_found": [{"type": "custom", "message": "Code is empty."}], "quality_score": 0.0, f"{language}_linter_issues": 0, "custom_issues": 1}

        if language.lower() == "python":
            flake8_issues_raw = run_flake8_on_code(code)
            linter_issue_count = len(flake8_issues_raw)
            logger.debug("Flake8 analysis complete for Python", num_issues_found=linter_issue_count)
            for issue in flake8_issues_raw:
                all_issues_found.append({"type": "flake8", **issue})

            # Run Python-specific custom checks
            # Make custom line length check conditional on Flake8's E501 for Python
            # Store original max_line_length and temporarily modify if E501 is found
            original_max_line_length = self.max_line_length
            if any(issue.get("code") == "E501" for issue in flake8_issues_raw):
                logger.debug("Flake8 E501 found, disabling custom line length check for this run.")
                self.max_line_length = float('inf') # Effectively disable custom check

            custom_issue_messages = self._check_python_specific(code, lines, analysis_results)

            self.max_line_length = original_max_line_length # Restore original value

        elif language.lower() == "c":
            cppcheck_issues_raw = run_cppcheck_on_code(code)
            linter_issue_count = len(cppcheck_issues_raw)
            logger.debug("Cppcheck analysis complete for C", num_issues_found=linter_issue_count)
            for issue in cppcheck_issues_raw:
                # Ensure cppcheck issues have a consistent structure, map severity to a numeric if needed for scoring
                all_issues_found.append({"type": "cppcheck", **issue})

            # Run C-specific custom checks
            custom_issue_messages = self._check_c_specific(code, lines, analysis_results)

        else:
            logger.warn(f"Quality checks for language '{language}' are not fully implemented. Running generic custom checks only.")
            # Generic TODO/FIXME check for any language
            for i, line_content in enumerate(lines):
                if "TODO" in line_content.upper() or "FIXME" in line_content.upper():
                    custom_issue_messages.append(f"Line {i+1}: Contains 'TODO' or 'FIXME'.")


        for cust_msg in custom_issue_messages:
            all_issues_found.append({"type": "custom", "message": cust_msg}) # line, col, code might be missing for some custom

        num_custom_issues = len(custom_issue_messages)

        # Quality Score Calculation (example, can be language-specific)
        quality_score = 1.0
        # Penalties can be different for linter issues vs custom, or by severity for Cppcheck
        if language.lower() == "python":
            quality_score -= linter_issue_count * 0.1
            quality_score -= num_custom_issues * 0.05
        elif language.lower() == "c":
            # For Cppcheck, severity can be 'error', 'warning', 'style', 'performance', 'portability'
            for issue in all_issues_found:
                if issue.get("type") == "cppcheck":
                    severity = issue.get("severity", "style")
                    if severity == "error": quality_score -= 0.15
                    elif severity == "warning": quality_score -= 0.1
                    else: quality_score -= 0.05 # style, performance, portability
            quality_score -= num_custom_issues * 0.05 # Custom C checks
        else: # Generic language
            quality_score -= num_custom_issues * 0.05


        quality_score = max(0.0, round(quality_score, 2))

        logger.info(f"{language} code quality check complete.",
                    num_linter_issues=linter_issue_count,
                    num_custom_issues=num_custom_issues,
                    total_issues=len(all_issues_found),
                    quality_score=quality_score)

        return {
            "issues_found": all_issues_found,
            "quality_score": quality_score,
            f"{language.lower()}_linter_issue_count": linter_issue_count, # e.g. python_flake8_issue_count, c_cppcheck_issue_count
            "custom_issue_count": num_custom_issues
        }

# --- Main Code Generator Class ---

class MultiLanguageCodeGenerator:
    '''
    Generates code in multiple languages, guided by specifications and context.
    Integrates analysis, quality checking, and best practices for supported languages.
    Currently supports Python and C.
    '''

    def __init__(self,
                 llm_aggregator: LLMAggregator
                 ):
        self.llm_aggregator = llm_aggregator
        self.python_analyzer = PythonAnalyzer()
        self.c_analyzer = CAnalyzer() # Instantiate CAnalyzer
        # BestPracticesDatabase now loads ["python", "c"] by default
        self.best_practices_db = BestPracticesDatabase()
        self.code_quality_checker = CodeQualityChecker()
        logger.info("MultiLanguageCodeGenerator initialized with analyzers (Python, C), BestPracticesDatabase, and CodeQualityChecker.")

    async def generate_code(self, spec: CodeSpecification) -> CodeGenerationResult:
        logger.info("Starting code generation process",
                    spec_id=spec.spec_id, target_language=spec.target_language)

        language_lower = spec.target_language.lower()
        generated_code_content: Optional[str] = None
        error_message: Optional[str] = None
        analysis_results: Optional[Dict[str, Any]] = None
        quality_check_results: Optional[Dict[str, Any]] = None

        # Retrieve best practices for the target language
        # Determine categories based on spec or default to common ones for the language
        practice_categories = ["general_style", "functions", "error_handling"] # Generic default
        if language_lower == "python":
            practice_categories = ["general", "functions", "classes", "error_handling", "imports", "idiomatic_python"]
        elif language_lower == "c":
            practice_categories = ["general_style", "headers", "functions", "memory_management", "pointers", "error_handling"]

        retrieved_practices = self.best_practices_db.get_practices(language_lower, context_categories=practice_categories)
        logger.debug(f"Retrieved {len(retrieved_practices)} best practices for {language_lower} prompt", categories=practice_categories)

        # Construct the prompt
        prompt_parts = [f"Please generate {language_lower} code based on the following specification."]
        prompt_parts.append(f"Core Task/Logic: {spec.prompt_details}")
        if spec.context_summary: prompt_parts.append(f"Relevant Context: {spec.context_summary}")
        if spec.examples:
            example_str = "\n".join([f"- Input: {ex.get('input', 'N/A')}, Output: {ex.get('output', 'N/A')}" for ex in spec.examples])
            prompt_parts.append(f"Examples of desired behavior:\n{example_str}")
        if spec.constraints:
            constraint_str = "\n".join([f"- {c}" for c in spec.constraints])
            prompt_parts.append(f"Constraints or specific requirements:\n{constraint_str}")
        if retrieved_practices:
            practices_str = "\n".join([f"- {p}" for p in retrieved_practices])
            prompt_parts.append(f"Please adhere to the following {language_lower} best practices where applicable:\n{practices_str}")

        prompt_parts.append(f"\nEnsure the output contains ONLY the {language_lower} code. Do not include any explanations, introductions, or markdown fences like ```{language_lower} ... ``` unless the fence is part of a multi-line string within the code itself.")
        prompt_parts.append(f"If the request is unclear or cannot be fulfilled as {language_lower} code, please respond with an error message prefixed with 'ERROR:'.")
        final_prompt = "\n\n".join(prompt_parts)

        system_message_content = f"You are an expert {language_lower} code generation AI. Your output should be only the raw {language_lower} code as requested, adhering to provided specifications and best practices. If you cannot fulfill the request, respond with 'ERROR: Your reason for failure.'"

        messages = [
            OpenHandsMessage(role="system", content=system_message_content),
            OpenHandsMessage(role="user", content=final_prompt)
        ]
        request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.25)

        if language_lower not in ["python", "c"]:
            logger.warn(f"Unsupported language for code generation pipeline: {language_lower}", spec_id=spec.spec_id)
            return CodeGenerationResult(
                specification_id=spec.spec_id, language=spec.target_language,
                error_message=f"Language '{spec.target_language}' is not supported for full generation pipeline in this version."
            )

        try:
            logger.debug(f"Sending {language_lower} code generation request to LLM", spec_id=spec.spec_id, prompt_length=len(final_prompt))
            response = await self.llm_aggregator.chat_completion(request)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                llm_output = response.choices[0].message.content.strip()
                logger.debug(f"Received LLM response for {language_lower} code generation", spec_id=spec.spec_id, response_length=len(llm_output))

                if llm_output.startswith("ERROR:"):
                    error_message = llm_output
                    logger.warn(f"LLM indicated an error in generating {language_lower} code", spec_id=spec.spec_id, llm_error=error_message)
                else:
                    # Clean markdown fences (generic, then language specific if needed)
                    if llm_output.startswith(f"```{language_lower}"):
                        llm_output = llm_output[len(f"```{language_lower}"):].strip()
                        if llm_output.endswith("```"): llm_output = llm_output[:-3].strip()
                    elif llm_output.startswith("```"):
                        llm_output = llm_output[3:].strip()
                        if llm_output.endswith("```"): llm_output = llm_output[:-3].strip()

                    generated_code_content = llm_output
                    logger.info(f"Successfully generated {language_lower} code (pre-analysis)", spec_id=spec.spec_id, code_length=len(generated_code_content))

                    # --- Language-Specific Analysis and Quality Check ---
                    if generated_code_content:
                        if language_lower == "python":
                            logger.info("Analyzing generated Python code structure...", spec_id=spec.spec_id)
                            analysis_results = self.python_analyzer.analyze_code_structure(generated_code_content)
                            if analysis_results.get("error"): logger.warn("Error during AST analysis (Python)", spec_id=spec.spec_id, analysis_error=analysis_results.get("error"))
                            logger.info("Checking quality of generated Python code...", spec_id=spec.spec_id)
                            quality_check_results = self.code_quality_checker.check_code(generated_code_content, "python", analysis_results=analysis_results)

                        elif language_lower == "c":
                            logger.info("Analyzing generated C code structure (basic)...", spec_id=spec.spec_id)
                            analysis_results = self.c_analyzer.analyze_code_structure(generated_code_content) # Using CAnalyzer
                            if analysis_results.get("error"): logger.warn("Error during C code analysis", spec_id=spec.spec_id, analysis_error=analysis_results.get("error"))
                            logger.info("Checking quality of generated C code (cppcheck + custom)...", spec_id=spec.spec_id)
                            quality_check_results = self.code_quality_checker.check_code(generated_code_content, "c", analysis_results=analysis_results)

                        if quality_check_results:
                             logger.info(f"Code quality check complete for {language_lower}.", spec_id=spec.spec_id, issues_found=len(quality_check_results.get("issues_found",[])), score=quality_check_results.get("quality_score"))
                    # --- End Language-Specific ---
            else:
                error_message = f"LLM response was empty or malformed for {language_lower} code generation."
                logger.warn(error_message, spec_id=spec.spec_id, llm_response_response_object=response) # Log whole object if choices are missing

        except Exception as e:
            error_message = f"An unexpected error occurred during {language_lower} code generation pipeline: {str(e)}"
            logger.error(f"{language_lower} code generation pipeline failed", spec_id=spec.spec_id, error=str(e), exc_info=True)

        return CodeGenerationResult(
            specification_id=spec.spec_id,
            generated_code=generated_code_content,
            language=language_lower if generated_code_content else spec.target_language, # Use actual lang if generated
            error_message=error_message,
            issues_found=quality_check_results.get("issues_found", []) if quality_check_results else [],
            quality_score=quality_check_results.get("quality_score") if quality_check_results else None
            # Consider adding analysis_results to CodeGenerationResult if useful downstream
        )
