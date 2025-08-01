import re

def relocate_imports_inside_function(code_text):
    """
    Relocates all import statements in a given Python function and moves them inside the function definition.

    Parameters
    ----------
    code_text : str
        The Python code as a string.

    Returns
    -------
    str
        The modified Python code with imports relocated inside the function.
    """
    # Match all import statements
    import_pattern = r'^\s*(import\s+[^\n]+|from\s+\S+\s+import\s+[^\n]+)\s*$'
    imports = re.findall(import_pattern, code_text, re.MULTILINE)

    # Remove imports from the top-level code
    code_without_imports = re.sub(import_pattern, '', code_text, flags=re.MULTILINE).strip()

    # Find the function definition and insert the imports inside it
    function_pattern = r'(def\s+\w+\s*\(.*?\):)'
    match = re.search(function_pattern, code_without_imports)

    if match:
        function_start = match.end()
        # Insert the imports right after the function definition
        imports_code = '\n    ' + '\n    '.join(imports)  # Indent imports
        modified_code = (
            code_without_imports[:function_start]
            + imports_code
            + code_without_imports[function_start:]
        )
        return modified_code

    # If no function is found, return the original code
    return code_text