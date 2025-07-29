# from datascience ai team
from langchain_core.output_parsers import BaseOutputParser

import re

# Python Parser for output standardization  
class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):        
        def extract_python_code(text):
            python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
            if python_code_match:
                python_code = python_code_match.group(1).strip()
                return python_code
            else:
                python_code_match = re.search(r"python(.*?)'", text, re.DOTALL)
                if python_code_match:
                    python_code = python_code_match.group(1).strip()
                    return python_code
                else:
                    return None
        python_code = extract_python_code(text)
        if python_code is not None:
            return python_code
        else:
            # Assume ```sql wasn't used
            return text