import pandas as pd
import plotly.express as px
from crewai.tools import tool
import io
import re


@tool("Python Code Executor Tool")
def python_code_executor_tool(code: str) -> str:
    """
    Executes a string of Python code to generate a Plotly chart.
    The code MUST save the chart to an HTML file named 'chart.html'.
    It uses 'exec()' to run the code. USE WITH CAUTION.
    The function returns a success message or an error.
    """
    try:
        # Define a dictionary to capture the locals and globals from exec
        # This includes providing necessary libraries to the executed code
        execution_globals = {
            "pd": pd,
            "px": px,
            "io": io
        }
        # The exec function will run the code within this context
        exec(code, execution_globals)
        return "Chart generated successfully and saved to chart.html."
    except Exception as e:
        return f"Error executing code: {e}"