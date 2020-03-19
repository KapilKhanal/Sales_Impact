import papermill as pm
import subprocess

pm.execute_notebook(
    'reports/template.ipynb',
    'reports/sales_impact_report.ipynb',
    parameters=dict(filename='./data/interim/rfmtable.csv')
)

def generate_html_report(notebook_file):
    generate = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            notebook_file,
            "--no-input",
            "--to=html",  
        ]
    )
    print("HTML Report was generated")
    return True
