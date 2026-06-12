import sys
import nbformat
from nbclient import NotebookClient

repo = "/home/michael-ramirez/GitHub/personal/paleo-gpr-ml"
path = f"{repo}/notebooks/05_synthetic_data_generation.ipynb"

nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb,
    timeout=-1,
    kernel_name="python3",
    resources={"metadata": {"path": repo}},  # kernel cwd = repo root
)
client.execute()
nbformat.write(nb, path)
print("NOTEBOOK EXECUTED OK", file=sys.stderr)
