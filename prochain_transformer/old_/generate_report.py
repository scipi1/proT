import nbformat as nbf
import yaml
from os.path import dirname, abspath, join
import sys


ROOT_DIR = dirname(dirname(abspath(__file__)))

sys.path.append(ROOT_DIR)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_results(results_path):
    import pandas as pd
    return pd.read_csv(results_path)

def generate_notebook(config, #results,
                      plots_path, 
                      template_path, output_path):
    with open(template_path) as f:
        nb = nbf.read(f, as_version=4)

    config_content = yaml.dump(config, default_flow_style=False)
    # accuracy = results['accuracy'].values[0]
    # loss = results['loss'].values[0]

    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            if cell.id == "config-cell":
                cell.source = cell.source.replace('{{ config_content }}', config_content)
            
        elif cell.cell_type == "code":
            if cell.id == "plots-cell":
                cell.source = cell.source.replace('{{ plots_path }}', plots_path)
            # cell.source = cell.source.replace('{{ accuracy }}', str(accuracy))
            # cell.source = cell.source.replace('{{ loss }}', str(loss))

    with open(output_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    exp_id = r"experiment_002"
    exps_dir = join(ROOT_DIR,r"experiments")
    exp_dir = join(exps_dir,exp_id)
    config_path = join(exp_dir, r"config.yaml")
    plots_path = join(exp_dir,r"logs",r"csv")
    #results_path = join(experiment_dir, 'results.csv')
    template_path = r"notebooks/nb_experiment_template.ipynb"
    output_path = join(exp_dir, r"report.ipynb")

    config = load_config(config_path)
    #results = load_results(results_path)

    generate_notebook(config, plots_path, template_path, output_path)
