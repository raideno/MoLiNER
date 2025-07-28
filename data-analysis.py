import os
import typing
import logging
import nbformat
import papermill

from concurrent.futures import ProcessPoolExecutor, as_completed

from hydra import main
from omegaconf import DictConfig
from nbconvert import HTMLExporter

from src.load import extract_best_ckpt, extract_ckpt
from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE
)

ANALYSIS_NOTEBOOK_DIR_PATH = './notebooks/data'
ANALYSIS_NOTEBOOK_NAME = 'template.analysis.ipynb'

logger = logging.getLogger(__name__)

def run_notebook_for_pipeline(configuration: typing.Tuple[str, str, str]):
    dataset_name, pipeline_name, raw_split_names = configuration
    
    split_names = raw_split_names.split(",") 
    
    logger.info(f"[data-analysis]: running analysis for '{dataset_name}({pipeline_name})'...")
    
    template_ipynb_path = os.path.join(ANALYSIS_NOTEBOOK_DIR_PATH, ANALYSIS_NOTEBOOK_NAME)
    output_ipynb_path = os.path.join(ANALYSIS_NOTEBOOK_DIR_PATH, f'{dataset_name}.{pipeline_name}.analysis.ipynb')
    
    papermill.execute_notebook(
        template_ipynb_path,
        output_ipynb_path,
        parameters={
            "dataset_name": dataset_name,
            "pipeline_name": pipeline_name,
            "split_names": split_names
        },
    )
    
    logger.info(f"[data-analysis]: analysis for pipeline '{dataset_name}({pipeline_name})' completed.")

    logger.info(f"[data-analysis]: exporting '{output_ipynb_path}' to HTML...")
    
    with open(output_ipynb_path) as notebook_file:
        nb_node = nbformat.read(notebook_file, as_version=4)
        
    (body, resources) = HTMLExporter().from_notebook_node(nb_node)
    
    html_output_path = os.path.join(ANALYSIS_NOTEBOOK_DIR_PATH, f'{dataset_name}.{pipeline_name}.analysis.html')
    
    with open(html_output_path, 'w', encoding='utf-8') as file:
        file.write(body)
    
    logger.info(f"[data-analysis]: HTML export for pipeline '{dataset_name}({pipeline_name})' completed.")
    
    return configuration

@main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="data-analysis", version_base=DEFAULT_HYDRA_VERSION_BASE)
def data_analysis(cfg: DictConfig):
    configurations = cfg.configurations
    max_workers = getattr(cfg, 'max_workers', None)
    
    logger.info("[data-analysis]: starting data analysis...")
    
    logger.info(f"[data-analysis]: max_workers: {max_workers}; configurations to analyze: {configurations}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_configuration = {executor.submit(run_notebook_for_pipeline, configuration): configuration for configuration in configurations}
        
        for future in as_completed(future_to_configuration):
            configuration = future_to_configuration[future]
            
            try:
                future.result()
            except Exception as exception:
                logger.error(f"[data-analysis]: configuration '{configuration}' generated an exception: {exception}")
            else:
                logger.info(f"[data-analysis]: configuration '{configuration}' completed successfully.")

    logger.info("[data-analysis]: all analyses completed.")
        

if __name__ == "__main__":
    data_analysis()
