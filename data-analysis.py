import os
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

def run_notebook_for_pipeline(pipeline):
    logger.info(f"[data-analysis]: running analysis for pipeline '{pipeline}'...")
    
    template_ipynb_path = os.path.join(ANALYSIS_NOTEBOOK_DIR_PATH, ANALYSIS_NOTEBOOK_NAME)
    output_ipynb_path = os.path.join(ANALYSIS_NOTEBOOK_DIR_PATH, f'{pipeline}.analysis.ipynb')
    
    papermill.execute_notebook(
        template_ipynb_path,
        output_ipynb_path,
        parameters=dict(pipeline_name=pipeline),
    )
    
    logger.info(f"[data-analysis]: analysis for pipeline '{pipeline}' completed.")

    logger.info(f"[data-analysis]: exporting '{output_ipynb_path}' to HTML...")
    
    with open(output_ipynb_path) as notebook_file:
        nb_node = nbformat.read(notebook_file, as_version=4)
        
    (body, resources) = HTMLExporter().from_notebook_node(nb_node)
    
    html_output_path = os.path.join(ANALYSIS_NOTEBOOK_DIR_PATH, f'{pipeline}.analysis.html')
    
    with open(html_output_path, 'w', encoding='utf-8') as file:
        file.write(body)
    
    logger.info(f"[data-analysis]: HTML export for pipeline '{pipeline}' completed.")
    
    return pipeline

@main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="data-analysis", version_base=DEFAULT_HYDRA_VERSION_BASE)
def data_analysis(cfg: DictConfig):
    pipelines = cfg.pipelines
    max_workers = getattr(cfg, 'max_workers', None)
    
    logger.info("[data-analysis]: starting data analysis...")
    
    logger.info(f"[data-analysis]: max_workers: {max_workers}; pipelines to analyze: {pipelines}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pipeline = {executor.submit(run_notebook_for_pipeline, pipeline): pipeline for pipeline in pipelines}
        
        for future in as_completed(future_to_pipeline):
            pipeline = future_to_pipeline[future]
            
            try:
                future.result()
            except Exception as exception:
                logger.error(f"[data-analysis]: pipeline '{pipeline}' generated an exception: {exception}")
            else:
                logger.info(f"[data-analysis]: pipeline '{pipeline}' completed successfully.")

    logger.info("[data-analysis]: all analyses completed.")
        

if __name__ == "__main__":
    data_analysis()
