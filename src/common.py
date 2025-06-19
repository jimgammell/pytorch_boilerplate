from typing import Any, Callable, Optional, Dict
import os
import socket
import yaml
import sys
import importlib
import logging

import torch

PLOT_WIDTH = 4
PLOT_KWARGS: Dict[str, Any] = dict(rasterized=True)
SAVEFIG_KWARGS: Dict[str, Any] = dict(dpi=300)

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'outputs')
CONFIG_DIR = os.path.join(PROJ_DIR, 'config')
RESOURCE_DIR = os.path.join(PROJ_DIR, 'resources')
HOSTNAME = socket.gethostname()
sys.path.insert(0, SRC_DIR)

AVAILABLE_CONFIG_NAMES = [x.split('.')[0] for x in os.listdir(CONFIG_DIR) if x != 'per_machine_config.yaml']
assert os.path.exists(os.path.join(CONFIG_DIR, 'per_machine_config.yaml'))
with open(os.path.join(CONFIG_DIR, 'per_machine_config.yaml'), 'r') as f:
    _config = yaml.load(f, Loader=yaml.FullLoader)
for hostname_component in _config.keys():
    if hostname_component in HOSTNAME:
        config = _config[hostname_component]
        OPENWEBTEXT_ROOT = config['openwebtext'] if 'openwebtext' in config else None
        IMAGENET_ROOT = config['imagenet'] if 'imagenet' in config else None
        IMAGENETTE_ROOT = config['imagenette'] if 'imagenette' in config else None
        ASCADv1_ROOT = config['ascadv1'] if 'ascadv1' in config else None
        ASCADv2_ROOT = config['ascadv2'] if 'ascadv2' in config else None
        JESTER_ROOT = config['jester'] if 'jester' in config else None

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESOURCE_DIR, exist_ok=True)

def get_worker_count() -> int:
    worker_count = os.cpu_count()
    assert worker_count is not None
    worker_count = 3*worker_count//4
    return worker_count

def get_huggingface_module(module_name): # necessary because apparently the HuggingFace libraries are just named e.g. 'datasets' instead of 'huggingface.datasets'
    local_path = os.path.abspath(os.path.join(SRC_DIR, module_name))
    orig_path = sys.path.copy()
    sys.path = [p for p in sys.path if os.path.abspath(p) != local_path]
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.path = orig_path
    return module

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = 10*gpu_properties.major + gpu_properties.minor
    if arch >= 70:
        torch.set_float32_matmul_precision('high')

logger = logging.getLogger('general_logger')
def init_logger(print: bool = True, logfile: Optional[str] = None, level: int = logging.NOTSET):
    global logger
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s -%(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    if print:
        print_handler = logging.StreamHandler()
        print_handler.setFormatter(formatter)
        logger.addHandler(print_handler)
    if logfile is not None:
        file_handler = logging.FileHandler(filename=logfile, mode='w', encoding='utf-8', delay=False)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    class StreamToLogger:
        def __init__(self, log_func: Callable[[str], None]):
            self.log_func = log_func
            self.encoding = None
        
        def write(self, message: str):
            if message.strip():
                self.log_func(message.strip())
        
        def flush(self):
            pass
    def log_exception(exc_type: Any, exc_val: Any, exc_traceback: Any):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_val, exc_traceback)
        else:
            logger.error('Unhandled exception:', exc_info=(exc_type, exc_val, exc_traceback))
    sys.stdout = StreamToLogger(logger.info)
    sys.stderr = StreamToLogger(logger.error)
    sys.excepthook = log_exception

    logger.debug('Initialized debugger and common settings.')
    logger.debug(f'Host name: {HOSTNAME}')
    logger.debug(f'Default device: {torch.cuda.get_device_name()}')
    logger.debug(f'Project directory: {PROJ_DIR}')
    logger.debug(f'Output directory: {OUTPUT_DIR}')
    logger.debug(f'Config directory: {CONFIG_DIR}')
    assert torch.cuda.is_available()
    if arch >= 70:
        logger.debug('Using high matmul precision')