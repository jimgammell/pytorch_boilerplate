import os
from typing import Callable
import time

class Trial:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.experiment_methods = {}
    
    def get_subtrial_dir(self, name: str):
        path = os.path.join(self.output_dir, name)
        os.makedirs(path, exist_ok=True)
        return path
    
    def register_subtrial_dir(self, name: str):
        return lambda: self.get_subtrial_dir(name)
    
    def register_experiment_method(self, method_name: str, method_fn: Callable):
        self.experiment_methods[method_name] = method_fn
    
    def __call__(self, **kwargs):
        for experiment_name, experiment_method in self.experiment_methods.items():
            should_run = f'run_{experiment_name}' in kwargs and kwargs[f'run_{experiment_name}']
            if should_run:
                print(f'Beginning experiment `{experiment_name}`.')
                start_time = time.time()
                experiment_method()
                end_time = time.time()
                print(f'Finished running experiment `{experiment_name}` in {end_time-start_time} sec.')