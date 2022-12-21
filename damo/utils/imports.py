# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import sys

if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    import importlib
    import importlib.util
    import sys

    def import_file(module_name, file_path, make_importable=False):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if make_importable:
            sys.modules[module_name] = module
        return module
else:
    import imp

    def import_file(module_name, file_path, make_importable=None):
        module = imp.load_source(module_name, file_path)
        return module
