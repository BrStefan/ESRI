import os


def compute_path(path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(project_root, path)