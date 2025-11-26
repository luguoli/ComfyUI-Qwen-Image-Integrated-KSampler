import importlib.util
import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

python = sys.executable

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def serialize(obj):
    if isinstance(obj, (str, int, float, bool, list, dict, type(None))):
        return obj
    return str(obj)  # 转为字符串


# 获取当前目录并查找所有 Python 文件
current_dir = get_ext_dir()
files = os.listdir(current_dir)
all_nodes = {}

for file in files:
    if not file.endswith(".py") or file.startswith("__"):
        continue  # 跳过不是 .py 文件或以 __ 开头的文件（如 __init__.py）

    name = os.path.splitext(file)[0]
    try:
        # 使用相对导入
        module_name = ".{}".format(name)
        imported_module = importlib.import_module(module_name, __name__)
        if hasattr(imported_module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
        if hasattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
        serialized_CLASS_MAPPINGS = {k: serialize(v) for k, v in getattr(imported_module, 'NODE_CLASS_MAPPINGS', {}).items()}
        serialized_DISPLAY_NAME_MAPPINGS = {k: serialize(v) for k, v in getattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS', {}).items()}
        all_nodes[file]={"NODE_CLASS_MAPPINGS": serialized_CLASS_MAPPINGS, "NODE_DISPLAY_NAME_MAPPINGS": serialized_DISPLAY_NAME_MAPPINGS}
    except Exception as e:
        # 跳过导入失败的文件（可能是由于依赖缺失）
        pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
