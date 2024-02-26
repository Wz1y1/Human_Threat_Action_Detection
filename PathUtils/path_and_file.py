import os
import re
from typing import Iterable, Tuple
from functools import singledispatch
from pathlib import Path
import shutil


def try_to_find_file(file_path):
    """
    寻找结果文件。有的话就返回True，如果没有的话则返回False
    """
    return os.path.isfile(file_path)


def try_to_find_file_if_exist_then_delete(file_):
    if try_to_find_file(file_):
        os.remove(file_)


def try_to_find_folder_if_exist_then_delete(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Delete the folder
        try:
            shutil.rmtree(folder_path)
            print(f"Successfully deleted the folder at {folder_path}")
        except Exception as e:
            print(f"An error occurred while deleting the folder: {e}")


# 使用泛型函数
@singledispatch
def try_to_find_folder_path_otherwise_make_one(folder_path):
    assert (isinstance(folder_path, tuple)) or isinstance(folder_path, str) or isinstance(folder_path, Path)


@try_to_find_folder_path_otherwise_make_one.register(str)
def _(folder_path: str):
    if os.path.exists(folder_path):
        return True
    else:
        os.makedirs(folder_path)
        return False


@try_to_find_folder_path_otherwise_make_one.register(tuple)
def _(folder_path: Tuple[str, ...]):
    for i in folder_path:
        try_to_find_folder_path_otherwise_make_one(i)


@try_to_find_folder_path_otherwise_make_one.register(Path)
def _(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=True)


def list_all_specific_format_files_in_a_folder_path(folder_path: Path, format_: str, order: str = 'time'):
    folder_path = folder_path.__str__()
    files = os.listdir(folder_path)
    files = [x for x in files if re.search(r'\.' + format_ + '$', x, re.IGNORECASE)]
    files = [folder_path + '\\' + x for x in files]
    if order == 'time':
        files = sorted(files, key=lambda x: os.path.getctime(x))
    return files


def remove_win10_max_path_limit():
    import winreg

    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem",
                         0, winreg.KEY_WRITE)
    winreg.SetValueEx(key, r"LongPathsEnabled", 0, winreg.REG_DWORD, 1)


def restore_win10_max_path_limit():
    import winreg

    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem",
                         0, winreg.KEY_WRITE)
    winreg.SetValueEx(key, r"LongPathsEnabled", 0, winreg.REG_DWORD, 0)


if __name__ == "__main__":
    remove_win10_max_path_limit()
