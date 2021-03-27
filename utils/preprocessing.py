from pathlib import Path
import random
from shutil import copy

from typing import List, Tuple


def randomSubsetFiles(
    root_directory: str,
    ratio: float,
    extension: str = 'png',
) -> Tuple[List[Path], List[Path]]:
    """
    Function to split files of root_directory into two lists

    Args:
        root_directory: Path to directory containing the files to be
        split into two lists.
        ratio: Ratio of files in root directory to be assigned to a
        different list
        extension: File extension of the files to be split into two lists

    Returns:
        Returns a tuple of two list containing the resulting split
        of all the files in root_directory

    """
    path = Path(root_directory)
    foundFiles = list(path.rglob('*.{}'.format(extension)))
    originalSubset, newSubset = [], []
    for file in foundFiles:
        rnd_num = random.uniform(0.0, 1.0)
        if rnd_num < ratio:
            newSubset.append(file)
        else:
            originalSubset.append(file)
    return originalSubset, newSubset


def insertFilenameString(filename: Path, string: str) -> str:
    """
    Inserts a string between two words separated by underscore.

    Args:
        filename: Path object containing the name of a file.
        string: String to append between words in the filename.

    Returns:
        Returns filename with string added in the middle between underscores.

    """
    sepFile = filename.name.split('_')
    return '_'.join([sepFile[0], string, sepFile[1]])


def splitMasksFolder(
    root_directory: str,
    original_subset: List[Path],
    new_subset: List[Path],
) -> Tuple[List[Path], List[Path]]:
    """
    Rearrange a list of Path objects into two lists according their filenames.

    Args:
        root_directory: Path to directory where its files are rearranged.
        original_subset: A list of Path objects whose filenames that match
        with filesnames in root_directory are added to a new list.
        new_subset: A list of Path object whose filenames that match with
        filenames in root_directory are added to a new list.

    Returns:
        Tuple with two lists containing Path objects rearranged in the same
        fashion as original_subset and new_subset

    """
    path = Path(root_directory)
    path = list(path.rglob('*.png'))[0].parent
    originalMasks, newMasks = [], []
    for p in original_subset:
        new_p = path / insertFilenameString(p, 'road')
        originalMasks.append(new_p)
    for p in new_subset:
        new_p = path / insertFilenameString(p, 'road')
        newMasks.append(new_p)
    return originalMasks, newMasks


def createFolder(path: Path) -> None:
    """
    Function that creates a folder in path if that folder does not exist.

    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        print('Folder {} has been created.'.format(path))
    except FileExistsError:
        print('Folder {} already exists.'.format(path))


def copyFiles(
    list_paths: List[Path],
    directory: str,  # 'train' or 'eval'
    append_parent_dir: int = 0,
) -> None:
    """
    Function that copies all the files from a list of Path to a directory

    Args:
        list_paths: List of Path objects of files being copied
        directory: Directory where files in list_path will be copied to
        append_parent_dir: Number of parent directory to include when
        copying files from list_paths to the target directory

    """
    newPath = Path.cwd() / directory
    if append_parent_dir > 0:
        for i in range(append_parent_dir-1, -1, -1):
            parentDir = list_paths[0].parents[i].name
            newPath = newPath / parentDir
    createFolder(newPath)
    for file in list_paths:
        copy(file, newPath)
