"""
Module to handle abstracted functionalities
"""
import os

FULL_NAME = [
    'Background',
    'CorticalGrayMatter',
    'SubcorticalGrayMatter',
    'GrayMatter',
    'gm',
    'WhiteMatter',
    'wm',
    'CerebrospinalFluid',
    'Bone',
    'SoftTissue',
    'Nonbrain',
    'Lesion',
    'Brainstem',
    'Cerebellum'
]
SHORT_NAME = [
    'BG',
    'CGM',
    'SGM',
    'GM',
    'GM',
    'WM',
    'WM',
    'CSF',
    'B',
    'ST',
    'NB',
    'L',
    'BS',
    'CBM'
]


def make_path(paths: list[str]):
    """
    Creates a new path in local system by joining paths, and making any new directories, if
    necessary.

    Args:
        paths (list[str]): A list containing strings to be joined as a path in the system
            directory.

    Note:
        If the final string provided includes a period '.' (a proxy for checking if the path is a 
        file name) this method will result in creating the folder above the last provided string in
        the list.
    """
    end_dir = paths[-1]
    if end_dir.find('.') == -1:
        out_path = os.path.join(paths)
    else:
        out_path = os.path.join(paths[:-1])
    os.makedirs(out_path,exist_ok=True)


def abbreviate_region(region_name: str):
    """
    Converts long region names to their associated abbreviations.
    """
    name_out = region_name.replace('-','').replace('_','')
    for i,_d in enumerate(FULL_NAME):
        full_name = FULL_NAME[i]
        short_name = SHORT_NAME[i]
        name_out = name_out.replace(full_name,short_name)
    return name_out


def build_label_map(region_names: list[str]):
    """
    Builds a BIDS compliant label map. Loop through CTAB and convert names to
    abbreviations using :meth:`abbreviate_region`
    """
    abbreviated_names = list(map(abbreviate_region,region_names))
    return abbreviated_names
