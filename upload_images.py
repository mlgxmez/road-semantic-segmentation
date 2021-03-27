import argparse

from azureml.core import Workspace

from utils.storage import DataManager

parser = argparse.ArgumentParser()
parser.add_argument('--target_folder')

args = parser.parse_args()
ws = Workspace.from_config()

dm = DataManager(ws)
_ = dm.upload(
    folder_to_upload=args.target_folder,
    path_datastore="data")
