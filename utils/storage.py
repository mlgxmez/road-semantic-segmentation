from azureml.core import Datastore, Dataset
from azureml.data.datapath import DataPath
from azureml.data.file_dataset import FileDataset
import os


class DataManager(object):

    def __init__(self, workspace):
        """
        Setting resources to store your data
        """
        self.workspace = workspace
        self.datastore = Datastore.get_default(self.workspace)

    def upload(self, folder_to_upload, path_datastore, dataset_name=None):
        """
        Upload files to Azure Blob Storage attached to AzureML Workspace.

        Args:
            folder_to_upload: Local folder to be uploaded to the DataStore.
            path_datastore: Path in the Datastore where files in
            folder_to_upload will be stored.
            dataset_name: Name of the Dataset created as a result ot the
            upload.

        Returns:
            Returns a Filedataset of the uploaded folder in Datastore.

        """
        targetPath = DataPath(self.datastore, path_datastore)
        fileDataset = Dataset.File.upload_directory(
            folder_to_upload,
            targetPath)
        if dataset_name is not None:
            fileDataset.register(self.workspace, dataset_name)
        return fileDataset

    def download(self, dataset_name, path_local):
        """
        Download files from a single folder.

        Args:
            dataset_name: Name of Dataset to be downloaded from Datastore.
            path_local: Path where files from Datastore will be downloaded.

        Returns:
            List of paths with all files downloaded from Datastore.

        """
        dataToDownload = self.workspace.datasets[dataset_name]
        pathDownloadedFiles = dataToDownload.download(path_local)
        return pathDownloadedFiles

    def splitDataset(self, dataset_name, percentage, seed=None):
        """
        Split Dataset into two subsets.

        Args:
            dataset_name: FileDataset or name of the Dataset to be splitted.
            percentage: Percentage of files to be move to another Dataset
            seed: Seed number

        Returns:
            Two Dataset objects the first containing the amount of files
            specified in percentage, and the other Dataset containing the
            remaining files

        """
        if isinstance(dataset_name, FileDataset):
            print('dataset_name is already a FileDataset')
            dataset = dataset_name
        else:
            dataset = self.workspace.datasets[dataset_name]
        ds1, ds2 = dataset.random_split(percentage=percentage, seed=seed)
        return ds1, ds2

    def filterDataset(self, dataset_name, pattern):
        """
        Creates a filtered dataset from a registered dataset.

        Args:
            dataset_name: Name of the source FileDataset from which to
            obtain the filtered dataset.
            pattern: Pattern from which to filter the dataset.

        Returns:
            An unregistered FileDataset filtered by the pattern specified.

        """
        return Dataset.File.from_files(
            (self.datastore, os.path.join('data', dataset_name, pattern))
            )


def validateDataset(dataset1, dataset2, match_chars=9):
    """
    This function compares sizes between two FileDataset. And
    throws an error if there is a mismatch between the number
    of files in both 'dataset1' and 'dataset2'.

    Args:
        dataset1: FileDataset object
        dataset2: Another FileDataset object
        match_chars: Number of ending characters to match to
                     detect mismatches in filenames

    """
    files_dataset1 = sorted(dataset1.to_path())
    files_dataset2 = sorted(dataset2.to_path())
    try:
        assert len(files_dataset1) == len(files_dataset2)
        print("Both FileDataset contain {} files.".format(len(files_dataset1)))
    except AssertionError:
        print("First FileDataset",
              "contains {} files.".format(len(files_dataset1)))
        print("Second FileDataset"
              "contains {} files.".format(len(files_dataset2)))

    valid_count = 0
    invalid_count = 0
    for f1, f2 in zip(files_dataset1, files_dataset2):
        # Count the number of matches of ending filenames
        match_pos = 0
        for c1, c2 in zip(f1[::-1], f2[::-1]):
            if c1 == c2:
                match_pos += 1
        if match_pos >= match_chars:
            valid_count += 1
        else:
            invalid_count += 1

    print("Number of matches between datasets: {}".format(valid_count))
    print("Number of mismatches between datasets: {}".format(invalid_count))
