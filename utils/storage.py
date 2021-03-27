from azureml.core import Datastore, Dataset
from azureml.data.dataset_factory import FileDatasetFactory
from azureml.data.datapath import DataPath


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

    def download(self, folder_to_download, path_local):
        """
        Download files from a single folder.

        Args:
            folder_to_download: Folder to download from Datastore.
            path_local: Path where files from Datastore will be downloaded.

        """
        pathInDatastore = DataPath(self.datastore, folder_to_download)
        dataToDownload = FileDatasetFactory.from_files(pathInDatastore)
        pathDownloadedFiles = dataToDownload.download(path_local)
        return pathDownloadedFiles

    def splitDataset(self, dataset_name, percentage, seed=None):
        """
        Split Dataset into two subsets.

        Args:
            dataset_name: Name of the Dataset to be splitted.
            percentage: Percentage of files to be move to another Dataset
            seed: Seed number

        Returns:
            Two Dataset objects the first containing the amount of files
            specified in percentage, and the other Dataset containing the
            remaining files

        """
        dataset = self.workspace.datasets[dataset_name]
        ds1, ds2 = dataset.random_split(percentage=percentage, seed=seed)
        return ds1, ds2
