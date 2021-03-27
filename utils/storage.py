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

    def upload(self, folder_to_upload, path_datastore):
        """
        Upload files to Azure Blob Storage attached to AzureML Workspace
        """
        targetPath = DataPath(self.datastore, path_datastore)
        # TODO: Check if these folder have been uploaded
        fileDataset = Dataset.File.upload_directory(
            folder_to_upload,
            targetPath)
        return fileDataset

    def download(self, folder_to_download, path_local):
        """
        Download files from a single folder
        """
        pathInDatastore = DataPath(self.datastore, folder_to_download)
        dataToDownload = FileDatasetFactory.from_files(pathInDatastore)
        pathDownloadedFiles = dataToDownload.download(path_local)
        return pathDownloadedFiles
