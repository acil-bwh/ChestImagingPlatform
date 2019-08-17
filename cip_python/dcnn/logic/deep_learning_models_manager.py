import os
import os.path as osp
import requests
import sys

class DeepLearningModelsManager(object):
    # Model URLs
    _MODELS_ = {
          'LUNG_SEGMENTATION_AXIAL': 'lung_segmentation/lung_segmentation_axial.hdf5',
          'LUNG_SEGMENTATION_CORONAL': 'lung_segmentation/lung_segmentation_coronal.hdf5',
          'LOW_DOSE_TO_HIGH_DOSE': 'low_dose_to_high_dose/low_dose_to_high_dose_3d2d.h5'
    }

    def __init__(self, root_url="https://s3.amazonaws.com/acil-deep-learning-data/models/"):
        """
        Constructor

        Parameters
        ----------
        root_url: root url where all the models will be searched

        """
        self._ROOT_URL_ = root_url

    def get_model(self, key):
        """
        Get a model local path. Download the model if it has not been already downloaded
        Parameters
        ----------
        key: str. Model key (it should be one of the values of _MODELS_ dict

        Returns
        -------
        Path to a local file that contains a model that may have been downloaded from a remote location (if it was
        not saved locally yet)
        """
        if key not in self._MODELS_:
            raise Exception("Model '{}' is not one of the allowed values. Allowed values: {}".format(key,
                                                                                 self._MODELS_.keys()))
        local_model_path = osp.realpath(osp.join(self._get_models_root_local_folder_(), self._MODELS_[key]))
        url = self._ROOT_URL_ + self._MODELS_[key]
        if not osp.isfile(local_model_path):
            # Download the model
            print ("Model not found in {}. Downloading...".format(local_model_path))
            if not self._download_model_(url, local_model_path):
                raise Exception("Model file could not be retrieved")
            print ("Model saved!")
        return local_model_path

    def _get_models_root_local_folder_(self):
        """
        Get the local folder where the models will be stored.
        The folder will be created if needed
        Returns
        -------

        """
        real_path = osp.realpath(osp.join(osp.dirname(__file__), "..", "dcnn_models_cache"))
        if not osp.isdir(real_path):
            try:
                os.makedirs(real_path)
            except Exception as ex:
                raise Exception("Models folder could not be created ('{}'): {}".format(real_path, ex))
        return real_path


    def _download_model_(self, url, local_file):
        """
        Download a model from a remote location

        Parameters
        ----------
        key: model key
        local_file: full local path where the model is going to be stored

        Returns
        -------
        True if the process went good or False of the process failed

        """
        try:

            local_folder = osp.dirname(local_file)
            if not osp.isdir(local_folder):
                os.makedirs(local_folder)
                print ("{} folder created".format(local_folder))

            # NOTE the stream=True parameter
            r = requests.get(url, stream=True)
            total_size = 0
            with open(local_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=10000):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
                    total_size+=len(chunk)

            if os.stat(local_file).st_size < 4000:
                # There seems to be an error here. The file size should be way bigger
                with open(local_file, 'r') as f:
                    r = f.read()
                print("Error when downloading file: {}".format(r))
                # Remove wrong file
                os.remove(local_file)
                return False
            return True
        except Exception as ex:
            print ("There was an error while downloading the model: ", ex)
            return False
