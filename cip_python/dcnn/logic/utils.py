import numpy as np
import SimpleITK as sitk
import logging
import datetime
from git import Repo
import json

class Utils(object):
    @staticmethod
    def now():
        """
        Return current date/time in '%Y-%m-%d_%H-%M-%S' format
        :return: string representing the current date/time
        """
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    @staticmethod
    def configure_loggers(default_level=logging.DEBUG, log_file=None, file_logging_level=logging.DEBUG):
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        # logging.basicConfig(format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s", level=default_level)
        root_logger = logging.getLogger()
        root_logger.setLevel(min(default_level, file_logging_level))

        console_handler = logging.StreamHandler()
        # console_handler.setFormatter(log_formatter)
        console_handler.setLevel(default_level)
        root_logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(file_logging_level)
            root_logger.addHandler(file_handler)

        paramiko_logger = logging.getLogger("paramiko")
        if paramiko_logger:
            paramiko_logger.setLevel(logging.FATAL)

        # Disable all the info logging from boto
        for key in logging.Logger.manager.loggerDict:
            if key.startswith("boto"):
                logger = logging.getLogger(key)
                logger.setLevel(logging.FATAL)
        logger = logging.getLogger('s3transfer')
        if logger:
            logger.setLevel(logging.FATAL)

    @staticmethod
    def read_parameters_dict(file_path):
        """
        Read a json file that contains a dictionary of parameters
        :param file_path: str. Path to the json file
        :return: dictionary
        """
        with open(file_path, 'r') as f:
            return json.loads(f.read())

    @staticmethod
    def get_param(param_name, parameters_dict, value_if_not_present=None):
        """
        Get a parameter value from a dictionary of parameters.
        Return None if the parameter is not in the dictionary
        Args:
            param_name: str. Name of the parameter
            parameters_dict: Dictionary
            value_if_not_present: value that will be returned if the parameter does not exist in the dictionary

        Returns:
            Parameter value or value_if_not_present
        """
        if param_name not in parameters_dict:
            return value_if_not_present
        return parameters_dict[param_name]

    @staticmethod
    def get_git_url(file_path=None):
        """
        Get the current full url to the current commit

        Returns:
            tuple and url and checking if the Repository has uncommited changes (true==uncommited changes (dirty folder))
        """
        if file_path is None:
            file_path = __file__

        repo = Repo(file_path, search_parent_directories=True)
        remote = repo.remotes[0]
        url = ""
        for url in remote.urls:
            # Just grab the url of the remote (don't know a better way to do it!)
            pass
        if "@" in url:
            # Remove user-password information
            url = "https://" + url[url.index("@") + 1:]

        url = url.replace(".git", "")
        commit = repo.head.commit.hexsha
        full_url = "{}/commit/{}".format(url, commit)
        return (full_url, repo.is_dirty())

    @staticmethod
    def save_keras_model_summary(model, file_path):
        """
        Save a keras model summary to a text file
        :param model: keras model
        :param file_path: str. Path to the file
        """
        f = None
        try:
            f = open(file_path, 'w')

            def print_fn(s):
                f.write(s + "\n")

            model.summary(print_fn=print_fn)
        finally:
            if f is not None:
                f.close()