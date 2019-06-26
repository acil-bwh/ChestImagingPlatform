import os
import os.path as osp
import inspect

class Paths(object):
    __this_dir__ = osp.dirname(osp.realpath(__file__))

    ResourcesFolder = osp.realpath(osp.join(__this_dir__, '..', '..', 'Resources'))
    TestingDataFolder = osp.realpath(osp.join(__this_dir__, '..', '..', 'Testing', 'Data', 'Input'))
    TestingOuputFolder = osp.realpath(osp.join(__this_dir__, '..', 'tests_output'))

    @staticmethod
    def testing_file_path(_file):
        """
        Get the whole path to a data file stored in the Testing Data Input folder concatenating paths
        Parameters
        ----------
        _file   Name of the file / subpath

        Returns
        -------
        Path
        """
        return osp.join(Paths.TestingDataFolder, _file)


    @staticmethod
    def resources_file_path(_file):
        """
        Get the whole path to a data file stored in the Testing Data Input folder concatenating paths
        Parameters
        ----------
        _file   Name of the file / subpath

        Returns
        -------
        Path
        """
        return osp.join(Paths.ResourcesFolder, _file)

    @staticmethod
    def testing_baseline_file_path(_file):
        """
        Get the whole path to a data file stored in the Testing baseline folder.
        Each category will have its own baseline folder, which will be computed and created dynamically.
        Ex: cip_python/common/data/baseline

        Parameters
        ----------
        _file   Name of the file / subpath

        Returns
        -------
        Full path to the file in the baseline folder
        """
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 1)
        caller_full_path = calframe[1][1]
        caller_baseline_data_dir = osp.realpath(osp.join(osp.dirname(caller_full_path), "baseline"))
        if not osp.isdir(caller_baseline_data_dir):
            os.makedirs(caller_baseline_data_dir)
            print ("Folder {} was created".format(caller_baseline_data_dir))
        return osp.join(caller_baseline_data_dir, _file)

