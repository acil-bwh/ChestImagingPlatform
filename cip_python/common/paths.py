import os.path as osp

class Paths(object):
    __this_dir__ = osp.dirname(osp.realpath(__file__))

    ResourcesFolder = osp.realpath(osp.join(__this_dir__, '..', '..', 'Resources'))
    TestingDataFolder = osp.realpath(osp.join(__this_dir__, '..', '..', 'Testing', 'Data', 'Input'))

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

