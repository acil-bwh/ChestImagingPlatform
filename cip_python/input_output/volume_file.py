"""
Wrapper for an image loaded from a file to handle the data and the metadata in a single structure
"""
class VolumeFile(object):
    def __init__(self, file_path, data, origin, spacing, space_directions, full_metadata_dict):
        self.file_path = file_path
        self.data = data
        self.shape = data.shape
        self.origin = origin
        self.spacing = spacing
        self.space_directions = space_directions
        self.full_metadata_dict = full_metadata_dict
