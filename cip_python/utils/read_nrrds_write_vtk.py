import subprocess

class ReadNRRDsWriteVTK:
    """
    Interface to the ReadNRRDsWriteVTK program

    Parameters
    ----------
    out_file_name : string
        Name of output VTK file
    """
    def __init__(self, out_file_name,binary=True):
        self._argumentList = ""
        self._argumentList = self._argumentList + " -o " + out_file_name
        if binary == True:
            self._argumentList = self._argumentList + " -b"

    def add_file_name_array_name_pair( self, fileName, arrayName ):
        self._argumentList = self._argumentList + " -i " + fileName + " -a " + arrayName
    
    def add_metadata_name_value_pair( self, name, value):

        if type(value) == type(tuple()) or type(value) == type(list()):
            value=",".join([str(val) for val in value])
        
        self._argumentList = self._argumentList + " --%(name)s %(value)s" % \
                            {'name':name,'value':value}

    def set_cip_region( self, cip_region='UndefinedRegion'):
        """Set the CIP chest region for the particles.
        """
        self._argumentList = self._argumentList + ' --cipr ' + cip_region

    def set_cip_type( self, cip_type='UndefinedType'):
        """Set the CIP chest region for the particles.
        """
        self._argumentList = self._argumentList + ' --cipt ' + cip_type
        
    def execute( self ):
        print (self._argumentList)
        tmpCommand = "ReadNRRDsWriteVTK " + self._argumentList
        subprocess.call( tmpCommand, shell=True )
