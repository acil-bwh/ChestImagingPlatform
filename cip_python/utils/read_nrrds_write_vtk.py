#!/usr/bin/python

import subprocess

class ReadNRRDsWriteVTK:
    """
    Interface to the ReadNRRDsWriteVTK program

    Parameters
    ----------
    out_file_name : string
        Name of output VTK file
    """
    def __init__(self, out_file_name):
        self._argumentList = ""
        self._argumentList = self._argumentList + " -o " + out_file_name

    def add_file_name_array_name_pair( self, fileName, arrayName ):
        self._argumentList = self._argumentList + " -i " + fileName + " -a " + arrayName

    def execute( self ):
        tmpCommand = "ReadNRRDsWriteVTK " + self._argumentList                
        subprocess.call( tmpCommand, shell=True )
