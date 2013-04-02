#!/usr/bin/python

import subprocess

class ReadNRRDsWriteVTKModule:
        def __init__( self ):
                self._argumentList = ""

        # TODO: Add backslash check for last 'dir' character.
        # TODO: Add check for 'exe' extension to executable name.
        def SetCIPBuildDirectory( self, dir ):
                self._cipBuildDirectory = dir

        def AddFileNameArrayNamePair( self, fileName, arrayName ):
                self._argumentList = self._argumentList + " -i " + fileName + " -a " + arrayName

        def SetOutputFileName( self, fileName ):
                self._argumentList = self._argumentList + " -o " + fileName

        def Execute( self ):
                tmpCommand = self._cipBuildDirectory + "bin/ReadNRRDsWriteVTK " + self._argumentList                
                subprocess.call( tmpCommand, shell=True )
