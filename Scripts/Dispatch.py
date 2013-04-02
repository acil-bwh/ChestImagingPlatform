# Read: case name, format, input directory, output directory, temp directory, pipeline to dispatch

#import FileNameCreator from cipPython
#import <pipeline> from cipPython

inNameCreator = FileNameCreator()
inNameCreator.SetDirectory( inputDirectory )
inNameCreator.SetCase( case )
inNameCreator.SetFormat( format )

outNameCreator = FileNameCreator()
outNameCreator.SetDirectory( outputDirectory )
outNameCreator.SetCase( case )
outNameCreator.SetFormat( format )

tmpNameCreator = FileNameCreator()
tmpNameCreator.SetDirectory( tmpDirectory )
tmpNameCreator.SetCase( case )
tmpNameCreator.SetFormat( format )

# Instantiate pipeline as 'pipeline'
pipeline = <pipeline>
pipeline.SetInputNameCreator( inNameCreator )
pipeline.SetOutputNameCreator( outNameCreator )
pipeline.SetTemporaryNameCreator( tmpNameCreator )
pipeline.Execute()
