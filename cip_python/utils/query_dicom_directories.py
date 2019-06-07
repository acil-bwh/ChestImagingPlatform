import sys, math, os
from optparse import OptionParser
import pydicom as dicom
import csv
from os import listdir
from os.path import isfile, join

import pdb
    
def formatted_line_from_dicom_info(dicom_dir, dicom_info, attrs):
    """Given a dicom info structure and a list of desired attributes, return
    a comma-separated-value line with the corresponding entries.

    Paramters
    ---------
    dicom_dir : string
        DICOM directory in which query is being made
    
    dicom_info : DICOM info structure
        DICOM information structure as returned by the dicom.read_file routine

    attrs : list
        List of string-valued DICOM attributes as recognized within dicom_info.
        If a list entry can not be found in dicom_info, a blank space will be
        placed for that location in the formatted line.

    Returns
    -------
    formatted_line : string
        A comma-separated value line with values of the requested entries. Each
        entry in the formatted line is put in double qoates to escape entries
        that themselves have commas (such as the PixelSpacing entry). (Note that
        this approach to escaping only works if there are no spaces after a
        comma that is actually a separator between fields).
    """
    formatted_line = str("")
        
    for i in range(0, len(attrs)):
        if attrs[i] == 'NumSlices':
            files = [f for f in listdir(dicom_dir) if \
                     isfile(join(dicom_dir, f))]
            num_slices = len(files)
            formatted_line = formatted_line + '"' + str(num_slices) + '",'
        elif attrs[i] == 'Directory':
            formatted_line = formatted_line + '"' + dicom_dir + '",'
        elif ((attrs[i] in dicom_info) is True): 
            getattr(dicom_info, attrs[i])
            formatted_line = formatted_line + '"' + \
                str(getattr(dicom_info, attrs[i])) + '",'
        else:
            formatted_line = formatted_line + " ,"

    return formatted_line

def get_file_in_dir(directory, file_num=0):
    """"Get a file name (with directory) within a specified directory.

    Parameters
    ----------
    directory : string
        Directory in which to retrieve a file name

    file_num : integer, optional
        Which file name within the directory to return. By default, the first
        file listed will be returned.

    Returns
    -------
    dir_and_file_name : string
        The specified directory name plus with the file name appended.
    """
    if file_num < 0:
        raise ValueError("Requested file number must be 0 or greater")

    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    assert len(files) > 0, "Directory " + directory + " is empty"

    # Append the first file found in the directory to the path and
    # remove double slashes if necessary
    tmp = directory + '/' + files[0]
    dir_and_file_name = tmp.replace("//", "/")

    return dir_and_file_name

def print_dicom_tags():
   """Print the full list of DICOM tags to screen for reference"""
   print ('PatientName')
   print ('PatientID')
   print ('StudyDate')
   print ('InstitutionName')
   print ('Manufacturer')
   print ('ManufacturersModelName')
   print ('DateOfLastCalibration')
   print ('ConvolutionKernel')
   print ('StudyDescription')
   print ('ModalitiesInStudy')
   print ('ImageComments')
   print ('SliceThickness')
   print ('SpacingBetweenSlices')
   print ('ExposureTime')
   print ('XRayTubeCurrent')
   print ('KVP')
   print ('WindowCenter')
   print ('ContrastBolusAgent')
   print ('DataCollectionDiameter')
   print ('ReconstructionDiameter')
   print ('DistanceSourceToDetector')
   print ('DistanceSourceToPatient')
   print ('GantryDetectorTilt')
   print ('TableHeight')
   print ('Exposure')
   print ('FocalSpots')
   print ('ImagePositionPatient')
   print ('SliceLocation')
   print ('PixelSpacing')
   print ('RescaleIntercept')
   print ('RescaleSlope')
   print ('ProtocolName')
   print ('AcquisitionDate')
   print ('StudyID')
   print ('SeriesDescription')
   print ('SeriesTime')
   print ('PatientBirthDate')
   print ('FilterType')
   print ('stationNameID')
   print ('StationName')
   print ('AcquisitionTime')
   print ('PatientPosition')
   print ('studyInstanceUID')
   print ('seriesInstanceUID')
   print ('acquisitionDate')
   print ('seriesDate')
   print ('modality')
   print ('NumSlices')
   print ('Directory')

def print_info_to_screen(tags, info):
    """Print the tags and associated DICOM information to screen.

    Parameters
    ----------
    tags : string
        A single string with the queried DICOM tags listed

    info : list of strings
        Each entry of the list is a formatted string of DICOM information
        entries, one for each queried directory.
    """
    print (tags)
    for i in info:
        print (i)

def print_info_to_file(tags, info, out_file):
    """Print the tags and associated DICOM information to screen.

    Parameters
    ----------
    tags : string
        A single string with the queried DICOM tags listed

    info : list of strings
        Each entry of the list is a formatted string of DICOM information
        entries, one for each queried directory.

    out_file : string
        The file name of the output file to which to write the DICOM
        information.
    """
    writer = open(out_file, "w")
    writer.write(tags)
    writer.write("\n")

    for i in info:
        writer.write(i)
        writer.write("\n")

    writer.close()    
   

    
def query_dicom_directories(dirs_file, out_file, tags, dicom_dir, show_tags):

    # Print the tags to screen for reference if requested
    if show_tags:
       print_dicom_tags()

    if tags is not None:
        # Parse the list of tags the user has specified into a list of
        # attributes       
        attrs = tags.replace(" ", "").split(',')

        if dicom_dir is not None:
            if os.path.exists(dicom_dir):
                # Get all the files in the directory. We assume that these are
                # DICOM files. We will just use the first one in the returned
                # list for querying
                dir_and_file_name = get_file_in_dir(dicom_dir)               
                dicom_info = dicom.read_file(dir_and_file_name,force=True)

                formatted_line = \
                    formatted_line_from_dicom_info(dicom_dir,
                                                   dicom_info, attrs)
                print_info_to_screen(tags, [formatted_line])
            else:
                raise ValueError("Specified DICOM directory does not exist") 

        # Now read the directories file if the user has specified one
        if dirs_file is not None:
            assert os.path.isfile(dirs_file), "Directories file does \
            not exist"

            with open(dirs_file) as f:
                dirs = f.readlines()

            all_info = []
            for line in dirs:
                dicom_dir = line.rstrip('\n')
                dir_and_file_name = get_file_in_dir(dicom_dir)
                dicom_info = dicom.read_file(dir_and_file_name,force=True)
                all_info.append(formatted_line_from_dicom_info(dicom_dir,
                                                            dicom_info, attrs))

            if options.out_file is not None:
                print_info_to_file(tags, all_info, out_file)
            else:
                print_info_to_screen(tags, all_info)     


if __name__ == "__main__":
    desc = """Query a set of DICOM directories for tag information and write
    results to an output file."""
    
    parser = OptionParser(description=desc)
    parser.add_option('--dirs_file', 
                      help='File containing a list of DICOM directories to\
                      query. Each line should contain one full path to a\
                      given directory. The DICOM info will be printed to the \
                      specified output file. If no output file is specified, \
                      the info will be printed to the screen',
                      dest='dirs_file', metavar='<string>', default=None)
    parser.add_option('--out_file', 
                      help='The output (.csv) file to write the results to',
                      dest='out_file', metavar='<string>', default=None)
    parser.add_option('--tags', 
                      help='The names of the DICOM tags to qeury. If multiple \
                      tags are desired, specify them within quotes, separated \
                      by commas.', dest='tags', metavar='<string>',
                      default=None)
    parser.add_option('--dicom_dir', 
                      help='A DICOM directory in which to perform the query. \
                      The info will be printed to screen.',
                      dest='dicom_dir', metavar='<string>', default=None)
    parser.add_option('--show_tags', 
                      help='Print out a list of DICOM tags for reference. \
                      Note that the tags NumSlices and Directory are not \
                      DICOM tags, but they are available to the other inputs \
                      in this script as if they were.', dest='show_tags',
                      default=False, action='store_true')    

    (options, args) = parser.parse_args()
    query_dicom_directories(options.dirs_file, options.out_file, options.tags, options.dicom_dir, options.show_tags)