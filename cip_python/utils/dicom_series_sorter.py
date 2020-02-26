import os
import pydicom
import argparse


class Sorter():
    def __init__(self):
        """ Sorter initialization
        """
    
    def execute(self,folder):

        for root, dirs, files in os.walk(folder):

            if not dirs:
                for filename in files:
                    if not files[0] == '.':
                        try:
                            ds = pydicom.dcmread(os.path.join(root,filename))
                            patientid = ds['PatientID'].value
                            studyuid = ds['StudyInstanceUID'].value
                            seriesuid = ds['SeriesInstanceUID'].value
                            directory = os.path.join(folder,patientid,studyuid,seriesuid)
                            if not os.path.exists(directory):
                                os.makedirs(directory)

                            os.rename(os.path.join(root,filename), os.path.join(directory,filename))
                        except:
                            print ("Error reading %s." % os.path.join(root,filename))
                            err_folder = os.path.join(folder,"errors",root)
                            if not os.path.exists(err_folder):
                                os.makedirs(err_folder)
                            os.rename(os.path.join(root,filename), os.path.join(err_folder,filename))


    
    def clean_empty_folders(self,folder):
        for root, dirs, files in os.walk(folder,topdown=False):
            try:
                os.removedirs(root)
            # If path is not a directory 
            except NotADirectoryError: 
                print("%s is not a directory." % root) 
              
            # If permission related errors 
            except PermissionError: 
                print("Permission denied for %s." % root) 

            except OSError as error: 
                print("Directory %s can not be removed." % root) 




if __name__ == '__main__':

    #Sort dicom series in different folders

    parser = argparse.ArgumentParser(description='Sort dicom series in different folders.')
    parser.add_argument('-i', dest='in_folder', metavar='in_folder', required=True, help="Path to folder containing dicom hierarchy.")
    parser.add_argument("--clean", dest='clean', help="Clean empty directories after sorting dicoms.", action="store_true")



    op = parser.parse_args()

    st = Sorter()
    st.execute(op.in_folder)

    if op.clean:
        print("Cleaning empty folders...")    
        st.clean_empty_folders(op.in_folder)

    print ("DONE.")