import os
import pydicom
import argparse
import shutil

class Sorter():
    def __init__(self):
        """ Sorter initialization
        """
    
    def execute(self, folder, output, cop):

        for root, dirs, files in os.walk(folder):

            if not dirs:
                for filename in files:
                    if not files[0] == '.':
                        try:
                            ds = pydicom.dcmread(os.path.join(root,filename))
                            patientid = ds['PatientID'].value
                            studyuid = ds['StudyInstanceUID'].value
                            seriesuid = ds['SeriesInstanceUID'].value
                            instanceuid = str(ds['SliceLocation'].value)
                            seriesNumber = str(ds.get("SeriesNumber","NA"))
                            instanceNumber = str(ds.get("InstanceNumber","0"))
                            fileName = "IM" + "-" + seriesNumber.zfill(4) + "-" + instanceNumber.zfill(4) + ".dcm"
                            if output:
                                directory = os.path.join(output,patientid,studyuid,seriesuid)
                            else:
                                directory = os.path.join(folder,patientid,studyuid,seriesuid)

                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            if cop:
                                try:
                                    shutil.copy(os.path.join(root,filename), os.path.join(directory,fileName))
                                except OSError as error:
                                    print(error)
                            else:
                                os.rename(os.path.join(root,filename), os.path.join(directory,fileName))
                        except:
                            print ("Error reading %s." % os.path.join(root,filename))
                            if output:
                                err_folder = os.path.join(output,"errors",root)
                            else:
                                err_folder = os.path.join(folder,"errors",root)
                            if not os.path.exists(err_folder):
                                os.makedirs(err_folder)
                            if cop:
                                shutil.copy(os.path.join(root,filename), os.path.join(err_folder,filename))
                            else:
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
    parser.add_argument('-o', dest='out_folder', metavar='out_folder', required=False, help="Path to output folder where to put the \
        sorted files. If not specified, original dicom folder is used.")
    parser.add_argument("--copy", dest='copy', help="Copy dicoms instead of moving them. This preserve original dicom hierarchy.", action="store_true")
    parser.add_argument("--clean", dest='clean', help="Clean empty directories after sorting dicoms.", action="store_true")



    op = parser.parse_args()

    st = Sorter()
    st.execute(op.in_folder, op.out_folder, op.copy)

    if op.clean:
        print("Cleaning empty folders...")
        if op.out_folder:
            st.clean_empty_folders(op.in_folder)
            st.clean_empty_folders(op.out_folder)
        else:
            st.clean_empty_folders(op.in_folder)

    print ("DONE.")