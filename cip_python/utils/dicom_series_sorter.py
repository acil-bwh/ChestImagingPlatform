import os
import pydicom
import argparse
import shutil

class Sorter():
    def __init__(self):
        """ Sorter initialization """
        self.processed_slice_locations = {}

    def execute(self, folder, output, cop):
        for root, dirs, files in os.walk(folder):
            if not dirs:
                for filename in files:
                    if filename.startswith('.'):
                        continue
                    try:
                        filepath = os.path.join(root, filename)
                        ds = pydicom.dcmread(filepath)
                        
                        patientid = ds['PatientID'].value
                        studyuid = ds['StudyInstanceUID'].value
                        seriesuid = ds['SeriesInstanceUID'].value
                        slice_location = ds.get('SliceLocation', None)
                        series_number = str(ds.get("SeriesNumber", "NA"))
                        instance_number = str(ds.get("InstanceNumber", "0"))
                        file_name = f"IM-{series_number.zfill(4)}-{instance_number.zfill(4)}.dcm"

                        if output:
                            directory = os.path.join(output, patientid, studyuid, seriesuid)
                        else:
                            directory = os.path.join(folder, patientid, studyuid, seriesuid)

                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        # Skip if slice location already processed for the series
                        series_key = (patientid, studyuid, seriesuid)
                        if series_key not in self.processed_slice_locations:
                            self.processed_slice_locations[series_key] = set()

                        if slice_location in self.processed_slice_locations[series_key]:
                            continue
                        
                        self.processed_slice_locations[series_key].add(slice_location)

                        # Copy or move the file
                        dest_path = os.path.join(directory, file_name)
                        if cop:
                            shutil.copy(filepath, dest_path)
                        else:
                            os.rename(filepath, dest_path)
                        
                    except Exception as e:
                        print(f"Error reading {os.path.join(root, filename)}: {e}")
                        if output:
                            err_folder = os.path.join(output, "errors", root)
                        else:
                            err_folder = os.path.join(folder, "errors", root)
                        if not os.path.exists(err_folder):
                            os.makedirs(err_folder)
                        if cop:
                            shutil.copy(filepath, os.path.join(err_folder, filename))
                        else:
                            os.rename(filepath, os.path.join(err_folder, filename))

    def clean_empty_folders(self, folder):
        for root, dirs, files in os.walk(folder, topdown=False):
            try:
                os.removedirs(root)
            except NotADirectoryError:
                print(f"{root} is not a directory.")
            except PermissionError:
                print(f"Permission denied for {root}.")
            except OSError:
                print(f"Directory {root} cannot be removed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sort dicom series in different folders.')
    parser.add_argument('-i', dest='in_folder', metavar='in_folder', required=True, help="Path to folder containing DICOM hierarchy.")
    parser.add_argument('-o', dest='out_folder', metavar='out_folder', required=False, help="Path to output folder where to put the sorted files. If not specified, original DICOM folder is used.")
    parser.add_argument("--copy", dest='copy', help="Copy DICOMs instead of moving them. This preserves the original DICOM hierarchy.", action="store_true")
    parser.add_argument("--clean", dest='clean', help="Clean empty directories after sorting DICOMs.", action="store_true")

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

    print("DONE.")