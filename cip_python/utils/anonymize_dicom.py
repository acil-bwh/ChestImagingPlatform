from os import listdir
from os.path import isfile, join, isdir
from optparse import OptionParser
import pydicom as dicom
from pydicom.tag import Tag

def anonymize_dicom(ds):
    """Anonymizes a dicom dataset. A value of 'Anonymized' is set for all tags
    that take a string-type entry. Other tags are set to empty entries.

    Parameters
    ----------
    ds : dicom dataset
        Dataset as returned from a dicom.read_file command

    References
    ----------
    For a description of the tags that are anonymized, see:
    ftp://medical.nema.org/medical/dicom/final/sup55_ft.pdf
    """
    anon_tags = [\
                 Tag(0x0008, 0x0014), # Instance Creator UID 
                 Tag(0x0008, 0x0018), # SOP Instance UID 
                 Tag(0x0008, 0x0050), # Accession Number 
                 Tag(0x0008, 0x0080), # Institution Name 
                 Tag(0x0008, 0x0081), # Institution Address
                 Tag(0x0008, 0x0090), # Referring Physician's Name
                 Tag(0x0008, 0x0092), # Referring Physician's Address 
                 Tag(0x0008, 0x0094), # Referring Physician's Telephone Numbers 
                 Tag(0x0008, 0x1010), # Station Name 
                 Tag(0x0008, 0x1030), # Study Description 
                 Tag(0x0008, 0x103E), # Series Description 
                 Tag(0x0008, 0x1040), # Institutional Department Name 
                 Tag(0x0008, 0x1048), # Physician(s) of Record 
                 Tag(0x0008, 0x1050), # Performing Physicians' Name 
                 Tag(0x0008, 0x1060), # Name of Physician(s) Reading Study 
                 Tag(0x0008, 0x1070), # Operators' Name 
                 Tag(0x0008, 0x1080), # Admitting Diagnoses Description 
                 Tag(0x0008, 0x1155), # Referenced SOP Instance UID 
                 Tag(0x0008, 0x2111), # Derivation Description 
                 Tag(0x0010, 0x0010), # Patient's Name 
                 Tag(0x0010, 0x0020), # Patient ID 
                 Tag(0x0010, 0x0030), # Patient's Birth Date 
                 Tag(0x0010, 0x0032), # Patient's Birth Time 
                 Tag(0x0010, 0x0040), # Patient's Sex 
                 Tag(0x0010, 0x1000), # Other Patient Ids 
                 Tag(0x0010, 0x1001), # Other Patient Names 
                 Tag(0x0010, 0x1010), # Patient's Age 
                 Tag(0x0010, 0x1020), # Patient's Size 
                 Tag(0x0010, 0x1030), # Patient's Weight 
                 Tag(0x0010, 0x1090), # Medical Record Locator 
                 Tag(0x0010, 0x2160), # Ethnic Group 
                 Tag(0x0010, 0x2180), # Occupation 
                 Tag(0x0010, 0x21B0), # Additional Patient's History 
                 Tag(0x0010, 0x4000), # Patient Comments 
                 Tag(0x0018, 0x1000), # Device Serial Number 
                 Tag(0x0018, 0x1030), # Protocol Name 
                 Tag(0x0020, 0x000D), # Study Instance UID 
                 Tag(0x0020, 0x000E), # Series Instance UID 
                 Tag(0x0020, 0x0010), # Study ID 
                 Tag(0x0020, 0x0052), # Frame of Reference UID 
                 Tag(0x0020, 0x0200), # Synchronization Frame of Reference UID 
                 Tag(0x0020, 0x4000), # Image Comments 
                 Tag(0x0040, 0x0275), # Request Attributes Sequence 
                 Tag(0x0040, 0xA124), # UID 
                 Tag(0x0040, 0xA730), # Content Sequence 
                 Tag(0x0088, 0x0140), # Storage Media File-set UID 
                 Tag(0x3006, 0x0024), # Referenced Frame of Reference UID 
                 Tag(0x3006, 0x00C2)] # Related Frame of Reference UID 
    
    keys = ds.keys()

    for t in anon_tags:
        if t in keys:
            if type(ds[t].value) == str or \
                type(ds[t].value) == dicom.valuerep.PersonName or \
                type(ds[t].value) == dicom.uid.UID:
                ds[t].value = 'Anonymized'
            else:
                ds[t].value = ''

if __name__ == "__main__":
    desc = """Anonymize a dicom image or all dicom images within a directory"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_im',\
                      help='Input dicom image',\
                      dest='in_im', metavar='<string>', default=None)
    parser.add_option('--out_im',\
                      help='Output dicom image. If none specified, the input \
                      file name will be used as the output file name',\
                      dest='out_im', metavar='<string>', default=None)
    parser.add_option('--in_dir',\
                      help='Input dicom directory. All dicom images inside \
                      this directory will be anonymized. It is assumed that \
                      all files inside this directory are dicom files.',\
                      dest='in_dir', metavar='<string>', default=None)
    parser.add_option('--out_dir',\
                      help='Output dicom directory. If none specified, the input \
                      directory be used as the output directory, overwriting \
                      the dicom images that were read from the directory.',\
                      dest='out_dir', metavar='<string>', default=None)    

    (options, args) = parser.parse_args()

    if options.in_im is not None:
        assert isfile(options.in_im), 'File does not exist'
        ds = dicom.read_file(options.in_im)
        anonymize_dicom(ds)
        if options.out_im is not None:
            dicom.write_file(options.out_im, ds)

    if options.in_dir is not None:
        assert isdir(options.in_dir), 'Directory does not exist'
        files = [f for f in listdir(options.in_dir) if \
                 isfile(join(options.in_dir, f))]
        for f in files:
            ds = dicom.read_file(join(options.in_dir, f))
            anonymize_dicom(ds)
            if options.out_dir is not None:
                assert isdir(options.out_dir), 'Directory does not exist'
                dicom.write_file(join(options.out_dir, f), ds)
