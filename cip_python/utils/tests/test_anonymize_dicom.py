import numpy as np
from cip_python.utils.anonymize_dicom import *
from pydicom.dataset import Dataset

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

def test_anonymize_dicom():
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

    ds = Dataset()
    if hasattr(ds, 'AddNew'): 
        ds.AddNew(Tag(0x0008, 0x0018), 'dicom.UID.UID', 'TestAnonymization')
        ds.AddNew(Tag(0x0008, 0x0080), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0008, 0x0081), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0008, 0x1010), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0008, 0x1030), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0008, 0x103e), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0010, 0x0010), 'dicom.valuerep.PersonName', 'TestAnonymization')
        ds.AddNew(Tag(0x0010, 0x0020), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0010, 0x0040), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0010, 0x1010), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0018, 0x1000), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0018, 0x1030), 'str', 'TestAnonymization')
        ds.AddNew(Tag(0x0020, 0x000d), 'dicom.UID.UID', 'TestAnonymization')
        ds.AddNew(Tag(0x0020, 0x000e), 'dicom.UID.UID', 'TestAnonymization')
        ds.AddNew(Tag(0x0020, 0x0052), 'dicom.UID.UID', 'TestAnonymization')
        ds.AddNew(Tag(0x0020, 0x4000), 'str', 'TestAnonymization')
    else:
        ds.add_new(Tag(0x0008, 0x0018), 'dicom.UID.UID', 'TestAnonymization')
        ds.add_new(Tag(0x0008, 0x0080), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0008, 0x0081), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0008, 0x1010), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0008, 0x1030), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0008, 0x103e), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0010, 0x0010), 'dicom.valuerep.PersonName', 'TestAnonymization')
        ds.add_new(Tag(0x0010, 0x0020), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0010, 0x0040), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0010, 0x1010), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0018, 0x1000), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0018, 0x1030), 'str', 'TestAnonymization')
        ds.add_new(Tag(0x0020, 0x000d), 'dicom.UID.UID', 'TestAnonymization')
        ds.add_new(Tag(0x0020, 0x000e), 'dicom.UID.UID', 'TestAnonymization')
        ds.add_new(Tag(0x0020, 0x0052), 'dicom.UID.UID', 'TestAnonymization')
        ds.add_new(Tag(0x0020, 0x4000), 'str', 'TestAnonymization')        

    anonymize_dicom(ds)

    keys = ds.keys()
    for t in anon_tags:
        if t in keys:
            assert ds[t].value == 'Anonymized' or ds[t].value == '', \
                'Dicom not anonymized'
