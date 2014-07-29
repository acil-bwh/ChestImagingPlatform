import os
import datetime
import pandas as pd

class Phenotypes:
  """Base class for phenotype genearting classes.
    
    Attributes
    ----------
    machine_ : string
        The machine name on which execution was performed

    os_name_ : string
        The operating system of the machine on which execution was performed

    os_version_ : string
        The operating system version of the machine on which execution was
        performed

    os_kernel_ : string
        The kernel of the machine on which execution was performed

    os_arch_ : string
        Architecture of the machine on which execution was performed

    run_time_stamp_ : string
        Info recording time of execution
  """
  def __init__(self):
    """
    """
    self.version_ = 'NaN' # TODO: Figure out how to obtain this
    self.machine_ = os.uname()[1]
    self.os_name_ = os.uname()[0]
    self.os_version_ = os.uname()[3]
    self.os_kernel_ = os.uname()[2]
    self.os_arch_ = os.uname()[4]
    self.run_time_stamp_ = datetime.datetime.now().isoformat()
    
    self._df = pd.DataFrame({'Version': [self.version_],
                             'Machine': [self.machine_],
                             'OS_Name': [self.os_name_],
                             'OS_Version': [self.os_version_],
                             'OS_Kernel': [self.os_kernel_],
                             'OS_Arch': [self.os_arch_],
                             'Run_TimeStamp': [self.run_time_stamp_]})
          
  def save_to_csv(self, filename):
    self._df.to_csv(filename, index=False)
  
  def execute(self):
    pass
  
