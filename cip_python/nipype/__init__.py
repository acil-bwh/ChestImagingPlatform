import os
if os.sys.platform == "win32":
  print ("Warning: nipype not available in Windows")
else:
  #import nipype
  from cip_python.nipype.cip_convention_manager import *
  from cip_python.nipype.cip_node import *

  from cip_python.nipype import interfaces
  #import workflows
