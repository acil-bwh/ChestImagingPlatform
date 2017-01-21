import os
if os.sys.platform == "win32":
  print ("Warning: nipype not available in Windows")
else:
  #import nipype
  from cip_convention_manager import *
  from cip_node import *

  import interfaces
  import workflows