import os
import datetime
import pandas

class Phenotypes:
  """Base class for phenotype genearting classes.
    
    Subclasses should overload the declare_quantities method to list the
    phenotypes quantities that the subclass is going to produce.
    Optionally, the subclass can overload the cid_generator method if
    the subclass has an internal way to generate the case id that
    identifies that phenotype.
    
    Parameters
    ----------
    cid: case id for the phenotypes that will be generated
  """
  def __init__(self, cid):
    
    self.cid = cid
    self.pheno_dict = dict()
    self.pheno_function = dict()
    self._pheno_static_names = list()
    self._pheno_static_functions = list()
    self._pheno_keys_names=['Region','Type']
    self._pheno_quantities_names = list()
      
    self._empty_val='NA'

    self.declare_static_cols()
    self.declare_key_cols()
    self.declare_pheno_cols(self.declare_quantities())
  
  
    #Temp code until we unified cipConventions with cpython
    self._region_hierarchy_top_bottom = dict()
    self._region_hierarchy_bottom_top = dict()
    self.__create_region_hierarchy()
    self.__create_region_type_list()
  
  def declare_static_cols(self):
    
    self._pheno_static_names.append('CID')
    self._pheno_static_functions.append(self.cid_generator)
    
    self._pheno_static_names.append('Generator')
    self._pheno_static_functions.append(self.generator)
    
    self._pheno_static_names.append('Version')
    self._pheno_static_functions.append(self.version)
    
    self._pheno_static_names.append('Machine')
    self._pheno_static_functions.append(self.machine)
    
    self._pheno_static_names.append('OS_Name')
    self._pheno_static_functions.append(self.os_name)
    
    self._pheno_static_names.append('OS_Version')
    self._pheno_static_functions.append(self.os_version)
    
    self._pheno_static_names.append('OS_Kernel')
    self._pheno_static_functions.append(self.os_kernel)
    
    self._pheno_static_names.append('OS_Arch')
    self._pheno_static_functions.append(self.os_arch)
    
    self._pheno_static_names.append('Run_TimeStamp')
    self._pheno_static_functions.append(self.run_timestamp)
    
    for key, function in zip(self._pheno_static_names,
                             self._pheno_static_functions):
      self.pheno_dict[key] = list()
      self.pheno_function[key]=function
    
  def declare_key_cols(self):
    for key in self._pheno_keys_names:
      self.pheno_dict[key] = list()
      self.pheno_dict[key] = list()
  
  def declare_pheno_cols(self, cols):
    for name in cols:
      self.pheno_dict[name] = list()
      self._pheno_quantities_names.append(name)

  def declare_quantities(self):
    cols = list()
    return cols

  def generator(self):
    return self.__class__.__name__
  
  def cid_generator(self):
    return self.cid
  
  def version(self):
    #Find how to grab git hashtag
    return 0

  def os_name(self):
    return os.uname()[0]

  def machine(self):
    return os.uname()[1]

  def os_kernel(self):
    return os.uname()[2]

  def os_version(self):
    return os.uname()[3]

  def os_arch(self):
    return os.uname()[4]

  def run_timestamp(self):
    return datetime.datetime.now().isoformat()

  def add_pheno(self, key_values, pheno_name, pheno_val):

    if len(key_values) != len(self._pheno_keys_names):
      raise "Wrong number of keys in tuple"
    
    # Find entry location (loc) by searching whether the key_values already
    # exists in list.
    index_set = set()
    for key_name, key_value in zip(self._pheno_keys_names,key_values):
      test = set([i for i, x in enumerate(self.pheno_dict[key_name]) \
                  if x == key_value])
      if len(test) == 0:
        break
      
      if len(index_set) == 0:
        index_set = test
      else:
        index_set = index_set.intersection(test)

    if len(index_set) == 0:
      loc=self.add_new_row(key_values)
    else:
      loc=index_set.pop()
        
    self.pheno_dict[pheno_name][loc] = pheno_val
    
  def add_new_row(self, key_values):

    for name in self._pheno_static_names:
      self.pheno_dict[name].append(self.pheno_function[name]())

    for key, key_val in zip(self._pheno_keys_names, key_values):
      self.pheno_dict[key].append(key_val)
    for name in self._pheno_quantities_names:
      self.pheno_dict[name].append(self._empty_val)

    return len(self.pheno_dict[name])-1
  
  
  def get_pheno_data_frame(self):

    cols = list()
    cols.extend(self._pheno_static_names)
    cols.extend(self._pheno_keys_names)
    cols.extend(self._pheno_quantities_names)
    
    df = pandas.DataFrame(self.pheno_dict, columns=cols)

    return df

  def save_to_csv(self, filename):
    df = self.get_pheno_data_frame()
    df.to_csv(filename, index=False)
  
  def execute(self):
    pass
  
  def region_has(self, region_value):
    if self._region_hierarchy_top_bottom.has_key(region_value):
      return self._region_hierarchy_top_bottom[region_value]
    else:
      return []

  def region_belongs(self, region_value):
    if self._region_hierarchy_bottom_top.has_key(region_value):
      return self._region_hierarchy_bottom_top[region_value]
    else:
      return []

  def __create_region_hierarchy(self):
    # In the meantime, this give us a working structure all within python
    # Multiple lists indicate and "or" relation
    self._region_hierarchy_top_bottom[1] = [[2, 3], [20, 21, 22]]
    self._region_hierarchy_top_bottom[2] = [[4, 5, 6], [14, 13, 12]]
    self._region_hierarchy_top_bottom[3] = [[7, 8], [9, 10, 11]]
    self._region_hierarchy_top_bottom[20] = [[9, 12]]
    self._region_hierarchy_top_bottom[21] = [[10, 13]]
    self._region_hierarchy_top_bottom[22] = [[11, 14]]
    
    self._region_hierarchy_bottom_top[2] = 1
    self._region_hierarchy_bottom_top[3] = 1
    self._region_hierarchy_bottom_top[4] = 2
    self._region_hierarchy_bottom_top[5] = 2
    self._region_hierarchy_bottom_top[6] = 2
    self._region_hierarchy_bottom_top[7] = 3
    self._region_hierarchy_bottom_top[8] = 3
    self._region_hierarchy_bottom_top[9] = [3, 20]
    self._region_hierarchy_bottom_top[10] = [3, 21]
    self._region_hierarchy_bottom_top[11] = [3, 22]
    self._region_hierarchy_bottom_top[12] = [2, 20]
    self._region_hierarchy_bottom_top[13] = [2, 21]
    self._region_hierarchy_bottom_top[14] = [2, 22]
    
    self._region_hierarchy_bottom_top[20] = 1
    self._region_hierarchy_bottom_top[21] = 1
    self._region_hierarchy_bottom_top[22] = 1

  def __create_region_type_list(self):
    self._region_name = list()
    self._region_name.append( "UNDEFINEDREGION" )
    self._region_name.append( "WHOLELUNG" )
    self._region_name.append( "RIGHTLUNG" )
    self._region_name.append( "LEFTLUNG" )
    self._region_name.append( "RIGHTSUPERIORLOBE" )
    self._region_name.append( "RIGHTMIDDLELOBE" )
    self._region_name.append( "RIGHTINFERIORLOBE" )
    self._region_name.append( "LEFTSUPERIORLOBE" )
    self._region_name.append( "LEFTINFERIORLOBE" )
    self._region_name.append( "LEFTUPPERTHIRD" )
    self._region_name.append( "LEFTMIDDLETHIRD" )
    self._region_name.append( "LEFTLOWERTHIRD" )
    self._region_name.append( "RIGHTUPPERTHIRD" )
    self._region_name.append( "RIGHTMIDDLETHIRD" )
    self._region_name.append( "RIGHTLOWERTHIRD" )
    self._region_name.append( "MEDIASTINUM" )
    self._region_name.append( "WHOLEHEART" )
    self._region_name.append( "AORTA" )
    self._region_name.append( "PULMONARYARTERY" )
    self._region_name.append( "PULMONARYVEIN" )
    self._region_name.append( "UPPERTHIRD" )
    self._region_name.append( "MIDDLETHIRD" )
    self._region_name.append( "LOWERTHIRD" )
    self._region_name.append( "LEFT" )
    self._region_name.append( "RIGHT" )
    self._region_name.append( "LIVER" )
    self._region_name.append( "SPLEEN" )
    self._region_name.append( "ABDOMEN" )
    self._region_name.append( "PARAVERTEBRAL" )
    
    self._type_name = list()
    self._type_name.append( "UNDEFINEDTYPE" )
    self._type_name.append( "NORMALPARENCHYMA" )
    self._type_name.append( "AIRWAY" )
    self._type_name.append( "VESSEL" )
    self._type_name.append( "EMPHYSEMATOUS" )
    self._type_name.append( "GROUNDGLASS" )
    self._type_name.append( "RETICULAR" )
    self._type_name.append( "NODULAR" )
    self._type_name.append( "OBLIQUEFISSURE" )
    self._type_name.append( "HORIZONTALFISSURE" )
    self._type_name.append( "MILDPARASEPTALEMPHYSEMA" )
    self._type_name.append( "MODERATEPARASEPTALEMPHYSEMA" )
    self._type_name.append( "SEVEREPARASEPTALEMPHYSEMA" )
    self._type_name.append( "MILDBULLA" )
    self._type_name.append( "MODERATEBULLA" )
    self._type_name.append( "SEVEREBULLA" )
    self._type_name.append( "MILDCENTRILOBULAREMPHYSEMA" )
    self._type_name.append( "MODERATECENTRILOBULAREMPHYSEMA" )
    self._type_name.append( "SEVERECENTRILOBULAREMPHYSEMA" )
    self._type_name.append( "MILDPANLOBULAREMPHYSEMA" )
    self._type_name.append( "MODERATEPANLOBULAREMPHYSEMA" )
    self._type_name.append( "SEVEREPANLOBULAREMPHYSEMA" )
    self._type_name.append( "AIRWAYWALLTHICKENING" )
    self._type_name.append( "AIRWAYCYLINDRICALDILATION" )
    self._type_name.append( "VARICOSEBRONCHIECTASIS" )
    self._type_name.append( "CYSTICBRONCHIECTASIS" )
    self._type_name.append( "CENTRILOBULARNODULE" )
    self._type_name.append( "MOSAICING" )
    self._type_name.append( "EXPIRATORYMALACIA" )
    self._type_name.append( "SABERSHEATH" )
    self._type_name.append( "OUTPOUCHING" )
    self._type_name.append( "MUCOIDMATERIAL" )
    self._type_name.append( "PATCHYGASTRAPPING" )
    self._type_name.append( "DIFFUSEGASTRAPPING" )
    self._type_name.append( "LINEARSCAR" )
    self._type_name.append( "CYST" )
    self._type_name.append( "ATELECTASIS" )
    self._type_name.append( "HONEYCOMBING" )
    self._type_name.append( "TRACHEA" )
    self._type_name.append( "MAINBRONCHUS" )
    self._type_name.append( "UPPERLOBEBRONCHUS" )
    self._type_name.append( "AIRWAYGENERATION3" )
    self._type_name.append( "AIRWAYGENERATION4" )
    self._type_name.append( "AIRWAYGENERATION5" )
    self._type_name.append( "AIRWAYGENERATION6" )
    self._type_name.append( "AIRWAYGENERATION7" )
    self._type_name.append( "AIRWAYGENERATION8" )
    self._type_name.append( "AIRWAYGENERATION9" )
    self._type_name.append( "AIRWAYGENERATION10" )
    self._type_name.append( "CALCIFICATION" )
    self._type_name.append( "ARTERY" )
    self._type_name.append( "VEIN" )
    self._type_name.append( "PECTORALISMINOR" )
    self._type_name.append( "PECTORALISMAJOR" )
    self._type_name.append( "ANTERIORSCALENE" )
    self._type_name.append( "FISSURE" )
    self._type_name.append( "VESSELGENERATION0" )
    self._type_name.append( "VESSELGENERATION1" )
    self._type_name.append( "VESSELGENERATION2" )
    self._type_name.append( "VESSELGENERATION3" )
    self._type_name.append( "VESSELGENERATION4" )
    self._type_name.append( "VESSELGENERATION5" )
    self._type_name.append( "VESSELGENERATION6" )
    self._type_name.append( "VESSELGENERATION7" )
    self._type_name.append( "VESSELGENERATION8" )
    self._type_name.append( "VESSELGENERATION9" )
    self._type_name.append( "VESSELGENERATION10" )
    self._type_name.append( "PARASEPTALEMPHYSEMA" )
    self._type_name.append( "CENTRILOBULAREMPHYSEMA" )
    self._type_name.append( "PANLOBULAREMPHYSEMA" )
    self._type_name.append( "SUBCUTANEOUSFAT" )
    self._type_name.append( "VISCERALFAT" )
    self._type_name.append( "INTERMEDIATEBRONCHUS" )
    self._type_name.append( "LOWERLOBEBRONCHUS" )
    self._type_name.append( "SUPERIORDIVISIONBRONCHUS" )
    self._type_name.append( "LINGULARBRONCHUS" )
    self._type_name.append( "MIDDLELOBEBRONCHUS" )
    self._type_name.append( "BRONCHIECTATICAIRWAY" )
    self._type_name.append( "NONBRONCHIECTATICAIRWAY" )
    self._type_name.append( "AMBIGUOUSBRONCHIECTATICAIRWAY" )
    self._type_name.append( "MUSCLE" )
    self._type_name.append( "DIAPHRAGM" )
