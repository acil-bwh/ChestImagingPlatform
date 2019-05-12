import os, sys, nrrd, tempfile, shutil, vtk, nrrd, pdb
import nibabel as nb
import numpy as np
from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
from cip_python.phenotypes import ParenchymaPhenotypes
from cip_python.phenotypes import BodyCompositionPhenotypes
from cip_python.phenotypes.fissure_phenotypes import FissurePhenotypes
from cip_python.particles import FissureParticles

import warnings

# example http://nipy.sourceforge.net/nipype/devel/python_interface_devel.html
class fissure_particlesInputSpec(BaseInterfaceInputSpec):
    ict = File(exists=True, desc='Input CT file', mandatory=True)
    ilm = File(exists=True, desc='Input mask for seeding', mandatory=True)
    tmp = traits.Str(desc='Temp directory in which to store intermediate files', 
        mandatory=False)
    scale = traits.Float(desc='The scale of the fissure to consider in scale \
      space', mandatory=False)
    rate = traits.Float(desc='Down sampling rate. Must be greater than \
      or equal to 1.0 (default 1.0)', mandatory=False)
    lth = traits.Float(desc='Threshold to use when pruning particles during \
      population control. Must be less than zero', mandatory=False)
    sth = traits.Float(desc='Threshold to use when initializing particles. \
      Must be less than zero', mandatory=False)
    min_int = traits.Int(desc='Histogram equilization will be applied to \
      enhance the fissure features within the range [min_int, max_int]', 
      mandatory=False)     
    max_int = traits.Int(desc='Histogram equilization will be applied to \
      enhance the fissure features within the range [min_int, max_int]', 
      mandatory=False)     
    iters = traits.Int(desc='Number of algorithm iterations', mandatory=False)
    perm = traits.Bool(desc="Allow mask and CT volumes to have different \
      shapes or meta data.", argstr="--perm ")
    
class fissure_particlesOutputSpec(TraitedSpec):      
    op = File(desc='Output particles (vtk format)', mandatory=False)
    
class fissure_particles(BaseInterface):
    input_spec = fissure_particlesInputSpec
    output_spec = fissure_particlesOutputSpec
    
    def _run_interface(self, runtime):
        particles_tmp_dir = tempfile.mkdtemp()

        wf_tmp_dir, _, _ = split_filename(self.inputs.ilm)
        op = os.path.join(wf_tmp_dir, 'cid_fissureParticles.vtk')

        dp = FissureParticles(self.inputs.ict, op, particles_tmp_dir, 
            self.inputs.ilm, float(self.inputs.scale), float(self.inputs.lth),
            float(self.inputs.sth), float(self.inputs.rate), 
            float(self.inputs.min_int), float(self.inputs.max_int), 
            int(self.inputs.iters))
        dp.execute()

        shutil.rmtree(particles_tmp_dir)

        return runtime
            
    def _list_outputs(self):
        outputs = self._outputs().get()

        wf_tmp_dir, _, _ = split_filename(self.inputs.ilm)
        outputs['op'] = os.path.join(wf_tmp_dir, 'cid_fissureParticles.vtk')
        
        return outputs
    
class parenchyma_phenotypesInputSpec(BaseInterfaceInputSpec):
    in_ct = File(exists=True, desc='Input CT file', mandatory=True)
    in_lm = File(exists=True, 
        desc='Input label map containing structures of interest', 
        mandatory=True)
    out_csv = \
      File(desc='Output csv file in which to store the computed dataframe',
           mandatory=True)
    cid = traits.Str(desc='The database case ID', mandatory=True)
    chest_regions = traits.Str(desc='Chest regions', mandatory=False)
    chest_types = traits.Str(desc='Chest types', mandatory=False)
    pairs = traits.Str(desc='Chest region/type pairs', mandatory=False)
    pheno_names = traits.Str(desc='Phenotype names', mandatory=False)
    out_csv = \
      File(desc='Output csv file in which to store the computed dataframe', 
           mandatory=False)

class parenchyma_phenotypesOutputSpec(TraitedSpec):
    out_csv = \
      File(desc='Output csv file in which to store the computed dataframe', 
           mandatory=False)

class parenchyma_phenotypes(BaseInterface):
    input_spec = parenchyma_phenotypesInputSpec
    output_spec = parenchyma_phenotypesOutputSpec
    
    def _run_interface(self, runtime):
        lm, lm_header = nrrd.read(self.inputs.in_lm)
        ct, ct_header = nrrd.read(self.inputs.in_ct)
    
        spacing = np.zeros(3)
        spacing[0] = ct_header['space directions'][0][0]
        spacing[1] = ct_header['space directions'][1][1]
        spacing[2] = ct_header['space directions'][2][2]
    
        regions = None
        print(self.inputs.chest_regions)
        if hasattr(self, 'inputs.chest_regions'):
        #if self.inputs.chest_regions is not '_Undefined':
            regions = self.inputs.chest_regions.split(',')
        types = None
        print(self.inputs.chest_types)
        if hasattr(self, 'inputs.chest_types'):
        #if self.inputs.chest_types is not '_Undefined':
            types = self.inputs.chest_types.split(',')
        pairs = None
        if hasattr(self, 'inputs.pairs'):
        #if self.inputs.options.pairs is not None:
            tmp = pairs.split(',')
            assert len(tmp)%2 == 0, 'Specified pairs not understood'
            pairs = []
            for i in xrange(0, len(tmp)/2):
                pairs.append([tmp[2*i], tmp[2*i+1]])
        pheno_names = None
        if self.inputs.pheno_names is not None:
            pheno_names = self.inputs.pheno_names.split(',')
    
        paren_pheno = ParenchymaPhenotypes(chest_regions=regions,  #self.inputs.chest_regions,
            chest_types=types, pairs=pairs, pheno_names=pheno_names)
    
        df = paren_pheno.execute(ct, lm, self.inputs.cid, spacing)
    
        if self.inputs.out_csv is not None:
            df.to_csv(self.inputs.out_csv, index=False)
                
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.out_csv
        _, base, _ = split_filename(fname)
        outputs["out_csv"] = os.path.abspath(fname)
        return outputs

"""
Body composition phenotyopes
"""
class body_composition_phenotypesInputSpec(BaseInterfaceInputSpec):
    in_ct = File(exists=True, desc='Input CT file', mandatory=True)
    in_lm = File(exists=True, \
        desc='Input label map containing structures of interest', 
        mandatory=True)
    out_csv = \
      File(desc='Output csv file in which to store the computed dataframe', 
           mandatory=True)
    cid = traits.Str(desc='The database case ID', mandatory=True)
    chest_regions = traits.Str(desc='Chest regions', mandatory=False)
    chest_types = traits.Str(desc='Chest types', mandatory=False)
    pairs = traits.Str(desc='Chest region/type pairs', mandatory=False)
    pheno_names = traits.Str(desc='Phenotype names', mandatory=False)

class body_composition_phenotypesOutputSpec(TraitedSpec):
    out_csv = \
      File(desc='Output csv file in which to store the computed dataframe', 
           mandatory=False)

class body_composition_phenotypes(BaseInterface):
    input_spec = parenchyma_phenotypesInputSpec
    output_spec = parenchyma_phenotypesOutputSpec
    
    def _run_interface(self, runtime):        
        lm, lm_header = nrrd.read(self.inputs.in_lm)
        ct, ct_header = nrrd.read(self.inputs.in_ct)
        
        spacing = np.zeros(3)
        spacing[0] = ct_header['space directions'][0][0]
        spacing[1] = ct_header['space directions'][1][1]
        spacing[2] = ct_header['space directions'][2][2]
        
        pheno_names = None
        if self.inputs.pheno_names is not None:
            pheno_names = self.inputs.pheno_names.split(',')
        
        body_pheno = BodyCompositionPhenotypes(chest_regions=self.inputs.chest_regions,
                                           chest_types=None, pairs=None, pheno_names=pheno_names)
        
        df = body_pheno.execute(ct, lm, self.inputs.cid, spacing)
        
        if self.inputs.out_csv is not None:
            df.to_csv(self.inputs.out_csv, index=False)
            
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.out_csv
        _, base, _ = split_filename(fname)
        outputs["out_csv"] = os.path.abspath(fname)

        return outputs

"""
Fissure phenotyopes
"""
class fissure_phenotypesInputSpec(BaseInterfaceInputSpec):
    in_lm = File(exists=True, desc='Lung lobe label map', 
        mandatory=True)
    out_csv = \
      File(desc='Output csv file in which to store the computed dataframe', 
           mandatory=True)    
    cid = traits.Str(desc='Case ID', mandatory=True)
    down = traits.Int(desc='The amount by which to downsample the surface \
        points before computing the TPS (1 -> no downsampling, 2 -> half the \
        points will be used, etc). This option is irrelevant if completeness \
        type is set to domain.', mandatory=False)
    type = traits.Str(desc="Either 'surface', in which case the surface \
        area of the lobe boundaries and fissure regions will be compared, or \
        'count', in which case only the voxel counts of the boundary and \
        fissure regions will be compared.", mandatory=False)
    lop = File(exists=True, desc='Left oblique points file (vtk). If \
        specified, the surface defined by these points will override the \
        surface defined in the label map image.', mandatory=False)
    rop = File(exists=True, desc='Right oblique points file (vtk). If \
        specified, the surface defined by these points will override the \
        surface defined in the label map image.', mandatory=False)
    rhp = File(exists=True, desc='Right horizontal points file (vtk). If \
        specified, the surface defined by these points will override the \
        surface defined in the label map image.', mandatory=False)        
    ray_tol = traits.Float(desc='A particle mesh point must be within this \
        distance of the lobe boundary to be considered fissure. If fissure \
        point polydata are not being used to define the fissure surface, \
        this parameter is irrelevant.', mandatory=False)

class fissure_phenotypesOutputSpec(TraitedSpec):
    out_csv = \
      File(desc='Output csv file in which to store the computed dataframe', 
           mandatory=False)
    
class fissure_phenotypes(BaseInterface):
    input_spec = fissure_phenotypesInputSpec
    output_spec = fissure_phenotypesOutputSpec
    
    def _run_interface(self, runtime):        
        lm, lm_header = nrrd.read(self.inputs.in_lm)

        if 'spacing' in lm_header.keys():
            spacing = lm_header['spacing']
        elif 'space directions' in lm_header.keys():
            spacing = np.diag(lm_header['space directions'])
        
        lop_poly = None
        if self.inputs.lop is not None:
            lop_reader = vtk.vtkPolyDataReader()
            lop_reader.SetFileName(self.inputs.lop)
            lop_reader.Update()
            lop_poly = lop_reader.GetOutput()

        rop_poly = None
        if self.inputs.rop is not None:
            rop_reader = vtk.vtkPolyDataReader()
            rop_reader.SetFileName(self.inputs.rop)
            rop_reader.Update()
            rop_poly = rop_reader.GetOutput()

        rhp_poly = None
        if self.inputs.rhp is not None:
            rhp_reader = vtk.vtkPolyDataReader()
            rhp_reader.SetFileName(self.inputs.rhp)
            rhp_reader.Update()
            rhp_poly = rhp_reader.GetOutput()        
        
        completeness_phenos = FissurePhenotypes()

        df = completeness_phenos.execute(lm, lm_header['space origin'],
          spacing, self.inputs.cid, self.inputs.type,
          int(self.inputs.down), lop_poly=lop_poly, rop_poly=rop_poly,
          rhp_poly=rhp_poly)
        df.to_csv(self.inputs.out_csv, index=False)
                            
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.out_csv
        _, base, _ = split_filename(fname)
        outputs["out_csv"] = os.path.abspath(fname)

        return outputs    

#########################################################################
"""
input: case_id, input file, convention
outputs are case_id_{convention}.nhdr and case_id_{convention}.raw.gz 

assumption: input file has the formet {somename}_{convention}.nhdr, where
there are no underscores in "{somename}"
"""

class nhdr_handlerInputSpec(BaseInterfaceInputSpec):
    in_nhdr = File(exists=True, desc='Input nhdr file', mandatory=True)
    case_id = File( desc='Input case_id', mandatory=False)
    out_nhdr = File( desc='Output nhdr renamed file', mandatory=False)
    out_rawgz = File( desc='Output raw.gz renamed file', mandatory=False)                    
                                                            
class nhdr_handlerOutputSpec(TraitedSpec):
    out_nhdr = File( desc='Output nhdr renamed file', mandatory=True)
    out_rawgz = File( desc='Output raw.gz renamed file', mandatory=False)

class nhdr_handler(BaseInterface):
    input_spec = nhdr_handlerInputSpec
    output_spec = nhdr_handlerOutputSpec
    
    def _run_interface(self, runtime):
               
        if(len(self.inputs.in_nhdr.split(".")[0].split("_"))>1):
            suffix = ('_')+self.inputs.in_nhdr.split(".")[0].split("_")[-1]
        else:
            suffix=""
        self.output_spec.out_nhdr =  os.path.abspath(self.inputs.case_id+suffix+".nhdr")  
        self.output_spec.out_rawgz =   os.path.abspath(self.inputs.case_id+suffix+'.raw.gz')            
        
        print("************************")
        print(self.inputs.in_nhdr)
        print("************************")
        print(self.output_spec.out_nhdr )
        print("************************")
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()       
        if(len(self.inputs.in_nhdr.split(".")[0].split("_"))>1):
            suffix = ('_')+self.inputs.in_nhdr.split(".")[0].split("_")[-1]
        else:
            suffix=""
            
        outputs["out_nhdr"] = os.path.abspath(self.inputs.case_id+suffix+".nhdr")  
        outputs["out_rawgz"] = os.path.abspath(self.inputs.case_id+suffix+'.raw.gz')     
        #pdb.set_trace()
        return outputs


"""
input is some temp_{convention}.nhdr and a case_id.
outputs are case_id_{convention}.nhdr and case_id.raw.gz 
"""

