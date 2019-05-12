from argparse import ArgumentParser
import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.unu as unu
import cip_python.nipype.interfaces.cip.cip_python_interfaces \
  as cip_python_interfaces
import nipype.pipeline.engine as pe
from nipype.pipeline.engine import Workflow
import tempfile, shutil, sys, os, pickle, gzip

class FissurePhenotypesWorkflow(Workflow):
    """Nipype workflow that manages computation of fissure completeness 
    phenotypes given a lung lobe label map and a fissure detection or feature 
    strength volume.

    Parameters
    ----------
    lobes_file_name : str
        The file name of the lung lobe label mape.

    fissures_file_name : str
        Volume containing fissure detections (binary: 1 = fissure, 
        0 = background) or fissure feature strength (continuous on [0, 1]). 
        The volume will be blurred before being passed to the fissure particles
        routine.

    out_csv : str
        Output csv file in which to store the computed dataframe
                     
    cid : str
        Case ID

    dist_thresh : float
        A particle will be classified as fissure if the Fischer linear 
        discrimant classifier classifies it as such and if it is within this
        distance to the lobe surface model.

    ray_tol : float
        A particle mesh point must be within this distance of the lobe 
        boundary to be considered fissure. If fissure point polydata are not 
        being used to define the fissure surface, this parameter is irrelevant.

    tmp_dir : str
        Temporary directory that contains intermediate files.
    """
    def __init__(self, lobes_file_name, fissures_file_name, out_csv,
                     cid, dist_thresh, ray_tol, tmp_dir):

        Workflow.__init__(self, 'FissurePhenotypesWorkflow')

        lobes_file_name = os.path.realpath(lobes_file_name)
        fissures_file_name = os.path.realpath(fissures_file_name)
        self._blurred_file_name = os.path.join(tmp_dir, cid + '_blurred.nrrd')
        self._blurred_scaled_file_name = \
          os.path.join(tmp_dir, cid + '_blurredScaled.nrrd')
        self._seed_mask_file_name = \
          os.path.join(tmp_dir, cid + '_seedMask.nrrd')
        self._particles_file_name = \
          os.path.join(tmp_dir, cid + '_particles.vtk')
        self._lo_file_name = \
          os.path.join(tmp_dir, cid + '_leftObliqueParticles.vtk')
        self._ro_file_name = \
          os.path.join(tmp_dir, cid + '_rightObliqueParticles.vtk')
        self._rh_file_name = \
          os.path.join(tmp_dir, cid + '_rightHorizontalParticles.vtk')          
          
        file_and_path = os.path.abspath(__file__)
        file_path = os.path.dirname(file_and_path)

        #----------------------------------------------------------------------
        # Create node for blurring the fissure detections / feature strength
        # volume
        #----------------------------------------------------------------------
        unu_resample = pe.Node(interface=unu.unu_resample(),
                                   name='unu_resample')
        unu_resample.inputs.input = fissures_file_name
        unu_resample.inputs.size = 'x1 x1 x1'
        unu_resample.inputs.kernel = 'gauss:1,3'
        unu_resample.inputs.output = self._blurred_file_name

        #----------------------------------------------------------------------
        # Create node for scaling the blurred fissure volume
        #----------------------------------------------------------------------
        unu_2op_rescale = pe.Node(interface=unu.unu_2op(),
                                      name='unu_2op_rescale')
        unu_2op_rescale.inputs.operator = 'x'
        unu_2op_rescale.inputs.in2_scalar = 10000
        unu_2op_rescale.inputs.output = self._blurred_scaled_file_name
        
        #----------------------------------------------------------------------
        # Create node for generating mask for fissure particles
        #----------------------------------------------------------------------
        unu_2op_mask = pe.Node(interface=unu.unu_2op(),
                                   name='unu_2op_mask')
        unu_2op_mask.inputs.in1_file = fissures_file_name
        unu_2op_mask.inputs.operator = 'gt'        
        unu_2op_mask.inputs.in2_scalar = 0.5
        unu_2op_mask.inputs.output = self._seed_mask_file_name        
        
        #----------------------------------------------------------------------
        # Create node for running fissure particles
        #----------------------------------------------------------------------
        fissure_particles_node = \
          pe.Node(interface=cip_python_interfaces.fissure_particles(), 
                  name='fissure_particles_node')
        fissure_particles_node.inputs.scale = 3
        fissure_particles_node.inputs.lth = -15
        fissure_particles_node.inputs.sth = -30
        fissure_particles_node.inputs.rate = 1.0
        fissure_particles_node.inputs.min_int = 0
        fissure_particles_node.inputs.max_int = 10000
        fissure_particles_node.inputs.iters = 300
        fissure_particles_node.inputs.perm
        fissure_particles_node.inputs.ilm = lobes_file_name
        
        #----------------------------------------------------------------------
        # Create node for classifying particles
        #----------------------------------------------------------------------
        classify_particles = \
          pe.Node(interface=cip.ClassifyFissureParticles(),
                      name='classify_particles')
        if dist_thresh is not None:
            classify_particles.inputs.dist_thresh = dist_thresh
        classify_particles.inputs.lm = lobes_file_name
        classify_particles.inputs.olo = self._lo_file_name
        classify_particles.inputs.oro = self._ro_file_name
        classify_particles.inputs.orh = self._rh_file_name        
        
        #----------------------------------------------------------------------
        # Create the fissure completeness phenotype node
        #----------------------------------------------------------------------
        phenotypes_node = \
          pe.Node(interface=cip_python_interfaces.fissure_phenotypes(), 
                  name='phenotypes_node')        
        phenotypes_node.inputs.cid = cid
        phenotypes_node.inputs.down = 5
        phenotypes_node.inputs.type = 'surface'
        phenotypes_node.inputs.in_lm = lobes_file_name
        phenotypes_node.inputs.out_csv = out_csv
        
        #----------------------------------------------------------------------
        # Connect the nodes
        #----------------------------------------------------------------------
        self.connect(unu_resample, 'output', unu_2op_rescale, 'in1_file')
        self.connect(unu_2op_rescale, 'output', fissure_particles_node, 'ict')
        self.connect(unu_2op_mask, 'output', fissure_particles_node, 'ilm')
        self.connect(fissure_particles_node, 'op', classify_particles, 'ifp')
        self.connect(classify_particles, 'olo', phenotypes_node, 'lop')
        self.connect(classify_particles, 'oro', phenotypes_node, 'rop')
        self.connect(classify_particles, 'orh', phenotypes_node, 'rhp')

if __name__ == "__main__":
    desc = """Nipype workflow that manages computation of fissure \
    completeness phenotypes given a lung lobe label map and a fissure \
    detection or feature strength volume."""

    parser = ArgumentParser(description=desc)
    parser.add_argument('--in_fiss', help='Fissure detections or fissure \
      strength volume', dest='in_fiss', metavar='<string>', required=True)
    parser.add_argument('--in_lobes', help='Lung lobe label map file name. It \
      is assumed that this is a volume of unsigned short data type with \
      structures labeled according to the CIP labeling conventions',
      dest='in_lobes', metavar='<string>', required=True)
    parser.add_argument('--out_csv', help='Output csv file in which to store \
      the computed dataframe', dest='out_csv', metavar='<string>',
      required=True)
    parser.add_argument('--cid', dest='cid', required=True, help='Case id')
    parser.add_argument('--ray_tol', dest='ray_tol', required=False, \
        default=None, help='A particle mesh point must be within this \
        distance of the lobe boundary to be considered fissure. If fissure \
        point polydata are not being used to define the fissure surface, \
        this parameter is irrelevant.')    
    parser.add_argument('--dist_thresh', dest='dist_thresh', required=False,
        help='A particle will be classified as fissure if the Fischer linear \
        discrimant classifier classifies it as such and if it is within this \
        distance to the lobe surface model.', default=None)    

    op = parser.parse_args()

    tmp_dir = tempfile.mkdtemp()
    wf = FissurePhenotypesWorkflow(op.in_lobes, op.in_fiss, op.out_csv,
      op.cid, op.dist_thresh, op.ray_tol, tmp_dir)
    wf.run()
    shutil.rmtree(tmp_dir)
