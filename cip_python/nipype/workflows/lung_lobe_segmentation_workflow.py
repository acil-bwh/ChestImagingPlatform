from argparse import ArgumentParser
import cip_python.nipype.interfaces.cip as cip

import cip_python.nipype.interfaces.unu as unu
import cip_python.nipype.interfaces.cip.cip_python_interfaces as cip_python_interfaces
import nipype.pipeline.engine as pe         # the workflow and node wrappers
from nipype.pipeline.engine import Workflow
import tempfile, shutil, sys, os, pickle, gzip


class LungLobeSegmentationWorkflow(Workflow):
    """This workflow generates a lung lobe segmentation given an input CT image.

    Parameters
    ----------
    ct_file_name : str
        The file name of the CT image (single file, 3D volume) for which to 
        generate a lung lobe segmentation.    

    lobe_seg_file_name : str
        The file name of the output lung lobe label map.
        
    tmp_dir : str
        Directory in which to store intermediate files used for this computation

    reg : float, optional
        FitLobeSurfaceModelsToParticleData CLI parameter. The higher this value, 
        the more departures from the mean shape are penalized.

    ilap : string, optional
        FitLobeSurfaceModelsToParticleData CLI parameter. Left lung airway 
        particles file name. If specified, the airway particles will 
        contribute to the left lobe boundary fitting metric
    
    irap : string, optional
        FitLobeSurfaceModelsToParticleData CLI parameter. Right lung airway 
        particles file name. If specified, the airway particles will contribute 
        to the right lobe boundary fitting metric.
    
    ilvp : string, optional
        FitLobeSurfaceModelsToParticleData CLI parameter. Left lung vessel 
        particles file name. If specified, the vessel particles will contribute 
        to the left lobe boundary fitting metric.
    
    irvp : string, optional
        FitLobeSurfaceModelsToParticleData CLI parameter. Right lung vessel 
        particles file name. If specified, the vessel particles will contribute 
        to the right lobe boundary fitting metric.
    
    ilm : string, optional
        Input lung label map that will be used for various operations within the 
        workflow. The left and right lungs must be uniquely labeled. If this lung 
        label map is not specified, it will be generated by this workflow.
        
    ifp : string, optional
        Input fissure particles file name (left and right lungs combined). If 
        none specified, the fissure particles will be generated by this 
        workflow.    
    pre_dist : float, optional
        FilterFissureParticlesData CLI parameter. Maximum inter-particle 
        distance. Two particles must be at least this close together to be 
        considered for connectivity during particle pre-filtering stage.
    post_dist : float, optional
        FilterFissureParticlesData CLI parameter. Maximum inter-particle 
        distance. Two particles must be at least this close together to be 
        considered for connectivity during particle post-filtering stage.        
    pre_size : int, optional
        FilterFissureParticlesData CLI parameter. Component size cardinality 
        threshold for fissure particle pre-filtering. Only components with this 
        many particles or more will be retained in the output.
    dist_thersh : double, optional
        ClassifyFissureParticles CLI parameter. A particle will be classified
        as fissure if the Fischer linear discrimant classifier classifies it as
        such and if it is within this distance to the lobe surface model
    post_size : int, optional
        FilterFissureParticlesData CLI parameter. Component size cardinality 
        threshold for fissure particle post-filtering. Only components with this
        many particles or more will be retained in the output
    iters : int, optional
      fissure_particles parameter. Number of iterations for which to run the 
      fissure particles algorithm. Increasing this value can improve results but 
      will result in longer execution time.
    scale : float, optional
      fissure_particles parameter. The scale of the fissure to consider in 
      scale space.
    lth : float, optional
      fissure_particles parameter. Threshold to use when pruning particles 
      during population control. Must be less than zero.
    sth : float, optional
      fissure_particles parameter. Threshold to use when initializing particles. 
      Must be less than zero.        
    perm : bool, optional
      fissure_particles parameter. Allow mask and CT volumes to have different 
      shapes or meta data      .
    """
    def __init__(self, ct_file_name, lobe_seg_file_name, tmp_dir, reg=50,
        ilap=None, irap=None, ilvp=None, irvp=None, ilm=None, ifp=None, 
        pre_dist=3.0, post_dist=3.0, pre_size=110, dist_thresh=1000, 
        post_size=110, iters=200, scale=0.9, lth=-15, sth=-45, 
        perm=False, cid='cid'):

        Workflow.__init__(self, 'LungLobeSegmentationWorkflow')

        ct_file_name = os.path.realpath(ct_file_name)
        lobe_seg_file_name = os.path.realpath(lobe_seg_file_name)
        if ilm:
          ilm = os.path.realpath(ilm)

        file_and_path = os.path.abspath(__file__)
        file_path = os.path.dirname(file_and_path)
        
        self._resourcesDir = os.path.join(file_path, '..', '..', '..', 
            'Resources', 'LungLobeAtlasData', '')        
        self._referenceLabelMap = os.path.join(self._resourcesDir, 
            '10002K_INSP_STD_BWH_COPD_leftLungRightLung.nrrd')

        self._partialLungLabelMap = os.path.join(tmp_dir, cid +
            '_partialLungLabelMap.nrrd')
        self._rightLungLobesShapeModel = os.path.join(tmp_dir, cid + 
            '_rightLungLobesShapeModel.csv')
        self._leftLungLobesShapeModel = os.path.join(tmp_dir, cid + 
            '_leftLungLobesShapeModel.csv')
        self._preFilteredFissureParticles = os.path.join(tmp_dir, cid + 
            '_preFilteredFissureParticles.vtk')        
        self._rightFissureParticlesPreFiltered = os.path.join(tmp_dir, cid +
            '_rightPreFilteredFissureParticles.vtk')
        self._leftFissureParticlesPreFiltered = os.path.join(tmp_dir, cid + 
            '_leftPreFilteredFissureParticles.vtk')
        self._fissureParticles = os.path.join(tmp_dir, cid + 
            '_fissureParticles.vtk')
        self._roClassifiedFissureParticles = os.path.join(tmp_dir, cid + 
            '_roClassifiedFissureParticles.vtk')
        self._rhClassifiedFissureParticles = os.path.join(tmp_dir, cid + 
            '_rhClassifiedFissureParticles.vtk')
        self._loClassifiedFissureParticles = os.path.join(tmp_dir, cid + 
            '_loClassifiedFissureParticles.vtk')
        self._loPostFilteredFissureParticles = os.path.join(tmp_dir, cid + 
            '_loPostFilteredFissureParticles.vtk')
        self._rhPostFilteredFissureParticles = os.path.join(tmp_dir, cid + 
            '_rhPostFiltereFissureParticles.vtk')
        self._roPostFilteredFissureParticles = os.path.join(tmp_dir, cid + 
            '_roPostFiltereFissureParticles.vtk')
        
        #----------------------------------------------------------------------
        # Create node for segmenting lungs
        #----------------------------------------------------------------------
        generate_partial_lung_label_map = \
          pe.Node(interface=cip.GeneratePartialLungLabelMap(), 
                  name='generate_partial_lung_label_map')
        generate_partial_lung_label_map.inputs.ict = ct_file_name
        generate_partial_lung_label_map.inputs.olm = self._partialLungLabelMap

        #----------------------------------------------------------------------
        # Create node for generate lobe surface models
        #----------------------------------------------------------------------
        generate_lobe_surface_models = \
          pe.Node(interface=cip.GenerateLobeSurfaceModels(), 
                  name='generate_lobe_surface_models')
        generate_lobe_surface_models.inputs.irlm = self._referenceLabelMap        
        generate_lobe_surface_models.inputs.dir = self._resourcesDir
        generate_lobe_surface_models.inputs.orsm = \
          self._rightLungLobesShapeModel
        generate_lobe_surface_models.inputs.olsm = self._leftLungLobesShapeModel
        if ilm is not None:
            generate_lobe_surface_models.inputs.ilm = ilm
        
        #----------------------------------------------------------------------        
        # Create node for running fissure particles
        #----------------------------------------------------------------------        
        fissure_particles_node = \
          pe.Node(interface=cip_python_interfaces.fissure_particles(), 
                  name='fissure_particles_node')
        fissure_particles_node.inputs.ict = ct_file_name
        fissure_particles_node.inputs.scale = 0.9
        fissure_particles_node.inputs.lth = -15
        fissure_particles_node.inputs.sth = -45
        fissure_particles_node.inputs.rate = 1.0
        fissure_particles_node.inputs.min_int = -920 
        fissure_particles_node.inputs.max_int =-400
        fissure_particles_node.inputs.iters = 200
        if perm:
            fissure_particles_node.inputs.perm
        if ilm is not None:
            fissure_particles_node.inputs.ilm = ilm        
            
        #----------------------------------------------------------------------
        # Create node for pre-filtering fissure particles
        #----------------------------------------------------------------------
        pre_filter_fissure_particle_data = \
          pe.Node(interface=cip.FilterFissureParticleData(), 
                  name='pre_filter_fissure_particle_data')
        if ifp is not None:        
            pre_filter_fissure_particle_data.inputs.ifp = ifp
        pre_filter_fissure_particle_data.inputs.size = pre_size
        pre_filter_fissure_particle_data.inputs.dist = pre_dist        
        pre_filter_fissure_particle_data.inputs.ofp = \
          self._preFilteredFissureParticles        

        #----------------------------------------------------------------------
        # Create nodes to isolate left particles
        #----------------------------------------------------------------------
        extract_left_particles = \
          pe.Node(interface=cip.ExtractParticlesFromChestRegionChestType(), 
                  name='extract_left_particles')
        extract_left_particles.inputs.cipr = 'LeftLung'
        extract_left_particles.inputs.op = self._leftFissureParticlesPreFiltered
        if ilm is not None:
            extract_left_particles.inputs.ilm = ilm
            
        #----------------------------------------------------------------------
        # Create nodes to isolate right particles
        #----------------------------------------------------------------------
        extract_right_particles = \
          pe.Node(interface=cip.ExtractParticlesFromChestRegionChestType(), 
                  name='extract_right_particles')        
        extract_right_particles.inputs.cipr = 'RightLung'
        extract_right_particles.inputs.op = \
          self._rightFissureParticlesPreFiltered
        if ilm is not None:
            extract_right_particles.inputs.ilm = ilm
        
        #----------------------------------------------------------------------
        # Create node for fitting lobe boundare shape models
        #----------------------------------------------------------------------
        fit_lobe_surface_models_to_particle_data = \
          pe.Node(interface=cip.FitLobeSurfaceModelsToParticleData(), 
                  name='fit_lobe_surface_models_to_particle_data')
        fit_lobe_surface_models_to_particle_data.inputs.iters = 200
        fit_lobe_surface_models_to_particle_data.inputs.olsm = \
          self._leftLungLobesShapeModel
        fit_lobe_surface_models_to_particle_data.inputs.orsm = \
          self._rightLungLobesShapeModel
        fit_lobe_surface_models_to_particle_data.inputs.reg = reg
        if ilap is not None:
            fit_lobe_surface_models_to_particle_data.inputs.ilap = ilap
        if irap is not None:
            fit_lobe_surface_models_to_particle_data.inputs.irap = irap
        if ilvp is not None:
            fit_lobe_surface_models_to_particle_data.inputs.ilvp = ilvp
        if irvp is not None:
            fit_lobe_surface_models_to_particle_data.inputs.irvp = irvp
        
        #----------------------------------------------------------------------
        # Create node for right fissure particle classification
        #----------------------------------------------------------------------
        classify_right_fissure_particles = \
          pe.Node(interface=cip.ClassifyFissureParticles(), 
                  name='classify_right_fissure_particles')        
        classify_right_fissure_particles.inputs.oro = \
          self._roClassifiedFissureParticles
        classify_right_fissure_particles.inputs.orh = \
          self._rhClassifiedFissureParticles
        classify_right_fissure_particles.inputs.dist_thresh = dist_thresh
        
        #----------------------------------------------------------------------
        # Create node for left fissure particle classification
        #----------------------------------------------------------------------
        classify_left_fissure_particles = \
          pe.Node(interface=cip.ClassifyFissureParticles(), 
                  name='classify_left_fissure_particles')        
        classify_left_fissure_particles.inputs.olo = \
          self._loClassifiedFissureParticles
        classify_left_fissure_particles.inputs.dist_thresh = dist_thresh
        
        #----------------------------------------------------------------------
        # Create node for post-filtering left oblique fissure particles
        #----------------------------------------------------------------------
        post_filter_lo_fissure_particle_data = \
          pe.Node(interface=cip.FilterFissureParticleData(), 
                  name='post_filter_lo_fissure_particle_data')
        post_filter_lo_fissure_particle_data.inputs.size = post_size
        post_filter_lo_fissure_particle_data.inputs.dist = post_dist        
        post_filter_lo_fissure_particle_data.inputs.ofp = \
          self._loPostFilteredFissureParticles
        
        #----------------------------------------------------------------------
        # Create node for post-filtering right oblique fissure particles
        #----------------------------------------------------------------------
        post_filter_ro_fissure_particle_data = \
          pe.Node(interface=cip.FilterFissureParticleData(), 
                  name='post_filter_ro_fissure_particle_data')
        post_filter_ro_fissure_particle_data.inputs.size = post_size
        post_filter_ro_fissure_particle_data.inputs.dist = post_dist        
        post_filter_ro_fissure_particle_data.inputs.ofp = \
          self._roPostFilteredFissureParticles

        #----------------------------------------------------------------------
        # Create node for post-filtering right horizontal fissure particles
        #----------------------------------------------------------------------
        post_filter_rh_fissure_particle_data = \
          pe.Node(interface=cip.FilterFissureParticleData(), 
                  name='post_filter_rh_fissure_particle_data')
        post_filter_rh_fissure_particle_data.inputs.size = 10
        post_filter_rh_fissure_particle_data.inputs.ofp = \
          self._rhPostFilteredFissureParticles
        
        #----------------------------------------------------------------------
        # Create node for segmenting lung lobes
        #----------------------------------------------------------------------
        segment_lung_lobes = \
          pe.Node(interface=cip.SegmentLungLobes(), name='segment_lung_lobes')
        if ilm is not None:
            segment_lung_lobes.inputs.ilm = ilm
        segment_lung_lobes.inputs.olm = lobe_seg_file_name

        #----------------------------------------------------------------------
        # Connect the nodes
        #----------------------------------------------------------------------
        if ifp is None:
            self.connect(fissure_particles_node, 'op', 
                         pre_filter_fissure_particle_data, 'ifp')
        self.connect(pre_filter_fissure_particle_data, 'ofp',
                     extract_right_particles, 'ip')
        self.connect(pre_filter_fissure_particle_data, 'ofp',
                     extract_left_particles, 'ip')        
        self.connect(extract_right_particles, 'op',
                     fit_lobe_surface_models_to_particle_data, 'irfp')
        self.connect(extract_left_particles, 'op',
                     fit_lobe_surface_models_to_particle_data, 'ilfp')
        if ilm is None:
            self.connect(generate_partial_lung_label_map, 'olm',
                         extract_right_particles, 'ilm')            
            self.connect(generate_partial_lung_label_map, 'olm',
                         extract_left_particles, 'ilm')                    
            self.connect(generate_partial_lung_label_map, 'olm',
                         generate_lobe_surface_models, 'ilm')
            self.connect(generate_partial_lung_label_map, 'olm',
                         fissure_particles_node, 'ilm')
            self.connect(generate_partial_lung_label_map, 'olm',
                         segment_lung_lobes, 'ilm')
        self.connect(fit_lobe_surface_models_to_particle_data, 'olsm',
                     segment_lung_lobes, 'ilsm')        
        self.connect(fit_lobe_surface_models_to_particle_data, 'orsm',
                     segment_lung_lobes, 'irsm')
        self.connect(extract_right_particles, 'op',
                     classify_right_fissure_particles, 'ifp')                
        self.connect(fit_lobe_surface_models_to_particle_data, 'orsm',
                     classify_right_fissure_particles, 'irsm')        
        self.connect(extract_left_particles, 'op',
                     classify_left_fissure_particles, 'ifp')                
        self.connect(fit_lobe_surface_models_to_particle_data, 'olsm',
                     classify_left_fissure_particles, 'ilsm')
        self.connect(classify_left_fissure_particles, 'olo',
                     post_filter_lo_fissure_particle_data, 'ifp')
        self.connect(classify_right_fissure_particles, 'orh',
                     post_filter_rh_fissure_particle_data, 'ifp')        
        self.connect(classify_right_fissure_particles, 'oro',
                     post_filter_ro_fissure_particle_data, 'ifp')        
        self.connect(post_filter_lo_fissure_particle_data, 'ofp',
                     segment_lung_lobes, 'lofp')
        self.connect(post_filter_ro_fissure_particle_data, 'ofp',
                     segment_lung_lobes, 'rofp')
        #self.connect(post_filter_rh_fissure_particle_data, 'ofp',
        #             segment_lung_lobes, 'rhfp')                
        self.connect(generate_lobe_surface_models, "olsm",
                     fit_lobe_surface_models_to_particle_data, "ilsm")
        self.connect(generate_lobe_surface_models, "orsm",
                     fit_lobe_surface_models_to_particle_data, "irsm")

if __name__ == "__main__":
    desc = """This workflow produces a lung lobe segmentation given an input
    CT image."""

    parser = ArgumentParser(description=desc)
    parser.add_argument('--in_ct', help='The file name of the CT image (single \
      file, 3D volume) for which to generate a lung lobe segmentation. Must \
      be in nrrd format', dest='in_ct', metavar='<string>', required=True)
    parser.add_argument('--out', 
      help='The file name of the output lung lobe label map.', 
      dest='out', metavar='<string>', required=True)
    parser.add_argument('--reg', 
      help='FitLobeSurfaceModelsToParticleData CLI parameter. The higher this \
      value, the more departures from the mean shape are penalized. (Optional)', 
      dest='reg', metavar='<float>', default=50)   
    parser.add_argument('--ilap', 
      help='FitLobeSurfaceModelsToParticleData CLI parameter. Left lung airway \
      particles file name. If specified, the airway particles will contribute \
      to the left lobe boundary fitting metric. (Optional)', 
      dest='ilap', metavar='<string>', default=None)
    parser.add_argument('--irap', 
      help='FitLobeSurfaceModelsToParticleData CLI parameter. Right lung \
      airway particles file name. If specified, the airway particles will \
      contribute to the right lobe boundary fitting metric. (Optional)', 
      dest='irap', metavar='<string>', default=None)     
    parser.add_argument('--ilvp', 
      help='FitLobeSurfaceModelsToParticleData CLI parameter. Left lung \
      vessel particles file name. If specified, the vessel particles will \
      contribute to the left lobe boundary fitting metric. (Optional)', 
      dest='ilvp', metavar='<string>', default=None)
    parser.add_argument('--irvp', 
      help='FitLobeSurfaceModelsToParticleData CLI parameter. Right lung \
      vessel particles file name. If specified, the vessel particles will \
      contribute to the right lobe boundary fitting metric. (Optional)', 
      dest='irvp', metavar='<string>', default=None)     
    parser.add_argument('--ilm', 
      help='Input lung label map that will be used for various operations \
      within the workflow. The left and right lungs must be uniquely \
      labeled. If this lung label map is not specified, it will be generated \
      by this workflow. (Optional)', 
      dest='ilm', metavar='<string>', default=None)
    parser.add_argument('--ifp', 
      help='Input fissure particles file name (left and right lungs combined). \
      If none specified, the fissure particles will be generated by this \
      workflow. (Optional)', 
      dest='ifp', metavar='<string>', default=None)    
    parser.add_argument('--predist', 
      help='FilterFissureParticlesData CLI parameter. Maximum inter-particle \
      distance. Two particles must be at least this close together to be \
      considered for connectivity during particle pre-filtering stage. \
      (Optional)',
      dest='pre_dist', metavar='<float>', default=3.0)
    parser.add_argument('--postdist', 
      help='FilterFissureParticlesData CLI parameter. Maximum inter-particle \
      distance. Two particles must be at least this close together to be \
      considered for connectivity during particle post-filtering stage. \
      (Optional)',
      dest='post_dist', metavar='<float>', default=3.0)
    parser.add_argument('--presize', 
      help='FilterFissureParticlesData CLI parameter. Component size \
      cardinality threshold for fissure particle pre-filtering. Only \
      components with this many particles or more will be retained in the \
      output. (Optional)',
      dest='pre_size', metavar='<int>', default=110)
    parser.add_argument('--dist_thresh', 
      help='ClassifyFissureParticles CLI parameter. A particle will be classified \
      as fissure if the Fischer linear discrimant classifier classifies it as \
      such and if it is within this distance to the lobe surface model. (Optional)',
      dest='dist_thresh', metavar='<float>', default=1000)
    parser.add_argument('--postsize', 
      help='FilterFissureParticlesData CLI parameter. Component size \
      cardinality threshold for fissure particle post-filtering. Only \
      components with this many particles or more will be retained in the \
      output. (Optional)',
      dest='post_size', metavar='<int>', default=110)
    parser.add_argument('--iters', 
      help='fissure_particles parameter. Number of iterations for which to run \
      the fissure particles algorithm. Increasing this value can improve \
      results but will result in longer execution time. (Optional)',
      dest='iters', metavar='<int>', default=100)
    parser.add_argument('--scale', 
      help='fissure_particles parameter. The scale of the fissure to consider \
      in scale space. (Optional)', dest='scale', metavar='<float>', default=0.9)
    parser.add_argument('--lth', 
      help='fissure_particles parameter. Threshold to use when pruning \
      particles during population control. Must be less than zero. (Optional)',
      dest='lth', metavar='<float>', default=-15)
    parser.add_argument('--sth', 
      help='fissure_particles parameter. Threshold to use when initializing \
      particles. Must be less than zero. (Optional)',
      dest='sth', metavar='<float>', default=-45)
    parser.add_argument("--perm", 
      help='fissure_particles parameter. Allow mask and CT volumes to have \
      different shapes or meta data', dest="perm", action='store_true')
    parser.add_argument("--cid",
      help='Case ID string to name output files before suffix',
      dest='cid', metavar='<string>', default='cid')
    
    op = parser.parse_args()

    tmp_dir = tempfile.mkdtemp()
    wf = LungLobeSegmentationWorkflow(op.in_ct, op.out, tmp_dir, 
        reg=float(op.reg), ilap=op.ilap, irap=op.irap, ilvp=op.ilvp, 
        irvp=op.irvp, ilm=op.ilm, ifp=op.ifp, pre_dist=float(op.pre_dist), 
        post_dist=float(op.post_dist), pre_size=int(op.pre_size),
        dist_thresh=float(op.dist_thresh), post_size=int(op.post_size),
        iters=int(op.iters), scale=float(op.scale), lth=float(op.lth),
        sth=float(op.sth), perm=op.perm, cid=op.cid)

    wf.run()
    shutil.rmtree(tmp_dir)
