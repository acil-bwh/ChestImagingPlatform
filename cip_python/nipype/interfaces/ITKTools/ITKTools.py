from nipype.interfaces.base import CommandLine, CommandLineInputSpec, \
     SEMLikeCommandLine, TraitedSpec, File, Directory, traits, isdefined, \
     InputMultiPath, OutputMultiPath
import os

class pxdistancetransformInputSpec(CommandLineInputSpec):
    i = File(desc="the input image (a binary mask threshold at 0 is performed \
    if the image is not binary)", exists=True, argstr="-in %s")
    s = traits.Bool(desc="flag: if set, output squared distances instead of \
    distances", argstr="-s ")
    m = traits.Enum("Maurer", "Danielsson", "Morphological", \
                    "MorphologicalSigned", desc="method", argstr="-m %s")
    out = traits.Either(traits.Bool, File(), hash_files=False, 
                        desc="the output of distance transform", 
                        argstr="-out %s")
                    
class pxdistancetransformOutputSpec(TraitedSpec):
    out = File(desc="the output of distance transform", exists=True)
    
class pxdistancetransform(CommandLine):    
    input_spec = pxdistancetransformInputSpec
    output_spec = pxdistancetransformOutputSpec
    _cmd = " pxdistancetransform "
    _outputs_filenames = {'out':'out.nii'}

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out'] = os.path.abspath(self.inputs.out)
        return outputs
    
