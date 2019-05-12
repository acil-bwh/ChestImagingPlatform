from nipype.interfaces.base import CommandLine, CommandLineInputSpec, \
     SEMLikeCommandLine, TraitedSpec, File, Directory, traits, isdefined, \
     InputMultiPath, OutputMultiPath
import os

class unu_heqInputSpec(CommandLineInputSpec):
    input = File(desc="Input nrrd", exists=True, argstr="-i %s")
    amount = traits.Float(desc="Extent to which the histogram equalizing \
    mapping should be applied; 0.0: no change, 1.0: full equalization (float)",
    argstr="--amount %f")
    smart = traits.Int(desc="bins in value histogram to ignore in calculating \
    the mapping. Bins are ignored when they get more hits than other bins, and \
    when the values that fall in them are constant. This is an effective way \
    to prevent large regions of background value from distorting the \
    equalization mapping. (optional int); default: 0", argstr="--smart %d")
    bin = traits.Int(desc="bins to use in histogram that is created in order \
    to calculate the mapping that achieves the equalization.", 
    argstr="--bin %d")
    output = traits.Either(traits.Bool, File(), hash_files=False, 
                           desc="Output nrrd (string)", argstr="--output %s")
    
class unu_heqOutputSpec(TraitedSpec):
    output = File(desc="Output nrrd (string)", exists=True)
    map = File(desc="The value mapping used to achieve histogram equalization \
    is represented by a univariate regular map. By giving a filename here, \
    that map can be saved out and applied to other nrrds with unu rmap",
    exists=True)
    
class unu_heq(CommandLine):
    _cmd = 'unu heq '
    input_spec = unu_heqInputSpec
    output_spec = unu_heqOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output'] = os.path.abspath(self.inputs.output)
        return outputs    
    
class unu_2opInputSpec(CommandLineInputSpec):
    operator = traits.Str(desc="Binary operator", position = 0, argstr="%s")
    in1_file = File(desc="First input. Must be a nrrd.", position = 1, 
                    exists=True, argstr="%s")
    in2_file = File(desc="Second input. Must be a nrrd.", position = 2, 
                    exists=True, argstr="%s")
    in1_scalar = traits.Float(desc="First input. Must be a single value.", 
                              position = 1, argstr="%f")
    in2_scalar = traits.Float(desc="Second input. Must be a single value.", 
                              position = 2, argstr="%f")    
    seed = traits.Int(desc="seed value for RNG for nrand, so that you can get \
    repeatable results between runs, or, by not using this option, the RNG \
    seeding will be based on the current time", argstr="--seed %d")
    type = traits.Str(desc="type to convert all INPUT nrrds to, prior to doing \
    operation, useful for doing, for instance, the difference between two \
    unsigned char nrrds. This will also determine output type. By default (not \
    using this option), the types of the input nrrds are left unchanged.", 
    argstr="--type %s")
    which = traits.Int(desc="Which argument (0 or 1) should be used to \
    determine the shape of the output nrrd. By default (not using this \
    option), the first non-constant argument is used. (int)", 
    argstr="--which %d")
    output = traits.Either(traits.Bool, File(), hash_files=False, 
                           desc="Output nrrd (string)", argstr="--output %s")
    
class unu_2opOutputSpec(TraitedSpec):
    output = File(desc="Output nrrd (string)", exists=True)
    
class unu_2op(CommandLine):
    _cmd = 'unu 2op '
    input_spec = unu_2opInputSpec
    output_spec = unu_2opOutputSpec    

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output'] = os.path.abspath(self.inputs.output)
        return outputs

class unu_convertInputSpec(CommandLineInputSpec):
    type = traits.Str(desc="type to convert to", argstr="--type %s")
    input = File(desc="input nrrd", argstr="--input %s")
    output = traits.Either(traits.Bool, File(), hash_files=False, 
                           desc="Output nrrd (string)", argstr="--output %s")
    
class unu_convertOutputSpec(TraitedSpec):
    output = File(desc="Output nrrd (string)", exists=True)
    
class unu_convert(CommandLine):
    _cmd = 'unu convert '
    input_spec = unu_convertInputSpec
    output_spec = unu_convertOutputSpec    

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output'] = os.path.abspath(self.inputs.output)
        return outputs    

class unu_resampleInputSpec(CommandLineInputSpec):
    input = File(desc="input nrrd", argstr="--input %s")
    size = traits.Str(desc="For each axis, info about how many samples in output", argstr="--size %s")
    kernel = traits.Str(desc="Kernel to use for resampling", argstr="--kernel %s")    
    output = traits.Either(traits.Bool, File(), hash_files=False, 
                           desc="Output nrrd (string)", argstr="--output %s")

class unu_resampleOutputSpec(CommandLineInputSpec):
    output = File(desc="Output nrrd (string)", exists=True)
    
class unu_resample(CommandLine):
    _cmd = 'unu resample '
    input_spec = unu_resampleInputSpec
    output_spec = unu_resampleOutputSpec    

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output'] = os.path.abspath(self.inputs.output)
        return outputs    
