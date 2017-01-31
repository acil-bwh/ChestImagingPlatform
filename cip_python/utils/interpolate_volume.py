
import os
import numpy as np
import SimpleITK as sitk
from optparse import OptionParser


class InterpolateVolume():

    def __init__(self,delta):
    
        #Define some params
        self.save_deformations=False
        self.deformation_folder=None
        self.new_delta = delta
        self.rescale = False


    def demons_registration(self,F, M, nit=100, std=1.5, fname=None, wfield=False,
                        verbose=False):

        run_demons = True

        if fname is not None:

            df01_fname = fname + '01.nrrd'
            df10_fname = fname + '10.nrrd'

            run_demons = not(os.path.isfile(df01_fname) | os.path.isfile(df10_fname))

        if run_demons:
            print('Running registration')

            dif_demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
            dif_demons.SetNumberOfIterations(nit)
            dif_demons.SetStandardDeviations(std)

            #Rescale Images to 0-1
            if self.rescale == True:
                rescaler = sitk.RescaleIntensityImageFilter()
                rescaler.SetOutputMaximum(1)
                rescaler.SetOutputMinimum(0)
                F=rescaler.Execute(sitk.Cast(F,sitk.sitkFloat32))
                M=rescaler.Execute(sitk.Cast(M,sitk.sitkFloat32))
            
            F.SetOrigin(M.GetOrigin())
            df01 = dif_demons.Execute(F, M)
            df10 = dif_demons.Execute(M, F)
        else:
            print('Reading deformation fields')

            df01 = sitk.ReadImage(df01_fname)
            df10 = sitk.ReadImage(df10_fname)

        if wfield:
            if df01_fname is None:
                df01_fname = '01.mhd'
                df10_fname = '10.mhd'

            w = sitk.ImageFileWriter()
            w.Execute(df01, df01_fname, False)
            w.Execute(df10, df10_fname, False)

        return df01, df10

    def execute(self,imgOriginal):

        dim = imgOriginal.GetSize()

        dfield01 = []
        dfield10 = []

        for i in range(dim[2]-1):
            F = imgOriginal[:, :, i]
            M = imgOriginal[:, :, i+1]

            print('Processing registration ({})'.format(i))
            
            # Lectura de deformaciones en sentido directo
            if self.save_deformations:
                fname = os.path.join(self.deformation_folder,'{}_'.format(i))
            else:
                fname = None
            df01, df10 = self.demons_registration(F, M, fname=fname)
            dfield01.append(df01)
            dfield10.append(df10)

        wraper = sitk.WarpImageFilter()  # Interpolador para deform

        # Definicion de los intervalos donde itnerpolaremos
        dim_pixel = imgOriginal.GetSpacing()
        deltax = float(dim_pixel[0])
        deltay = float(dim_pixel[1])
        delta=self.new_delta

        delta_z = dim_pixel[2]
        origin = 0
        destination = imgOriginal.GetDepth()-1
        din_range = destination - origin
        step = np.arange(origin+delta/delta_z, destination, delta/delta_z)


        output_type = sitk.GetArrayFromImage(imgOriginal).dtype
        vol_output = np.zeros((len(step)+1, dim[1], dim[0]), output_type)
        O3 = np.zeros((len(step)+1, 2), np.float)

        for z_index in range(0, len(step)):
            print("z_index = " + str(z_index))
            
            index_0 = int(np.floor(step[z_index]))
            index_1 = int(np.floor(step[z_index])+1)
            tau = step[z_index] - index_0
            tau = 1-tau
            
            # print('i_0: {}\ti_1: {}'.format(index_0, index_1))
            
            F = imgOriginal[:, :, index_0]
            M = imgOriginal[:, :, index_1]
            
            
            df01 = dfield01[index_0]
            df10 = dfield10[index_0]
            
            disp1 = sitk.GetImageFromArray(sitk.GetArrayFromImage(df01)*tau,
                                           isVector=True)
            disp1.SetSpacing(df01.GetSpacing())
            disp1.SetOrigin(df01.GetOrigin())
            disp1.SetDirection(df01.GetDirection())
                                   
            disp2 = sitk.GetImageFromArray(sitk.GetArrayFromImage(df10)*(1-tau),
                                                                  isVector=True)
            disp2.SetSpacing(df10.GetSpacing())
            disp2.SetOrigin(df10.GetOrigin())
            disp2.SetDirection(df10.GetDirection())

            wraper.SetOutputParameteresFromImage(M)  # Forzar salida del tipo de la entrada
            F2 = wraper.Execute(M, disp1)

            wraper.SetOutputParameteresFromImage(F)  # Forzar salida del tipo de la entrada
            M2 = wraper.Execute(F, disp2)

            output = (tau)*sitk.GetArrayFromImage(M2) + (1-tau)*sitk.GetArrayFromImage(F2)
            
            if z_index == 0:
               vol_output[0, :, :] = sitk.GetArrayFromImage(F)

            vol_output[z_index+1, :, :] = output


            O1 = np.array(F.GetOrigin())
            O2 = np.array(M.GetOrigin())
            O3[z_index, :] = (O2-O1)*tau

        interp_image = sitk.GetImageFromArray(vol_output)
        interp_image.SetSpacing((deltax, deltay, delta))
        interp_image.SetDirection(imgOriginal.GetDirection())
        interp_image.SetOrigin(imgOriginal.GetOrigin())
        
        return interp_image


if __name__ == '__main__':

#Read/Write Images

    desc = """ Interpolate data along z asis using pixel transfer via Demons registration."""
    parser = OptionParser(description=desc)

    parser.add_option('-i',help='Input image file',dest='in_im',default=None)
    parser.add_option('-o',help='Output image file',dest='out_im',default=None)
    parser.add_option('-s',help='Output spacing',dest='out_spacing',default=1)
    (options, args) = parser.parse_args()
    
    #orig_image = sitk.Cast(sitk.ReadImage(options.in_im), sitk.sitkInt16)
    orig_image = sitk.ReadImage(options.in_im)

    interp = InterpolateVolume(np.float(options.out_spacing))

    out_image=interp.execute(orig_image)

    sitk.WriteImage(out_image, options.out_im)

