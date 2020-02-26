import vtk, math
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from numpy import linalg as LA
import pandas as pd
from optparse import OptionParser
from scipy.stats import kde
import os.path

import matplotlib.pyplot as plt
from cip_python.common import ChestConventions

class Bronchiectasis:

    def __init__(self, vascular_file, airway_file, output_prefix, plot=True):
        self.vascular_file = vascular_file
        self.airway_file = airway_file
        self.output_prefix = output_prefix
        self.plot = plot
        self.distance_th = 30.0
        self.angle1_th = math.pi/4
        self.angle2_th = math.pi/3
        self.scale_ratio_th = 3.0
        self.distance_decay = 1.0
        self.airway_scale_th = 1.0
        self.vessel_scale_th = 0.6
        self.airway_dnn_th = 0.5
        self.vessel_dnn_th = 0.5
        self.sigma = 0.18

        self.dnn_ = False
        self.artery = False
        self.add_ratio_to_particles = False

    @staticmethod
    def get_artery_indices(vessel_pp):
        tmp = vessel_pp.GetFieldData().GetArray('ChestRegionChestType')
        if not isinstance(tmp, vtk.vtkDataArray):
            tmp = vessel_pp.GetPointData().GetArray('ChestRegionChestType')

        array_v = vtk_to_numpy(tmp).astype(np.uint16)
        type_values = ChestConventions.GetChestTypeFromValue(array_v)
        artery_type = ChestConventions.GetChestTypeValueFromName('Artery')
        return np.argwhere(type_values == artery_type)[:, 0]

    def test_condition(self, distance, kk2, angle1, angle2, a_s, v_s, a_r, v_r):
        if not self.dnn_:
            return (distance < self.distance_th - self.distance_decay * kk2) & (angle1 < self.angle1_th) & (
                angle2 > self.angle2_th) & (a_s / v_s < self.scale_ratio_th) & (v_s / a_s < self.scale_ratio_th) & (
                       a_r > self.airway_scale_th) & (v_r > self.vessel_scale_th)
        else:
            return (distance < self.distance_th - self.distance_decay * kk2) & (angle1 < self.angle1_th) & (
                angle2 > self.angle2_th) & (a_r > self.airway_dnn_th) & (v_r > self.vessel_dnn_th)

    def extract_artery_polydata(self, vessel_pp):
        artery_indices = self.get_artery_indices(vessel_pp)
        artery_polydata = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(artery_indices.shape[0])

        for ii, vv in enumerate(artery_indices):
            vtk_points.SetPoint(ii, vessel_pp.GetPoint(vv))

        artery_polydata.SetPoints(vtk_points)

        for ii in range(vessel_pp.GetPointData().GetNumberOfArrays()):
            array_name = vessel_pp.GetPointData().GetArrayName(ii)
            vessel_array_vtk = vessel_pp.GetPointData().GetArray(array_name)
            vessel_array_np = vtk_to_numpy(vessel_array_vtk)

            vtk_array = vessel_array_vtk.NewInstance()
            vtk_array.DeepCopy(vessel_array_vtk)
            vtk_array.SetName(array_name)
            vtk_array.SetNumberOfValues(artery_indices.shape[0])
            vtk_array.SetNumberOfTuples(artery_indices.shape[0])
            vtk_array.SetNumberOfComponents(vessel_array_vtk.GetNumberOfComponents())

            for jj, vv in enumerate(artery_indices):
                if np.shape([vessel_array_np[vv]])[-1] > 1:
                    vtk_array.SetTuple(jj, vessel_array_np[vv])
                else:
                    vtk_array.SetValue(jj, vessel_array_np[vv])

            artery_polydata.GetPointData().AddArray(vtk_array)
        for ii in range(vessel_pp.GetFieldData().GetNumberOfArrays()):
            array_name = vessel_pp.GetFieldData().GetArrayName(ii)
            if array_name == 'AV_classification_dnn_model_name' or array_name == 'vessel_sizing_dnn_model_name':
                vessel_array_vtk = vessel_pp.GetFieldData().GetAbstractArray(array_name)
                artery_polydata.GetFieldData().AddArray(vessel_array_vtk)
            else:
                vessel_array_vtk = vessel_pp.GetFieldData().GetArray(array_name)
                vessel_array_np = vtk_to_numpy(vessel_array_vtk)

                if vessel_array_np.shape[0] == vessel_pp.GetNumberOfPoints():
                    vtk_array = vessel_array_vtk.NewInstance()
                    vtk_array.DeepCopy(vessel_array_vtk)
                    vtk_array.SetName(array_name)
                    vtk_array.SetNumberOfValues(artery_indices.shape[0])
                    vtk_array.SetNumberOfTuples(artery_indices.shape[0])
                    vtk_array.SetNumberOfComponents(vessel_array_vtk.GetNumberOfComponents())

                    for jj, vv in enumerate(artery_indices):
                        if np.shape([vessel_array_np[vv]])[-1] > 1:
                            vtk_array.SetTuple(jj, vessel_array_np[vv])
                        else:
                            vtk_array.SetValue(jj, vessel_array_np[vv])

                    artery_polydata.GetFieldData().AddArray(vtk_array)
                else:
                    artery_polydata.GetFieldData().AddArray(vessel_array_vtk)

        return artery_polydata

    def execute(self):
        readerV = vtk.vtkPolyDataReader()
        readerV.SetFileName(self.vascular_file)
        readerA = vtk.vtkPolyDataReader()
        readerA.SetFileName(self.airway_file)
        readerV.Update()
        if self.artery:
            vessel = self.extract_artery_polydata(readerV.GetOutput())
        else:
            vessel = readerV.GetOutput()

        readerA.Update()
        airway = readerA.GetOutput()

        array_a = dict()
        array_v = dict()

        if self.dnn_:
            aa_field_names = ["scale", "dnn_lumen_radius", "hevec0", "hevec1", "hevec2", "h0", "h1", "h2"]
            vv_field_names = ["scale", "dnn_radius", "hevec0", "hevec1", "hevec2", "h0", "h1", "h2"]
        else:
            aa_field_names = ["scale", "hevec0", "hevec1", "hevec2", "h0", "h1", "h2"]
            vv_field_names = ["scale", "hevec0", "hevec1", "hevec2", "h0", "h1", "h2"]

        for ff in aa_field_names:
            tmp = airway.GetFieldData().GetArray(ff)
            if not isinstance(tmp, vtk.vtkDataArray):
                tmp = airway.GetPointData().GetArray(ff)
            
            array_a[ff] = tmp

        for ff in vv_field_names:
            tmp = vessel.GetFieldData().GetArray(ff)
            if not isinstance(tmp, vtk.vtkDataArray):
                tmp = vessel.GetPointData().GetArray(ff)
            array_v[ff] = tmp

        pL = vtk.vtkPointLocator()
        pL.SetDataSet(vessel)
        pL.BuildLocator()

        idList = vtk.vtkIdList()
        na = airway.GetNumberOfPoints()
        a_radius = list()
        v_radius = list()
        av_ratio_to_particles = list()
        csa = np.arange(0, 40, 0.05)
        rad = np.sqrt(csa/math.pi)
        bden = np.zeros(csa.size)
        cden = np.zeros(csa.size)

        print ("Number of Airway Points " + str(na))
        if self.artery:
            print ("Number of Artery Points " + str(vessel.GetNumberOfPoints()))
        else:
            print ("Number of Vessel Points " + str(vessel.GetNumberOfPoints()))

        for kk in range(na):
            a_p = airway.GetPoint(kk)
            pL.FindClosestNPoints(20, a_p, idList)

            a_s = array_a["scale"].GetValue(kk)
            if self.dnn_:
                a_r = array_a["dnn_lumen_radius"].GetValue(kk)
            else:
                a_r = self.airway_radius_from_sigma(a_s)

            mean_v_r = list()
            a_v = array_a["hevec2"].GetTuple3(kk)
            for kk2 in range(idList.GetNumberOfIds()):
                # Get info about point
                test_id = idList.GetId(kk2)
                v_p = vessel.GetPoint(test_id)
                v_s = array_v["scale"].GetValue(test_id)
                if self.dnn_:
                    v_r = array_v["dnn_radius"].GetValue(test_id)
                else:
                    v_r = self.vessel_radius_from_sigma(v_s)

                v_v = array_v["hevec0"].GetTuple3(test_id)

                distance = LA.norm(np.array(v_p) - np.array(a_p))
                tmp_val = abs(sum(np.array(v_v) * np.array(a_v)))
                # Check to prevent domain error ir acos
                # Vector should be normalized but due to numeric errors tmp_val can be slightly > 1
                # It is not worth renormalizing v_v and a_v
                if tmp_val > 1:
                  tmp_val = 1
                
                angle1 = math.acos(tmp_val)
                vv = LA.norm(np.array(v_v)+np.array(a_v))
                foo = abs(1/(vv*distance) * sum ( (np.array(v_v)+np.array(a_v)) * (np.array(v_p)-np.array(a_p))))
                angle2 = math.acos(foo)

                # Test conditions about point
                if self.test_condition(distance, kk2, angle1, angle2, a_s, v_s, a_r, v_r):
                    #a_r=1.2
                    a_radius.append(a_r)
                    #v_r=1.0
                    v_radius.append(v_r)
                    mean_v_r.append(v_r)

            if len(mean_v_r) > 0 and np.mean(mean_v_r) > 0.0:
                av_ratio_to_particles.append(a_r / np.mean(mean_v_r))
            else:
                av_ratio_to_particles.append(0.0)

        if self.add_ratio_to_particles:
            av_ratio_to_particles = np.asarray(av_ratio_to_particles)
            vtk_array = vtk.vtkFloatArray()
            vtk_array.SetNumberOfValues(airway.GetNumberOfPoints())
            vtk_array.SetNumberOfComponents(1)
            vtk_array.SetName('AV_ratio')

            for ii in range(av_ratio_to_particles.shape[0]):
                vtk_array.SetValue(ii, av_ratio_to_particles[ii])
            airway.GetPointData().AddArray(vtk_array)
            writer = vtk.vtkPolyDataWriter()
            file_name = self.airway_file.split('/')[-1].split('.')[0]
            out_file_name = self.output_prefix + '_{}.vtk'.format('AVRatio')
            writer.SetFileName(out_file_name)

            if vtk.VTK_MAJOR_VERSION <= 5:
                writer.SetInput(airway)
            else:
                writer.SetInputData(airway)

            writer.SetFileTypeToBinary()
            writer.Update()

        a_radius = np.array(a_radius,dtype=float)
        v_radius = np.array(v_radius,dtype=float)
                        #bron_array =((a_radius**2)/(v_radius**2))
                        #accept_mask = self.reject_outliers_mask(bron_array)
                        #print np.mean(a_radius2)
        accept_a_mask = (np.abs(a_radius - np.mean(a_radius)) < 4 * np.std(a_radius))
        accept_v_mask = (np.abs(v_radius - np.mean(v_radius)) < 4 * np.std(v_radius))
                        #accept_a_mask = self.reject_outliers_mask(a_radius2)
                        #accept_v_mask = self.reject_outliers_mask(v_radius2)
        
        a_radius = a_radius[np.logical_and(accept_a_mask, accept_v_mask)]
        v_radius = v_radius[np.logical_and(accept_a_mask, accept_v_mask)]

        for a_r,v_r in zip(a_radius,v_radius):
            bden += 1/(math.sqrt(2*math.pi)*self.sigma) * a_r/v_r * np.exp(-(csa-math.pi*(v_r)**2)**2/(2*self.sigma**2))
            cden += 1/(math.sqrt(2*math.pi)*self.sigma) *           np.exp(-(csa-math.pi*(v_r)**2)**2/(2*self.sigma**2))
    
        # This is how you can do it using a kde method in scipy
        print ("Number of computing points "+str(len(a_radius)))
        csa_samples = math.pi*(v_radius**2)
        kcden = kde.gaussian_kde(csa_samples)
        # cden = kcden(csa)
        bden = bden/(float(len(a_radius)))
        cden = cden/(float(len(a_radius)))
        cden[cden<np.spacing(1e10)]=np.spacing(1e10)
        bron = bden/cden
        print (kcden.factor)

        # bden[cden<0.01]=0
        if self.plot == True:
            fig=plt.figure()
            ax1=fig.add_subplot(211)
            ax1.plot(rad,bron)
            ax1.set_ylim([0,2])
            ax1.grid(True)
            plt.ylabel('Airway Radius / Vessel Radius')
            ax2=fig.add_subplot(212,sharex=ax1)
            ax2.plot(rad,cden)
            ax2.grid(True)
            plt.xlabel('Vessel Radius (mm)')
            plt.ylabel('CSA Density')
            #ax=fig.add_subplot(223)
            #ax.plot(rad,kcden(csa))
            #ax=fig.add_subplot(224)
            #ax.scatter(v_radius,np.array(a_radius)/np.array(v_radius))
            fig.savefig(self.output_prefix+'_bronchiectasisPlot.png',dpi=180)

        #print np.mean(sum(ba/ca[ba/ca>1]))
    
        # Compute phenotypes and save result
        # Mean Bronchiectasis Phenotpyes
        ser=list()
        col=list()
        col.append('CID')
        ser.append(os.path.split(self.output_prefix)[1])
        col.append('NPoints')
        ser.append(len(a_radius))
        # Compute distribution of ratio with CSA
        foo = np.array(a_radius)/np.array(v_radius)
        col.append('meanRatio')
        ser.append(np.mean(foo[foo>0]))
        col.append('stdRatio')
        ser.append(np.std(foo[foo>0]))

        for th in (0,5,10,20,30):
            mb=np.mean(bron[csa>th])
            col.append('meanBgt'+str(th))
            stdb = np.std(bron[csa>th])
            col.append('stdBgt'+str(th))
            ser.append(mb)
            ser.append(stdb)
                
        # Integral Bronchiectasis Phenotpyes
        delta=(csa[2]-csa[1])
        for th in (0,5,10,20,30):
            int_val = bron[np.logical_and((bron-1)>0,csa>th)]-1
            intb = delta*np.sum(int_val) /(delta * np.sum(csa>th) )
            col.append('intBgt'+str(th))
            ser.append(intb)
                
        # Percentage of bronchiectasis
        for th in (0,5,10,20,30):
            perc= 100*np.sum(np.logical_and((bron-1)>0,csa>th))/np.sum(csa>th)
            col.append('Bpercgt'+str(th))
            ser.append(perc)

        # Integral of CSA density function to have a reference
        for th in (0,5,10,20,30):
            int_val = cden[csa>th]
            intc = delta*np.sum(int_val)
            col.append('intCSAgt'+str(th))
            ser.append(intc)
        df=pd.DataFrame(np.array(ser).reshape([1,len(ser)]),columns=col)
        df.to_csv(self.output_prefix+'_bronchiectasisPhenotypes.csv')

    @staticmethod
    def reject_outliers_mask(data, m=2.0):
        return np.abs(data - np.mean(data)) < m * np.std(data)

    @staticmethod
    def airway_radius_from_sigma(sigma):
        return 0.625 * math.sqrt(2) * sigma

    @staticmethod
    def vessel_radius_from_sigma(sigma):
        return 0.625 * math.sqrt(2) * sigma
                                    

if __name__ == "__main__":
    desc = """Compute bronchiecstatis phenotypes"""
                                    
    parser = OptionParser(description=desc)
    parser.add_option('-v',help='VTK vessel particle filename',
                      dest='vessel_file',metavar='<string>',default=None)
    parser.add_option('-a',help='VTK airway particle filename',
                    dest='airway_file',metavar='<string>',default=None)
    parser.add_option('-o',help='Output prefix name',
                                    dest='output_prefix',metavar='<string>',default=None)
    parser.add_option('--p', help='Flag to enable plotting', dest='plot', action='store_true')
    parser.add_option('--dnn', help='Flag to use dnn measurements', action="store_true", dest="dnn")
    parser.add_option('--artery', help='Flag to use only particles classified as artery', action="store_true",
                      dest="artery")
    parser.add_option('--ratio_to_particles', help='Flag to add AV ratio to airway particle file', action="store_true",
                      dest="ratio_to_particles")

    (options, args) = parser.parse_args()
    bb = Bronchiectasis(options.vessel_file, options.airway_file, options.output_prefix, options.plot)
    bb.dnn_ = options.dnn
    bb.artery = options.artery
    bb.add_ratio_to_particles = options.ratio_to_particles
    bb.execute()
