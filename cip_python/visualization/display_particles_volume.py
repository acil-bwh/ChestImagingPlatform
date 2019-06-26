import vtk
import math
import nrrd
import numpy as np
from optparse import OptionParser
from vtk.util.numpy_support import vtk_to_numpy

class DisplayParticlesVolume:
    def __init__(self, file_list,spacing_list,feature_type,irad = 1.2, h_th=-200,
                 glyph_type='sphere', glyph_scale_factor=1,use_field_data=True, opacity_list=[],
                 color_list=[], lung=[]):

        assert feature_type == "ridge_line" or feature_type == "valley_line" \
        or feature_type == "ridge_surface" or feature_type == "valley_surface" \
        or feature_type == "vessel" or feature_type == "airway" \
        or feature_type == "fissure", "Invalid feature type"

        if feature_type == "airway":
          feature_type = "valley_line"
        elif feature_type == "vessel":
          feature_type = "ridge_line"
        elif feature_type == "fissure":
          feature_type = "ridge_surface"

        self.mapper_list = list()
        self.actor_list  = list()
        self.glyph_list  = list()
        self.glyph_type  = glyph_type
        self.file_list   = file_list
        self.spacing_list = spacing_list
        self.opacity_list = opacity_list
        self.irad = irad
        self.h_th = h_th
        self.color_list = color_list
        self.lung = lung
        self.use_field_data = use_field_data
        self.feature_type = feature_type
        self.normal_map=dict()
        self.normal_map['ridge_line'] = "hevec0"
        self.normal_map['valley_line'] = "hevec2"
        self.normal_map['ridge_surface'] = "hevec2"
        self.normal_map['valley_surface'] = "hevec0"
        self.strength_map=dict()
        self.strength_map['ridge_line'] = "h1"
        self.strength_map['valley_line'] = "h1"
        self.strength_map['ridge_surface'] = "h2"
        self.strength_map['valley_surface'] = "h0"

        if feature_type == 'ridge_line' or feature_type == 'valley_line':
            self.height = irad
            self.radius = 0.5
        elif feature_type == 'ridge_surface' or feature_type == 'valley_surface':
            self.height = 0.5
            self.radius = irad

        self.min_rad = 0.5
        self.max_rad = 6
        self.glyph_scale_factor = glyph_scale_factor

        self.capture_prefix = ""
        self.capture_count = 1
        self.image_count = 1

        # VTK Objects
        self.ren = vtk.vtkOpenGLRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

        # Volume rendering objects
        self.volumeActor = vtk.vtkVolume()
        self.volumeMapper = vtk.vtkVolumeRayCastMapper()
        #self.volumeMapper = vtk.vtkGPUVolumeRayCastMapper()

        # Variables for the interaction
        self.volume_loaded = False
        self.volume_added_to_renderer = False
        self.volume_size = [0,0,0]

        self.particles_loaded = False
        self.particles_added_to_renderer = False
        self.planeWidgetX = vtk.vtkImagePlaneWidget()
        self.planeWidgetY = vtk.vtkImagePlaneWidget()
        self.planeWidgetZ = vtk.vtkImagePlaneWidget()
        self.lastPlaneWidget = self.planeWidgetX
        self.lastPlaneWidgetIndex = 0
        self.planeWidgetLayer = [0,0,0]
        self.boxWidgetVolume = vtk.vtkBoxWidget()
        self.boxWidgetParticles = vtk.vtkBoxWidget()

        self.planesParticles = vtk.vtkPlanes()
        self.planesParticles.SetBounds(-10000,10000,-10000,10000,-10000,10000); # For some reason planes needs initializaiton
        self.planesVolume = vtk.vtkPlanes()
        self.planesVolume.SetBounds(-10000,10000,-10000,10000,-10000,10000); # For some reason planes needs initializaiton



    def compute_radius (self,poly,spacing):
        if self.use_field_data == False:
            scale = poly.GetPointData().GetArray("scale")
            strength = poly.GetPointData().GetArray(self.strength_map[self.feature_type])
            val = poly.GetPointData().GetArray('val')
        else:
            scale=poly.GetFieldData().GetArray("scale")
            strength = poly.GetFieldData().GetArray(self.strength_map[self.feature_type])
            val = poly.GetFieldData().GetArray('val')

        np = poly.GetNumberOfPoints()
        print np
        radiusA=vtk.vtkDoubleArray()
        radiusA.SetNumberOfTuples(np)
        si=0.2

        print "Feature type " + self.feature_type
        print "Strength feature " + self.strength_map[self.feature_type]

        arr = vtk_to_numpy(strength)
        print arr[0]
        for kk in range(np):

            #rad=math.sqrt(2.0) * ( math.sqrt(spacing**2.0 (ss**2.0 + si**2.0)) - 1.0*spacing*s0 )
            if self.feature_type == 'ridge_line':
              test= arr[kk] > self.h_th
            elif self.feature_type == 'valley_line':
              test= arr[kk] < self.h_th
            elif self.feature_type == 'ridge_surface':
              test= arr[kk] > self.h_th
            elif self.feature_type == 'valley_surface':
              test= arr[kk] < self.h_th

            ss=float(scale.GetValue(kk))
            rad=math.sqrt(2)*spacing*ss
            # test = False

            # if test==True:
                # rad=0
            # if rad < spacing/2:
                # rad=0
            radiusA.SetValue(kk,rad)

        poly.GetPointData().SetScalars(radiusA)
        return poly

    def create_glyphs (self, poly):
        if self.glyph_type == 'sphere':
            glyph = vtk.vtkSphereSource()
            glyph.SetRadius(1)
            glyph.SetPhiResolution(8)
            glyph.SetThetaResolution(8)
        elif self.glyph_type == 'cylinder':
            glyph = vtk.vtkCylinderSource()
            glyph.SetHeight(self.height)
            glyph.SetRadius(self.radius)
            glyph.SetCenter(0,0,0)
            glyph.SetResolution(10)
            glyph.CappingOn()

        tt = vtk.vtkTransform()
        tt.RotateZ(90)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInput(glyph.GetOutput())
        tf.SetTransform(tt)
        tf.Update()

        glypher = vtk.vtkGlyph3D()
        glypher.SetInput(poly)
        glypher.SetSource(tf.GetOutput())
        glypher.SetVectorModeToUseNormal()
        glypher.SetScaleModeToScaleByScalar()
        glypher.SetScaleFactor(self.glyph_scale_factor)
        print "Scale factor %f" % self.glyph_scale_factor
        glypher.Update()

        # Color the glyph according to whatever other value is in the vtk file
        


        glypher.GetOutput().GetPointData().SetScalars(glypher.GetOutput().GetPointData().GetArray("h2"))
        
        # print(glypher.GetOutput())

        return glypher

    def create_actor (self, glyph , opacity=1,color=[0.1,0.1,0.1]):

        ## Adds clipping abilities to the actor
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputConnection(glyph.GetOutputPort())
        clipper.SetClipFunction(self.planesParticles)
        clipper.InsideOutOn()

        mapper=vtk.vtkPolyDataMapper()
        mapper.SetColorModeToMapScalars()

        # mapper.SetScalarRange(-700, -400)  # This is for val
        # mapper.SetScalarRange(0, 300)  # This is for val
        mapper.SetScalarRange(self.min_rad,self.max_rad)
        mapper.SetScalarRange(-500, -200)
        # if len(color) > 0:
            # mapper.ScalarVisibilityOn()
        # #mapper.SetScalarRange(self.min_rad,self.max_rad)
        # else:
            # mapper.SetColorModeToDefault()
            # print color

        mapper.SetInputConnection(clipper.GetOutputPort())
        # mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if len(color) > 0 :
            actor.GetProperty().SetColor(color)
            print "Setting the color of the particles to"
            print color

        actor.GetProperty().SetOpacity(opacity)
        actor.VisibilityOn()
        self.mapper_list.append(mapper)

        self.actor_list.append(actor)
        self.ren.AddActor(actor)
        self.particles_loaded = True
        self.particles_added_to_renderer = True
        return actor

    def add_color_bar(self):
        colorbar=vtk.vtkScalarBarActor()
        colorbar.SetMaximumNumberOfColors(400)
        colorbar.SetLookupTable(self.mapper_list[0].GetLookupTable())
        colorbar.SetWidth(0.09)
        colorbar.SetPosition(0.91,0.1)
        colorbar.SetLabelFormat("%.3g")
        colorbar.VisibilityOn()

        if len(self.color_list) == 0:
            self.ren.AddActor(colorbar)


    def render(self,widht=800,height=800):
        # create a rendering window and renderer
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(widht,height)
        self.renWin.SetAAFrames(0)

        # create a renderwindowinteractor
        self.iren.SetRenderWindow(self.renWin)

        # enable user interface interactor
        # Set observer
        self.iren.AddObserver('KeyPressEvent', self.keyboard_interaction, -1.0)

        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

    def execute(self):
        for kk,file_name in enumerate(self.file_list):
            print file_name
            reader=vtk.vtkPolyDataReader()
            reader.SetFileName(file_name)
            reader.Update()

            poly = self.compute_radius(reader.GetOutput(),spacing_list[kk])
            if self.use_field_data == False:
                poly.GetPointData().\
                    SetNormals(poly.GetPointData().\
                               GetArray(self.normal_map[self.feature_type]))
            else:
                poly.GetPointData().\
                    SetNormals(poly.GetFieldData().\
                               GetArray(self.normal_map[self.feature_type]))

            glypher=self.create_glyphs(poly)
            if len(self.color_list) <= kk:
                color=[]
            else:
                color=self.color_list[kk]
            if len(self.opacity_list) <= kk:
                opacity=1
            else:
                opacity=self.opacity_list[kk]
            self.create_actor(glypher,color=color,opacity=opacity)

            # Adds the box interaction mechanism

            self.boxWidgetParticles.SetInteractor(self.iren)
            self.boxWidgetParticles.SetPlaceFactor(1.25)
            self.boxWidgetParticles.SetInput(glypher.GetOutput())
            self.boxWidgetParticles.PlaceWidget()
            self.boxWidgetParticles.AddObserver("EndInteractionEvent", self.SelectPolygons)
            self.boxWidgetParticles.EnabledOff()

            ## The SelectPolygons function is slow as hell - therefore it is only updated when it is released
            # self.boxWidgetParticles.AddObserver("StartInteractionEvent", self.StartInteraction)
            # self.boxWidgetParticles.AddObserver("InteractionEvent", self.SelectPolygons)
            # self.boxWidgetParticles.AddObserver("EndInteractionEvent", self.EndInteraction)



        if len(self.lung)>0:
            ## Generate a transfer function - the phantom is in the range -400, -200


            # Converts the nrrd into a vtk volume
            readdata, options = nrrd.read(self.lung)
            print options


            ## This is to be changed!! FIXME - the minimum value does not need to be of interest - find the min val of HUs
            readdata = readdata.transpose([2,1,0])   # yup, nrrd and axes

            minval = readdata.min()
            print minval
            print readdata.max()


            tfun = vtk.vtkPiecewiseFunction()
            # tfun.AddPoint(100, 0);
            # tfun.AddPoint(600, 1.0);
            # tfun.AddPoint(700, 1.0);
            # tfun.AddPoint(500, 0.0);
            # tfun.AddPoint(1000, 0);

            tfun.AddPoint(-minval-2000, 0);
            tfun.AddPoint(-minval-1000, 0.0);
            tfun.AddPoint(-minval-300, 1.0);
            tfun.AddPoint(-minval-100, 1.0);
            tfun.AddPoint(-minval-80, 0);
            readdata = np.ushort(readdata - minval)
            print readdata.min()
            print readdata.max()


            # tfun.AddPoint(-1000, 0.0);
            # tfun.AddPoint(- 800, 0.0);
            # tfun.AddPoint(- 400, 1.0);
            # tfun.AddPoint(- 200, 1.0);
            # tfun.AddPoint(- 180, 0.0);
            # tfun.AddPoint(  100, 0.0);
            # tfun.AddPoint(  300, 0.0);
            # tfun.AddPoint(  800, 0.0);
            # tfun.AddPoint( 1070, 0.0);

            sz = readdata.shape
            dataSpacing = options.get('space directions')   # scale of the image
            dataSpacing = [dataSpacing[0][0], dataSpacing[1][1], dataSpacing[2][2]]
            dataOrigin = options.get('space origin')
            dataImporter = vtk.vtkImageImport()
            data_string = readdata.tostring()
            dataImporter.CopyImportVoidPointer(data_string, len(data_string))

            dataImporter.SetDataScalarTypeToUnsignedShort()
            dataImporter.SetNumberOfScalarComponents(1)   # it is a bw image
            dataImporter.SetDataOrigin(dataOrigin) # location of the image
            dataImporter.SetDataSpacing(dataSpacing)
            dataImporter.SetDataExtent(0, sz[2]-1, 0, sz[1]-1, 0, sz[0]-1)
            dataImporter.SetWholeExtent(0, sz[2]-1, 0, sz[1]-1, 0, sz[0]-1)
            self.volume_size = [sz[2]-1, sz[1]-1, sz[0]-1]
            dataImporter.Update()

            print dataImporter.GetDataScalarTypeAsString()

            volumeProperty = vtk.vtkVolumeProperty()
            volumeProperty.SetScalarOpacity(tfun)
            volumeProperty.SetInterpolationTypeToLinear()


            # How is the volume going to be rendered?
            compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
            self.volumeMapper.SetVolumeRayCastFunction(compositeFunction)
            self.volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

            # Finally generate the volume
            self.volumeActor.SetMapper(self.volumeMapper)
            self.volumeActor.SetProperty(volumeProperty)

            # Add the new actors to the rendering pipeline
            self.ren.AddVolume(self.volumeActor)
            self.volume_added_to_renderer = True
            self.volume_loaded = True

            # Adds the box interaction mechanism
            self.boxWidgetVolume.SetInteractor(self.iren)
            self.boxWidgetVolume.SetPlaceFactor(1.0)

            self.boxWidgetVolume.SetInput(dataImporter.GetOutput())

            self.boxWidgetVolume.PlaceWidget()
            self.boxWidgetVolume.InsideOutOn()
            self.boxWidgetVolume.AddObserver("StartInteractionEvent", self.StartInteraction)
            self.boxWidgetVolume.AddObserver("InteractionEvent", self.ClipVolumeRender)
            self.boxWidgetVolume.AddObserver("EndInteractionEvent", self.EndInteraction)

            outlineProperty = self.boxWidgetVolume.GetOutlineProperty()
            outlineProperty.SetRepresentationToWireframe()
            outlineProperty.SetAmbient(1.0)
            outlineProperty.SetAmbientColor(1, 1, 1)
            outlineProperty.SetLineWidth(3)

            selectedOutlineProperty = self.boxWidgetVolume.GetSelectedOutlineProperty()
            selectedOutlineProperty.SetRepresentationToWireframe()
            selectedOutlineProperty.SetAmbient(1.0)
            selectedOutlineProperty.SetAmbientColor(1, 0, 0)
            selectedOutlineProperty.SetLineWidth(3)

            # picker = vtk.vtkCellPicker()
            # picker.SetTolerance(0.005)

            # Adds the plane interactors
            self.planeWidgetX.DisplayTextOn()
            self.planeWidgetY.DisplayTextOn()
            self.planeWidgetY.DisplayTextOn()
            self.planeWidgetX.SetInput(dataImporter.GetOutput())
            self.planeWidgetY.SetInput(dataImporter.GetOutput())
            self.planeWidgetZ.SetInput(dataImporter.GetOutput())
            self.planeWidgetX.SetPlaneOrientationToXAxes()
            self.planeWidgetY.SetPlaneOrientationToYAxes()
            self.planeWidgetZ.SetPlaneOrientationToZAxes()
            self.planeWidgetX.SetSliceIndex(1)
            self.planeWidgetY.SetSliceIndex(1)
            self.planeWidgetZ.SetSliceIndex(1)

            self.planeWidgetX.SetInteractor(self.iren)
            self.planeWidgetY.SetInteractor(self.iren)
            self.planeWidgetZ.SetInteractor(self.iren)



        self.ren.SetBackground(0, 0, 0)
        self.add_color_bar()
        self.render()

    ###########################################################################
    # Interaction related functions

    def keyboard_interaction(self,obj, event):
      # if self.capture_prefix == "":
        # return
      key = obj.GetKeySym()
      print "Key press "+key
      if key == "s":
        ff = vtk.vtkWindowToImageFilter()
        sf = vtk.vtkPNGWriter()

        ff.SetInput(self.renWin)
        ff.SetMagnification(4)
        sf.SetInput(ff.GetOutput())
        sf.SetFileName(self.capture_prefix+ "%03d.png" % self.capture_count )
        self.renWin.Render()
        ff.Modified()
        sf.Write()
        self.capture_count = 1+self.capture_count
      if key == "P":
          if self.particles_loaded:
              if self.boxWidgetParticles.GetEnabled():
                  self.boxWidgetParticles.EnabledOff()
              else:
                  self.boxWidgetParticles.EnabledOn()
                  self.boxWidgetVolume.EnabledOff()

      # Volume interaction
      if key == "p":
        if self.particles_loaded:
          if self.particles_added_to_renderer:
              for aa in self.actor_list:
                  self.ren.RemoveActor(aa)
              self.particles_added_to_renderer = False
              self.renWin.Render()
          else:
              for aa in self.actor_list:
                  self.ren.AddActor(aa)
              self.particles_added_to_renderer = True
              self.renWin.Render()
      if self.volume_loaded:
        if key == "v":
            if self.volume_added_to_renderer:
                self.ren.RemoveVolume(self.volumeActor)
                self.volume_added_to_renderer = False
                self.renWin.Render()
            else:
                self.ren.AddVolume(self.volumeActor)
                self.volume_added_to_renderer = True
                self.renWin.Render()
        if key == "V":
            if self.boxWidgetVolume.GetEnabled():
                self.boxWidgetVolume.EnabledOff()
            else:
                self.boxWidgetVolume.EnabledOn()
                self.boxWidgetParticles.EnabledOff()
        if key == "x":
            if self.planeWidgetX.GetEnabled():
                self.planeWidgetX.EnabledOff()
            else:
                self.planeWidgetX.EnabledOn()
                self.lastPlaneWidget = self.planeWidgetX
                self.lastPlaneWidgetIndex = 0
        if key == "y":
            if self.planeWidgetY.GetEnabled():
                self.planeWidgetY.EnabledOff()
            else:
                self.planeWidgetY.EnabledOn()
                self.lastPlaneWidget = self.planeWidgetY
                self.lastPlaneWidgetIndex = 1
        if key == "z":
            if self.planeWidgetZ.GetEnabled():
                self.planeWidgetZ.EnabledOff()
            else:
                self.planeWidgetZ.EnabledOn()
                self.lastPlaneWidget = self.planeWidgetZ
                self.lastPlaneWidgetIndex = 2
        if key == "t":
            self.planeWidgetLayer[self.lastPlaneWidgetIndex] \
                = self.planeWidgetLayer[self.lastPlaneWidgetIndex] + 1
            if self.planeWidgetLayer[self.lastPlaneWidgetIndex] > self.volume_size[self.lastPlaneWidgetIndex]:
                self.planeWidgetLayer[self.lastPlaneWidgetIndex] = self.volume_size[self.lastPlaneWidgetIndex]
            self.lastPlaneWidget.SetSliceIndex(self.planeWidgetLayer[self.lastPlaneWidgetIndex])
            self.renWin.Render()
        if key == "r":
            self.planeWidgetLayer[self.lastPlaneWidgetIndex] \
                = self.planeWidgetLayer[self.lastPlaneWidgetIndex] - 1
            if self.planeWidgetLayer[self.lastPlaneWidgetIndex] < 0:
                self.planeWidgetLayer[self.lastPlaneWidgetIndex] = 0
            self.lastPlaneWidget.SetSliceIndex(self.planeWidgetLayer[self.lastPlaneWidgetIndex])
            self.renWin.Render()


    def StartInteraction(self, obj, event):
        self.renWin.SetDesiredUpdateRate(1)

    # When interaction ends, the requested frame rate is decreased to
    # normal levels. This causes a full resolution render to occur.
    def EndInteraction(self, obj, event):
        self.renWin.SetDesiredUpdateRate(0.001)

    # The implicit function vtkPlanes is used in conjunction with the
    # volume ray cast mapper to limit which portion of the volume is
    # volume rendered.
    def ClipVolumeRender(self, obj, event):
        obj.GetPlanes(self.planesVolume)
        self.volumeMapper.SetClippingPlanes(self.planesVolume)

    # This callback funciton does the actual work: updates the vtkPlanes
    # implicit function.  This in turn causes the pipeline to update.
    def SelectPolygons(self, object, event):
        object.GetPlanes(self.planesParticles)



################################################################################
# Main function to make the class a command line
################################################################################

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", help='TODO', dest="file_name")
    parser.add_option("-s", help='TODO', dest="spacing")
    parser.add_option("--feature", help='TODO', dest="feature_type", \
                      default="vessel")
    parser.add_option("--irad", help='interparticle distance', dest="irad", \
                      default=1.2)
    parser.add_option("--hth", help='TODO', dest="hth", default=0)
    parser.add_option("--color", help='TODO', dest="color_list", default="")
    parser.add_option("--opacity", help='TODO', dest="opacity_list", \
                      default="")
    parser.add_option("-l", help='Lungs NRRDs or NHDRs', dest="lung_filename", default="")
    parser.add_option("--useFieldData", help='TODO', dest="use_field_data", \
                      action="store_true", default=False)

    parser.add_option("--glpyhScale", help='TODO', dest="glyph_scale_factor", \
                        default=1)

    parser.add_option("--capturePrefix", help='Prefix filename to save screenshots. This options enables screen capture. Press the "s" key to capture a screenshot.', \
                      dest="capture_prefix", default="")

    (options, args) = parser.parse_args()

    translate_color = dict()
    translate_color['red'] = [1, 0.1, 0.1]
    translate_color['green'] = [0.1, 0.8, 0.1]
    translate_color['orange'] = [0.95, 0.5, 0.01]
    translate_color['blue'] = [0.1, 0.1, 0.9]

    file_list = [i for i in str.split(options.file_name,',')]
    use_field_data = options.use_field_data
    spacing_list = [float(i) for i in str.split(options.spacing,',')]
    lung_filename = options.lung_filename

    if options.opacity_list == "":
        opacity_list=[]
    else:
        opacity_list = [float(i) for i in str.split(options.opacity_list,',')]

    if options.color_list == "" :
        color_list=[]
    else:
        color_list = [translate_color[val] for val in str.split(options.color_list,',')]

    print color_list

    print use_field_data

    dv = DisplayParticlesVolume(file_list, spacing_list, options.feature_type, float(options.irad), float(options.hth), \
        'cylinder', float(options.glyph_scale_factor), use_field_data, opacity_list, color_list, lung_filename)
    dv.capture_prefix = options.capture_prefix
    dv.execute()
