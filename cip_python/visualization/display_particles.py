import vtk
import math
import numpy as np
from argparse import ArgumentParser
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk



class DisplayParticles:
    def __init__(self, file_list,spacing_list,feature_type_list,irad = 1.2, h_th_list=[],
                 glyph_type='sphere', glyph_scale_factor=1,max_rad=6.0,min_rad=0.5,use_field_data=True, opacity_list=[],
                 color_list=[], lut_list=[], lung=[],smooth_glyphs=False):
      
        for feature_type in feature_type_list:
          print (feature_type)
          assert feature_type == "ridge_line" or feature_type == "valley_line" \
          or feature_type == "ridge_surface" or feature_type == "valley_surface" \
          or feature_type == "vessel" or feature_type == "airway" \
          or feature_type == "fissure", "Invalid feature type"
      
        for kk,feature_type in enumerate(feature_type_list):
        
          if feature_type == "airway":
            feature_type_list[kk] = "valley_line"
          elif feature_type == "vessel":
            feature_type_list[kk] = "ridge_line"
          elif feature_type == "fissure":
            feature_type_list[kk] = "ridge_surface"
    
        self.no_display = False
        self.mapper_list = list()
        self.actor_list = list()
        self.glyph_list = list()
        self.glyph_type = glyph_type
        self.smooth_glyphs  = smooth_glyphs
        self.use_glyphMapper = False
        self.file_list = file_list
        self.spacing_list = spacing_list
        self.opacity_list = opacity_list
        self.irad = irad
        self.h_th_list = h_th_list
        self.color_list = color_list
        self.lut_list = lut_list
        self.lung = lung
        self.use_field_data = use_field_data
        self.feature_type_list = feature_type_list
        self.units="%%"
        self.units="mm"
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
        
        self.color_by_array_name = None #By default we color by the particle radius that is computed from scale
        
        self.glyph_output = None
        
        self.clip_radius = True
        
        self.coordinate_system = "LPS"
        
        self.lung_opacity = 0.3
  
        if feature_type == 'ridge_line' or feature_type == 'valley_line':
            self.height = irad
            self.radius = 1.0
        elif feature_type == 'ridge_surface' or feature_type == 'valley_surface':
            self.height = 1.0
            self.radius = irad

        self.resolution=10
  
        self.min_rad = min_rad
        self.max_rad = max_rad
        self.glyph_scale_factor = glyph_scale_factor

        self.capture_prefix = ""
        self.capture_count = 1
        
        #Use radius from array name, otherwise use scale
        self.radius_array_name_list=None
        
        # VTK Objects
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()
  
        self.image_count = 1
    
        #Picker capabilities
        self.picker = vtk.vtkCellPicker()
    
        #Display picking results
        self.textMapper = vtk.vtkTextMapper()
        tprop = self.textMapper.GetTextProperty()
        tprop.SetFontFamilyToArial()
        tprop.SetFontSize(10)
        tprop.BoldOn()
        tprop.ShadowOn()
        tprop.SetColor(1, 0, 0)
        self.textActor = vtk.vtkActor2D()
        self.textActor.VisibilityOff()
        self.textActor.SetMapper(self.textMapper)
        
        #Point locator to extract particle info from picker info on glyhs
        self.particles_locator=vtk.vtkPointLocator()

    def annotatePick(self,object, event):
      print("pick")
      if self.picker.GetCellId() < 0:
        self.textActor.VisibilityOff()
      else:
        selPt = self.picker.GetSelectionPoint()
        pickPos = self.picker.GetPickPosition()
        pId = self.picker.GetPointId()
        #print pId
        particle_pId=self.particles_locator.FindClosestPoint(pickPos)
        print ("(%.6f, %.6f, %.6f)"%pickPos)
        print (particle_pId)
        self.textMapper.SetInput("(%.6f, %.6f, %.6f)"%pickPos)
        self.textActor.SetPosition(selPt[:2])
        self.textActor.VisibilityOn()

    def compute_radius (self,poly,spacing,feature_type,radius_array_name,h_th):
        if self.use_field_data == False:
            scale = poly.GetPointData().GetArray("scale")
            strength = poly.GetPointData().GetArray(self.strength_map[feature_type])
            val = poly.GetPointData().GetArray('val')
            if radius_array_name is not None:
                rad_arr =poly.GetPointData().GetArray(radius_array_name)
        else:
            scale=poly.GetFieldData().GetArray("scale")
            strength = poly.GetFieldData().GetArray(self.strength_map[feature_type])
            print(strength)
            val = poly.GetFieldData().GetArray('val')
            if radius_array_name is not None:
                rad_arr =poly.GetPointData().GetArray(radius_array_name)

        numpoints  = poly.GetNumberOfPoints()
        print (numpoints)
        radiusA=vtk.vtkDoubleArray()
        radiusA.SetNumberOfTuples(numpoints)
        si=float(0.2)
        s0=float(0.2)
              
        arr = vtk_to_numpy(strength)
        print (arr[0])
        for kk in range(numpoints):
            if radius_array_name is not None:
                rad = float(rad_arr.GetValue(kk))
            else:
                ss=float(scale.GetValue(kk))
                rad=np.sqrt(2.0) * ( np.sqrt( spacing**2 * (ss**2 + si**2) ) - 1.0*spacing*s0 )
                #rad=np.sqrt(2.0)*spacing*ss
                #rad=np.sqrt(2.0)*np.sqrt(spacing**2 * (ss**2 + si**2) )
            if h_th != None:
              if feature_type == 'ridge_line':
                test= arr[kk] > h_th
              elif feature_type == 'valley_line':
                test= arr[kk] < h_th
              elif feature_type == 'ridge_surface':
                test= arr[kk] > h_th
              elif feature_type == 'valley_surface':
                test= arr[kk] < h_th
            else:
              test = False

            if test==True:
                rad=0
            if rad < spacing/2.0:
                print ("Setting point to zero "+str(kk))
                rad=0
            
            if self.clip_radius is True:
                if rad < self.min_rad or rad>self.max_rad:
                    rad=0

            radiusA.SetValue(kk,rad)

        poly.GetPointData().SetScalars(radiusA)

        # After rotating the cylinder, its long axis is X. Therefore:
        #
        #     X scale = 1       -> fixed cylinder height
        #     Y scale = radius  -> radial scaling
        #     Z scale = radius  -> radial scaling
        #
        # The source cylinder has radius 1 and height irad.

        scale_vectors = np.column_stack(
            (
                np.ones(radiusA.GetNumberOfTuples(), dtype=np.float64),
                vtk_to_numpy(radiusA),
                vtk_to_numpy(radiusA)
            )
        )   
        vtk_scales = numpy_to_vtk(scale_vectors, deep=True)
        vtk_scales.SetName("GlyphScale")
        poly.GetPointData().AddArray(vtk_scales)

        return poly

    def create_glyphs (self, poly):    
        if self.glyph_type == 'sphere':
            glyph = vtk.vtkSphereSource()
            glyph.SetRadius(1)
            glyph.SetPhiResolution(self.resolution)
            glyph.SetThetaResolution(self.resolution)
        elif self.glyph_type == 'cylinder':
            glyph = vtk.vtkCylinderSource()
            glyph.SetHeight(self.height)
            glyph.SetRadius(self.radius)
            glyph.SetCenter(0.0,0.0,0.0)
            glyph.SetResolution(self.resolution)
            glyph.CappingOff()

        tt = vtk.vtkTransform()
        tt.RotateZ(90)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(glyph.GetOutputPort())
        tf.SetTransform(tt)
        tf.Update()

        # #Alternative use of a Glypher that scales independently along X,Y or Z.
        # try:
        #     glypher = vtk.vtkGlyph3DWithScaling()
        #     glypher.ScalingXOff()
        #     glypher.ScalingYOn()
        #     glypher.ScalingZOn()
        # except NameError:
        #     glypher = vtk.vtkGlyph3D()


        if self.use_glyphMapper:
            mapper = vtk.vtkGlyph3DMapper()
            mapper.SetInputData(poly)
            mapper.SetSourceConnection(tf.GetOutputPort())
            mapper.ScalingOn()
            mapper.SetScaleModeToScaleByVectorComponents()
            mapper.SetScaleArray("GlyphScale")
            mapper.OrientOn()
            orientation_array_name=poly.GetPointData().GetNormals().GetName()
            mapper.SetOrientationArray(orientation_array_name)
            mapper.SetScaleFactor(self.glyph_scale_factor)   

            return mapper

        else:

            glypher = vtk.vtkGlyph3D()
            
            print (glypher.GetClassName())
            glypher.SetInputData(poly)
            glypher.SetSourceConnection(tf.GetOutputPort())
            # Orientation comes from point normals
            glypher.OrientOn()
            glypher.SetVectorModeToUseNormal()

            #Isotropic scaling: Old code
            glypher.SetScaleModeToScaleByScalar()
            # Apply independent component scaling:
            # scale = (1, radius, radius).
            #glypher.SetScaleModeToScaleByVectorComponents()
            glypher.SetScaleFactor(self.glyph_scale_factor)
            glypher.Update()

            #glypher=self.materialize_glyphs(poly,tf.GetOutputPort())

            if self.smooth_glyphs is True:

                # Step 3: Merge nearby points to ensure connectivity
                clean_filter = vtk.vtkCleanPolyData()
                clean_filter.ConvertPolysToLinesOff()
                clean_filter.PointMergingOn()
                clean_filter.ToleranceIsAbsoluteOn()
                clean_filter.SetTolerance(self.irad/2.0)
                clean_filter.SetInputConnection(glypher.GetOutputPort())
                clean_filter.Update()

                # Step 4: Smooth the mesh
                smooth_filter = vtk.vtkSmoothPolyDataFilter()
                smooth_filter.SetInputConnection(clean_filter.GetOutputPort())
                smooth_filter.SetNumberOfIterations(50)
                smooth_filter.SetRelaxationFactor(0.3)
                smooth_filter.FeatureEdgeSmoothingOn()
                smooth_filter.BoundarySmoothingOn()
                smooth_filter.Update()

                smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
                smooth_filter.SetInputConnection(clean_filter.GetOutputPort())
                smooth_filter.SetNumberOfIterations(30)
                smooth_filter.BoundarySmoothingOn()
                smooth_filter.NonManifoldSmoothingOn()
                smooth_filter.NormalizeCoordinatesOn()
                smooth_filter.Update()

                return smooth_filter

            else:

                return glypher


    def materialize_glyphs(self,poly, source_polydata, scale_name="GlyphScale"):

        #Orientation is in normals
        pd = poly.GetPointData()
        scales = vtk_to_numpy(pd.GetArray(scale_name))
        orientations = vtk_to_numpy(pd.GetNormals())
        append = vtk.vtkAppendPolyData()
        for i in range(poly.GetNumberOfPoints()):
            px, py, pz = poly.GetPoint(i)
            sx, sy, sz = scales[i]
            direction = orientations[i]
            transform = vtk.vtkTransform()
            # Apply scaling in source coordinates
            transform.Scale(sx, sy, sz)
            # rotate source X axis to direction
            d = direction / np.linalg.norm(direction)
            x_axis = np.array([1.0, 0.0, 0.0])
            axis = np.cross(x_axis, d)
            dot = np.clip(np.dot(x_axis, d), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))

            if np.linalg.norm(axis) > 1e-8:
                axis /= np.linalg.norm(axis)
                transform.RotateWXYZ(angle, *axis)
            
            transform.Translate(px, py, pz)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputConnection(source_polydata)
            tf.SetTransform(transform)
            tf.Update()
                    
            glyph_poly = vtk.vtkPolyData()
            glyph_poly.DeepCopy(tf.GetOutput())

            # ------------------------------------
            # Copy scalar value to every
            # point in this glyph
            # ------------------------------------
            scalar_value = sy
            glyph_scalars = vtk.vtkDoubleArray()
            glyph_scalars.SetName("Radius")

            glyph_scalars.SetNumberOfComponents(1)
            glyph_scalars.SetNumberOfTuples(
                glyph_poly.GetNumberOfPoints()
            )

            glyph_scalars.Fill(float(scalar_value))
            glyph_poly.GetPointData().AddArray(glyph_scalars)
            append.AddInputData(glyph_poly)

        append.Update()
        return append
        #output = vtk.vtkPolyData()
        #output.ShallowCopy(append.GetOutput())
        #return output


    def create_lut (self, lut_arr):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(num_colors)
        lut.SetHueRange(0,1)
        lut.SetSaturationRange(0,1)
        lut.SetValueRange(0,1)
        lut.Build()
        for ii,cc in enumerate(lut_arr):
            lut.SetTableValue(ii,lut_arr[ii])

        return lut

    def output_port_to_polydata(output_port):
        producer = output_port.GetProducer()
        producer.Update()
        output = producer.GetOutputDataObject(
            output_port.GetIndex()
        )
        return vtk.vtkPolyData.SafeDownCast(output)

    def create_actor (self, glyph , opacity=1,color=[0.1,0.1,0.1],color_by_array_name=None,lut=None):
        #Accomodate the option if glpy is indeed a Glyp3DMapper otherwise create a vtkPolyDataMapper
        if isinstance(glyph, vtk.vtkGlyph3DMapper):
            mapper=glyph
        else:
            mapper=vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
        
        mapper.SetColorModeToMapScalars()
        if color_by_array_name is not None:
            #glyph_pd=glyph.GetOutput()
            glyph_pd = mapper.GetInput()
            color_array=glyph_pd.GetPointData().GetArray(color_by_array_name)
            if color_array is None:
                raise ValueError(
                    f"Color array '{color_by_array_name}' "
                    f"not found in input polydata"
                )
            
            range_val=color_array.GetRange()
            mapper.SetScalarRange(range_val[0],range_val[1])
            #mapper.SetScalarRange(18,35)
            mapper.SetScalarModetoUsePointFieldData()
            mapper.SetColorArray(color_by_array_name)
            #glyph_pd.GetPointData().SetScalars(glyph_pd.GetPointData().GetArray(color_by_array_name))

            if lut is not None:
              mapper.SetLookupTable(lut)
        else:
            mapper.SetScalarRange(self.min_rad,self.max_rad)
        if len(color) == 3 :
            mapper.ScalarVisibilityOff()
        #mapper.SetScalarRange(self.min_rad,self.max_rad)
            #else:
        #    mapper.SetColorModeToDefault()
        print (color) 
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if len(color) == 3 :
            actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        self.mapper_list.append(mapper)
        self.actor_list.append(actor)
        for aa in self.actor_list:
            self.ren.AddActor(aa)
            #self.ren.SetBackground(1,1,1)
            self.ren.SetBackground(0,0,0)

        return actor

    def set_camera(self):

        #Get mean bounding box for all actors
        actors = self.ren.GetActors()
        actors.InitTraversal()

        bounds_list = []
        while True:
            actor = actors.GetNextActor()
            if not actor:
                break
            bounds = actor.GetBounds()  # Get bounds [xmin, xmax, ymin, ymax, zmin, zmax]
            if bounds:
                bounds_list.append(bounds)

        if not bounds_list:
            raise ValueError("No actors with bounds in the renderer.")

        # Convert to numpy array for averaging
        bounds_array = np.array(bounds_list)

        # Compute the average bounding box
        mean_bounds = bounds_array.mean(axis=0)

        # Compute the mean center (midpoints of the averaged bounds)
        mean_center = [
            (mean_bounds[0] + mean_bounds[1]) / 2,  # Center X
            (mean_bounds[2] + mean_bounds[3]) / 2,  # Center Y
            (mean_bounds[4] + mean_bounds[5]) / 2,  # Center Z
        ]

        print("Setting camera")
        print(mean_center)

        #Set initial camera view 
        camera=self.ren.GetActiveCamera()
        camera.SetFocalPoint(mean_center[0], mean_center[1], mean_center[2])
        camera.SetPosition(mean_center[0] - 500, mean_center[1], mean_center[2])
        camera.SetViewUp(0,0,1)

    def set_camera_view(self, view):
        camera = self.ren.GetActiveCamera()
        # Get center of all visible actors
        bounds = self.ren.ComputeVisiblePropBounds()
        center = [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]

        # Use object size to determine camera distance
        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]

        distance = 2.0 * max(size_x, size_y, size_z)
        cx, cy, cz = center
        camera.SetFocalPoint(cx, cy, cz)
        if view == "sagittal":
            # LPS:
            # X = Left
            # Y = Posterior
            # Z = Superior

            # Looking from patient's right toward left
            camera.SetPosition(cx - distance, cy, cz)
            camera.SetViewUp(0, 0, 1)

        elif view == "coronal":
            # Looking from anterior toward posterior
            camera.SetPosition(cx, cy - distance, cz)
            camera.SetViewUp(0, 0, 1)

        elif view == "axial":
            # Looking from inferior toward superior
            camera.SetPosition(cx, cy, cz - distance)
            # Need Y direction as the screen vertical orientation
            camera.SetViewUp(0, -1, 0)
        else:
            raise ValueError(
                f"Unknown camera view: {view}"
            )

        camera.OrthogonalizeViewUp()
        # Optional, but for anatomical views I generally recommend
        # parallel projection instead of perspective.
        camera.ParallelProjectionOn()
        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()
        self.renWin.Render()

    def flip_camera(self):
        camera = self.ren.GetActiveCamera()
        # Get center of all visible actors
        bounds = self.ren.ComputeVisiblePropBounds()
        center = [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]

        # Use object size to determine camera distance
        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]

        distance = 2.0 * max(size_x, size_y, size_z)
        cx, cy, cz = center
        camera_pos=camera.GetPosition()
        camera_new_pos=-1.0*(np.array(camera_pos) - np.array([cx,cy,cz]))+np.array([cx,cy,cz])
        camera.SetPosition(camera_new_pos[0],camera_new_pos[1],camera_new_pos[2])

        camera.OrthogonalizeViewUp()
        # Optional, but for anatomical views I generally recommend
        # parallel projection instead of perspective.
        camera.ParallelProjectionOn()
        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()
        self.renWin.Render()

    def add_color_bar(self):
        colorbar=vtk.vtkScalarBarActor()
        colorbar.SetMaximumNumberOfColors(400)
        colorbar.SetLookupTable(self.mapper_list[0].GetLookupTable())
        colorbar.SetWidth(0.09)
        colorbar.SetPosition(0.91,0.1)
        colorbar.SetLabelFormat("%.3g {}".format(self.units))
        colorbar.VisibilityOn()
        
        if len(self.color_list) == 0:
            self.ren.AddActor(colorbar)

    def render(self,widht=800,height=800):
      
        # Now at the end of the pick event call the above function.
        self.picker.AddObserver("EndPickEvent", self.annotatePick)
  
        # create a rendering window and renderer
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(widht,height)
      #  self.renWin.SetAAFrames(0)

        # create a renderwindowinteractor
        self.iren.SetRenderWindow(self.renWin)
        self.iren.SetPicker(self.picker)

        # add actor
        self.ren.AddActor2D(self.textActor)

        # enable user interface interactor
        # Set observer
        self.iren.AddObserver('KeyPressEvent', self.capture_window, -1.0)

        self.iren.Initialize()

        #Set initial camera view 
        self.set_camera()
        self.renWin.Render()
        self.iren.Start()


                                
    def execute(self):
        for kk,file_name in enumerate(self.file_list):
            reader=vtk.vtkPolyDataReader()
            reader.SetFileName(file_name)
            reader.Update()
            
            #Locator is link to last particle file for now
            self.particles_locator.SetDataSet(reader.GetOutput())
            self.particles_locator.BuildLocator()
          
            if len(self.h_th_list)==0:
              h_th = None
            else:
              h_th = self.h_th_list[kk]
            
            if self.radius_array_name_list is not None:
              radius_array_name=radius_array_name_list[kk]
              if radius_array_name == "":
                radius_array_name=None
            else:
              radius_array_name=None
            
            poly = self.compute_radius(reader.GetOutput(),self.spacing_list[kk],self.feature_type_list[kk],radius_array_name,h_th)
            if self.use_field_data == False:
                poly.GetPointData().\
                    SetNormals(poly.GetPointData().\
                               GetArray(self.normal_map[self.feature_type_list[kk]]))
            else:
                poly.GetPointData().\
                    SetNormals(poly.GetFieldData().\
                               GetArray(self.normal_map[self.feature_type_list[kk]]))
        
            glypher=self.create_glyphs(poly)
            if len(self.color_list) <= kk:
                color=[]
            else:
                color=self.color_list[kk]
            if len(self.opacity_list) <= kk:
                opacity=1
            else:
                opacity=self.opacity_list[kk]
            
            if len(self.lut_list) <=kk:
                lut= None
            else:
                lut = self.create_lut(lut_list[kk])
            
            self.create_actor(glypher,color=color,opacity=opacity,lut=lut,color_by_array_name=self.color_by_array_name)
    
            if self.glyph_output is not None:
                tt=vtk.vtkTransform()
                tt.Identity()
                if self.coordinate_system == "RAS":
                    print ("Transforming to RAS")
                    tt.GetMatrix().SetElement(0,0,-1)
                    tt.GetMatrix().SetElement(1,1,-1)

                tf=vtk.vtkTransformPolyDataFilter()
                tf.SetTransform(tt)
                tf.SetInputData(glypher)
                tf.SetTransform(tt)
                tf.Update()
                writer=vtk.vtkPolyDataWriter()
                writer.SetInputData(tf.GetOutput())
                writer.SetFileName(self.glyph_output)
                writer.SetFileTypeToBinary()
                writer.Write()
            
        
        if len(self.lung)>0:
            reader=vtk.vtkPolyDataReader()
            reader.SetFileName(self.lung)
            reader.Update()
            tt=vtk.vtkTransform()
            tt.Identity()
            if self.coordinate_system == "RAS":
                tt.GetMatrix().SetElement(0,0,-1)
                tt.GetMatrix().SetElement(1,1,-1)
            
            tf=vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tt)
            tf.SetInputConnection(reader.GetOutputPort())
            tf.SetTransform(tt)
            color =[0.6,0.6,0.05]
            #color=[0.8,0.4,0.01]
            self.create_actor(tf,self.lung_opacity,color)

        if self.no_display == False:
          self.add_color_bar()
          self.render()
        
    def capture_window(self,obj, event):
        key = obj.GetKeySym()
        #print ("Key press "+key)
        if key == "s" and self.capture_prefix != "":
            ff = vtk.vtkWindowToImageFilter()
            sf = vtk.vtkPNGWriter()
            
            ff.SetInput(self.renWin)
            ff.SetMagnification(4)
            sf.SetInputData(ff.GetOutput())
            sf.SetFileName(self.capture_prefix+ "%03d.png" % self.capture_count )
            self.renWin.Render()
            ff.Modified()
            sf.Write()
            self.capture_count = 1+self.capture_count
        #Print current camera settings
        elif key == "p":
            camera=self.ren.GetActiveCamera()

            # Extract camera orientation details
            camera_position = camera.GetPosition()
            camera_focal_point = camera.GetFocalPoint()
            camera_view_up = camera.GetViewUp()
            # Print the camera orientation details
            print("Camera Position:", camera_position)
            print("Camera Focal Point:", camera_focal_point)
            print("Camera View-Up Vector:", camera_view_up)
        # Axial
        elif key == "a":
            self.set_camera_view("axial")
        # Coronal
        elif key == "c":
            self.set_camera_view("coronal")
        # Sagittal
        elif key == "g":
            self.set_camera_view("sagittal")
        #Flip the direction of the view
        elif key == "f":
            self.flip_camera()
    

if __name__ == "__main__":
    desc=" Visualization of particles vtk files"

    parser = ArgumentParser(description=desc)

    parser.add_argument("-i", help='Input particle files to render', dest="file_name")
    parser.add_argument("-s", help='Input spacing', dest="spacing")
    parser.add_argument("--feature", help='Feature type for each particle point. Options are: valley_line (or vessel), ridge_line (or airway), ridge_surface (or fissure) and valley_surface', \
                        dest="feature_type", default="vessel")
    parser.add_argument("--irad", help='Interparticle distance', dest="irad", \
                        default=1.2)
    parser.add_argument("--hth", help='Threshold on particle strength', dest="hth", default=None)
    parser.add_argument("--maxrad", help='Maximum radius to display', dest="max_rad", \
                        default=6.0)
    parser.add_argument("--minrad", help='Minimum  radius to display', dest="min_rad", \
                          default=0.5)
    parser.add_argument("--color", help='RGB color', dest="color_list", default=None)
    parser.add_argument("--opacity", help='Opacity values', dest="opacity_list", \
                        default=None)
    parser.add_argument("--lut", help='Look up table file list for each particle file (comma separated values with R,G,B,Alpha values)', \
                        dest="lut_list", default=None)
    parser.add_argument("-l", help='Lung mesh', dest="lung_filename", default=None)
    parser.add_argument("--useFieldData", help='Enable if particle features are stored in Field data instead of Point Data', dest="use_field_data", \
                        action="store_true", default=False)
    parser.add_argument("--glyphScale", help='Scaling factor for glyph', dest="glyph_scale_factor", \
                        default=1)
    parser.add_argument("--colorBy", help='Array name to color by', dest="color_by", \
                        default=None)
    parser.add_argument("--smooth", help='Enable glpyer smoothing to have a smoother transition between particles', dest="smooth_glyphs", \
                        action="store_true", default=False)
    parser.add_argument("--ras", help='Set output for RAS', dest="ras_coordinate_system", \
                        default=False,action="store_true")
    parser.add_argument("--glyphOutput", help='Output vtk with glpyh poly data', dest='glyph_output', \
                        default=None)
    parser.add_argument("--capturePrefix", help='Prefix filename to save screenshots. This options enables screen capture. Press the "s" key to capture a screenshot.', \
                      dest="capture_prefix", default=None)

    parser.add_argument("--radius_name", help='Array name with the radius information (optional).\
                                        If this is not provided the radius will be computed from the scale information.',
                                        dest='radius_array_name',metavar='<float>',default=None)
    parser.add_argument("--no-display", help='No display mode. Objects will be created but not render. It can be used for off-line saving of glyh vtk file',dest='nodisplay',action='store_true')
                                        
    options = parser.parse_args()

    translate_color = dict()
    translate_color['red'] = [1, 0.1, 0.1]
    translate_color['green'] = [0.1, 0.8, 0.1]
    translate_color['orange'] = [0.95, 0.5, 0.01]
    translate_color['blue'] = [0.1, 0.1, 0.9]
    translate_color['gold'] = [0.94509,0.8392,0.56862]
    translate_color['gray'] = [0.5,0.5,0.5]
    translate_color['cba'] = [0]

    file_list = [i for i in str.split(options.file_name,',')]
    use_field_data = options.use_field_data
    if options.spacing is not None:
      spacing_list = [float(i) for i in str.split(options.spacing,',')]
    
    if options.lung_filename == None:
        lung_filename=""
    else:
        lung_filename = options.lung_filename

    feature_type_list = [i for i in str.split(options.feature_type,',')]


    if options.opacity_list == None:
        opacity_list=[]
    else:
        opacity_list = [float(i) for i in str.split(options.opacity_list,',')]
                           
    if options.color_list == None:
        color_list=[]
    else:
        color_list = [translate_color[val] for val in str.split(options.color_list,',')]

    if options.hth == None:
        hth_list = []
    else:
        hth_list = [float(i) for i in str.split(options.hth,',')]

    if options.lut_list == None:
        lut_list=[]
    else:
        lut_list=[]
        for lut_file in str.split(options.lut_list,','):
          _df=pd.read_csv(lut_file)
          lut_list.append(_df.values())

    if options.radius_array_name == "" or options.radius_array_name is None:
        radius_array_name_list=None
    else:
        radius_array_name_list=[]
        radius_array_name_list = [str(i) for i in str.split(options.radius_array_name,',')]

    dv = DisplayParticles(file_list, spacing_list,feature_type_list,float(options.irad),hth_list, \
        'cylinder', float(options.glyph_scale_factor),float(options.max_rad),float(options.min_rad),use_field_data, opacity_list, color_list, lut_list,lung_filename,options.smooth_glyphs)
    if options.color_by is not None:
        dv.color_by_array_name=options.color_by
    if options.glyph_output is not None:
        dv.glyph_output=options.glyph_output
    if options.ras_coordinate_system:
        dv.coordinate_system="RAS"

    dv.no_display = options.nodisplay

    dv.radius_array_name_list=radius_array_name_list

    dv.capture_prefix = options.capture_prefix
    dv.execute()
