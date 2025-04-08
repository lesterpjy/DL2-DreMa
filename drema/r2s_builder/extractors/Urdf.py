import os
import shutil
import open3d as o3d
import numpy as np
from object2urdf import ObjectUrdfBuilder


class URDFBuilder:
    def __init__(self, output_path):

        self.output_path = output_path

        self.surface_output_path = os.path.join(self.output_path, "flat_surface")
        self.objects_output_path = os.path.join(self.output_path, "urdf")

        # if the path is not absolute, make it absolute
        if not os.path.isabs(self.objects_output_path):
            self.objects_output_path = os.path.abspath(self.objects_output_path)

        # create the output folders
        os.makedirs(self.surface_output_path, exist_ok=True)
        os.makedirs(self.objects_output_path, exist_ok=True)

        # move the _prototype files
        shutil.copyfile("assets/_prototype_flat_surface.urdf", os.path.join(self.surface_output_path, "_prototype_flat_surface.urdf"), follow_symlinks=True)
        shutil.copyfile("assets/_prototype.urdf", os.path.join(self.objects_output_path, "_prototype.urdf"), follow_symlinks=True)

        # create the builders
        self.surface_builder = ObjectUrdfBuilder(self.surface_output_path, urdf_prototype='_prototype_flat_surface.urdf')
        self.objects_builder = ObjectUrdfBuilder(self.objects_output_path, urdf_prototype='_prototype.urdf')


    def build_urdf_flat_surface(self, hull, center):

        # save mesh
        o3d.io.write_triangle_mesh(os.path.join(self.surface_output_path, "flat_surface_mesh.obj"), hull)

        # save the center
        np.save(os.path.join(self.surface_output_path, "position.npy"), center)

        # build the urdf
        self.surface_builder.build_urdf(filename=os.path.join(self.surface_output_path, "flat_surface_mesh.obj"),
                                        force_overwrite=True, decompose_concave=True, force_decompose=False,
                                        center='geometric')
        # find the generated file in the output folder that finishes with .urdf
        files = [f for f in os.listdir(self.surface_output_path) if os.path.isfile(os.path.join(self.surface_output_path, f))]
        name = [f for f in files if f.endswith(".urdf") and not "prototype" in f][0]
        print("Renaming: ", name, " to flat_surface_mesh.urdf")
        # rename the file
        os.rename(os.path.join(self.surface_output_path, name),
                  os.path.join(self.surface_output_path, "flat_surface_mesh.urdf"))


    def build_urdf_object(self, object, label):

        # create the output folder
        urdf_path = os.path.join(self.objects_output_path, label)
        os.makedirs(urdf_path, exist_ok=True)



        # translate mesh to origin
        vertices = np.array(object.vertices)
        center = np.mean(vertices, axis=0)
        norm_v = vertices - center
        object.vertices = o3d.utility.Vector3dVector(norm_v)

        # save mesh
        o3d.io.write_triangle_mesh(os.path.join(self.objects_output_path, label, "mesh" + label + ".obj"), object)

        # save the center
        np.save(os.path.join(urdf_path, "mesh_coordinates" + label + ".npy"), center)

        self.objects_builder.build_urdf(filename=os.path.join(urdf_path, "mesh" + label + ".obj"), force_overwrite=True,
                           decompose_concave=True, force_decompose=False, center='geometric')



