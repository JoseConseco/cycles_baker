#  (c) 2014 by Piotr Adamowicz (MadMinstrel)

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Cycles Baker",
    "author": "Bartosz Styperek",
    "version": (2, 0),
    "blender": (2, 80, 0),
    "location": "Npanel -> Tool shelf -> Baking (tab)",
    "description": "Addon for baking with Cycles.",
    "warning": "",
    "wiki_url": "",
    "category": "Object"}

import os
import bpy
import bgl
import bmesh
import aud
from bpy.app.handlers import persistent
from bpy.props import *
from mathutils import Matrix, Vector
from datetime import datetime, timedelta

# import sys
# dir = 'C:\\Users\\JoseConseco\\AppData\\Local\\Programs\\Python\\Python35\\Lib\\site-packages'
# if not dir in sys.path:
#     sys.path.append(dir )
# import ipdb


def draw_cage_callback(pairClass, context):
    frontRayDistanceMultiplier = pairClass.front_distance_modulator
    g_rv3d = context.region_data
    viewDir = g_rv3d.view_rotation * Vector((0.0, 0.0, 1.0))
    if not pairClass.front_distance_modulator_draw:
        return
    obj = bpy.data.objects[pairClass.lowpoly]
    objVector = Vector(obj.dimensions[:])
    objBBoxSize = objVector.length
    parentBakeJob = None
    for bj in bpy.data.scenes['Scene'].cycles_baker_settings.bake_job_queue:
        for pair in bj.bake_pairs_list:
            if pair == pairClass:
                parentBakeJob = bj
                break

    frontDistance = parentBakeJob.frontDistance
    if obj is None:
        return
    bgl.glEnable(bgl.GL_BLEND)
    if obj.type == 'MESH' and context.mode == 'OBJECT':
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        # get 2d screen vert coord from 3d view
        transformedVerts = []
        for vert in bm.verts:
            if pairClass.front_distance_modulator_draw:
                offset = vert.normal * frontRayDistanceMultiplier * frontDistance * objBBoxSize * 0.82  # 0,86 to match substance ray distance
            vector3d = obj.matrix_world @ (vert.co + offset)
            transformedVerts.append(vector3d)

        bgl.glEnable(bgl.GL_DEPTH_TEST)

        if pairClass.front_distance_modulator_draw:
            g_face_color = [0, 0.8, 0]
        else:
            g_face_color = [0.8, 0, 0]
        for face in bm.faces:
            cosine = viewDir.dot(face.normal) * 0.5 + 0.5
            bgl.glColor4f(g_face_color[0] * cosine, g_face_color[1] * cosine, g_face_color[2] * cosine, 0.5)
            bgl.glBegin(bgl.GL_POLYGON)
            for vert in face.verts:
                vertex = transformedVerts[vert.index]
                bgl.glVertex3f(vertex[0], vertex[1], vertex[2])
            bgl.glEnd()

        # restore opengl defaults
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        bgl.glDisable(bgl.GL_BLEND)
        bgl.glColor4f(0.0, 0.0, 0.0, 1.0)


bm_old = [None]


class CyclesBakerPreferences(bpy.types.AddonPreferences):
    bl_idname = 'cycles_baker'

    Info: bpy.props.StringProperty(name="Info", description="", default="")
    MAT_ID: bpy.props.StringProperty(name="Mat ID suffix", description="", default='id')
    AO: bpy.props.StringProperty(name="AO suffix", description="", default='ao')
    NORMAL: bpy.props.StringProperty(name="NORMAL suffix", description="", default="normal")
    HEIGHT: bpy.props.StringProperty(name="Thickness map suffix", description="", default="height")
    OPACITY: bpy.props.StringProperty(name="Opacity map suffix", description="", default="opacity")

    def draw(self, context):
        layout = self.layout
        layout.label(text=self.Info)
        layout.prop(self, "MAT_ID")
        layout.prop(self, "AO")
        layout.prop(self, "NORMAL")
        layout.prop(self, "HEIGHT")
        layout.prop(self, "OPACITY")


handleDrawRayDistance = []


class CyclesBakePair(bpy.types.PropertyGroup):
    def drawCage(self, draw):
        if draw:
            if handleDrawRayDistance:
                bpy.types.SpaceView3D.draw_handler_remove(handleDrawRayDistance[0], 'WINDOW')

            args = (self, bpy.context)  # u can pass arbitrary class as first param  Instead of (self, context)
            handleDrawRayDistance[:] = [bpy.types.SpaceView3D.draw_handler_add(draw_cage_callback, args, 'WINDOW', 'POST_VIEW')]
        else:
            if handleDrawRayDistance:
                bpy.types.SpaceView3D.draw_handler_remove(handleDrawRayDistance[0], 'WINDOW')
                handleDrawRayDistance[:] = []

    def DrawFrontRayDistance(self, context):
        draw = self.front_distance_modulator_draw
        self.drawCage(draw)

    activated: bpy.props.BoolProperty(
        name="Activated", description="Enable/Disable baking this pair of objects. Old bake result will be used if disabled", default=True)
    lowpoly: bpy.props.StringProperty(name="", description="Lowpoly mesh", default="")
    highpoly: bpy.props.StringProperty(name="", description="Highpoly mesh", default="")
    hp_obj_vs_group: bpy.props.EnumProperty(name="Object vs Group", description="", default="OBJ", items=[
                                            ('OBJ', '', 'Object', 'MESH_CUBE', 0), ('GRP', '', 'Group', 'GROUP', 1)])
    use_cage: bpy.props.BoolProperty(name="Use Cage", description="Use cage object", default=False)
    cage: bpy.props.StringProperty(name="", description="Cage mesh", default="")
    front_distance_modulator: bpy.props.FloatProperty(
        name="Front distance modulator", description="", default=1.0, min=0, max=10, subtype='FACTOR')
    front_distance_modulator_draw: bpy.props.BoolProperty(
        name="Draw Front distance", description="", default=False, update=DrawFrontRayDistance)
    no_materials: bpy.props.BoolProperty(name="No Materials", default=False)





# def getPasses(self,context):
#     # __import__('code').interact(local={k: v for ns in (globals(), locals()) for k, v in ns.items()})
#     items = []
#     jQueIndex = self.path_from_id()[-20] #works only for this kind of -path_from_id- 'cycles_baker_settings.bake_job_queue[0].bake_pass_list[1]'
#     CyclesBakeSettings = context.scene.cycles_baker_settings
#     bj = CyclesBakeSettings.bake_job_queue[int(jQueIndex)]
#     for passes in bj.bake_pass_list:
#         items.append((passes.pass_name,passes.pass_name,""))
#     return items

class CyclesBakePass(bpy.types.PropertyGroup):
    def upSuffix(self, context):
        addon_prefs = bpy.context.preferences.addons['cycles_baker'].preferences
        if self.pass_name == "AO":
            self.suffix = addon_prefs.AO
        if self.pass_name == "NORMAL":
            self.suffix = addon_prefs.NORMAL
        if self.pass_name == "MAT_ID":
            self.suffix = addon_prefs.MAT_ID
        if self.pass_name == "HEIGHT":
            self.suffix = addon_prefs.HEIGHT
        if self.pass_name == "OPACITY":
            self.suffix = addon_prefs.OPACITY

    activated: bpy.props.BoolProperty(name="Activated", default=True)

    pass_name: bpy.props.EnumProperty(name="Pass", default="NORMAL",
                                      items=(
                                           ("MAT_ID", "Material ID", ""),
                                           ("AO", "Ambient Occlusion", ""),
                                           ("NORMAL", "Normal", ""),
                                           ("HEIGHT", "Height", ""),
                                           ("OPACITY", "Opacity mask", ""),
                                      ), update=upSuffix)

    material_override: bpy.props.StringProperty(name="Material Override", description="", default="")
    ao_distance: bpy.props.FloatProperty(name="Distance", description="Maximum Occluder Distance", default=0.1, min=0.0, max=1.0)
    samples: bpy.props.IntProperty(name="Samples", description="", default=64, min=16, max=512)
    suffix: bpy.props.StringProperty(name="Suffix", description="", default="")  # addon_prefs.NORMAL

    bake_all_highpoly: bpy.props.BoolProperty(name="Highpoly", default=False)
    environment_obj_vs_group: bpy.props.EnumProperty(name="Object vs Group", description="", default="OBJ", items=[
                                                     ('OBJ', '', 'Object', 'MESH_CUBE', 0), ('GRP', '', 'Group', 'GROUP', 1)])
    environment_group: bpy.props.StringProperty(name="", description="Additional environment occluding object(or group)", default="")

    # nm_pass_index : bpy.props.EnumProperty(name = "Normal map pass", items = getPasses  )
    bit_depth: bpy.props.EnumProperty(name="Color Depth", description="", default="0",
                                      items=(("0", "8 bit(default)", ""),
                                             ("1", "16 bit", "")
                                             ))

    nm_space: bpy.props.EnumProperty(name="Normal map space", default="TANGENT",
                                     items=(("TANGENT", "Tangent Space", ""),
                                            ("OBJECT", "World Space", "")))
    nm_invert: bpy.props.EnumProperty(name="Invert green channel", default="POS_Y",
                                      items=(("POS_Y", "OpenGL", "Blender Compatible"),
                                             ("NEG_Y", "DirectX", "")))

    position_mode: bpy.props.EnumProperty(name="Mode", default="0",
                                          items=(("0", "All Axis", "bakes the position on the x,y, and z axis in the rgb channels"),
                                                 ("1", "One Axis", "bakes a single axis in a greyscale image")))

    position_mode_axis: bpy.props.EnumProperty(name="Axis", description="", default="1",
                                               items=(("0", "X", ""),
                                                      ("2", "Y", ""),
                                                      ("1", "Z (default)", "Default")
                                                      ))
    ray_distrib: bpy.props.EnumProperty(name="Ray distribution", description="", default="1",
                                        items=(("0", "Uniform", ""),
                                               ("1", "Cosine", "")
                                               ))

    def props(self):
        props = set()
        if self.pass_name == "AO":
            props = {"ao_distance", "samples", "environment_group", "ray_distrib"}
        if self.pass_name == "NORMAL":
            props = {"nm_space", "nm_invert", "bit_depth"}
        # if self.pass_name == "OPACITY":
        #     props = {'position_mode', 'OPACITY'}
        if self.pass_name == "HEIGHT":
            props = {'bit_depth'}

        return props

    def get_subst_designer_pass_name(self):  # convert pass name to SD bake command
        if self.pass_name == "MAT_ID":
            return "color-from-mesh"
        if self.pass_name == "AO":
            return "ambient-occlusion-from-mesh"
        if self.pass_name == "NORMAL":
            return "normal-from-mesh"
        if self.pass_name == "UV":
            return "uv-map"
        if self.pass_name == "HEIGHT":
            return "height-from-mesh"
        if self.pass_name == "OPACITY":
            return "opacity-mask-from-mesh"

    def get_filename(self, bj):
        name = bj.name
        if len(self.suffix) > 0:
            name += "_" + self.suffix
        return name



class CyclesBakeJob(bpy.types.PropertyGroup):

    def refreshRenderChange(self, context):
        for pairs in self.bake_pairs_list:
            pairs.activated = self.refreshRender

    activated: bpy.props.BoolProperty(name="Activated", description="Disable baking set of high-low pairs", default=True)
    refreshRender: bpy.props.BoolProperty(name="Refresh Render", description="Render on/off helper button ",
                                          default=True, update=refreshRenderChange)
    expand: bpy.props.BoolProperty(name="Expand", default=True)
    bakeResolution: bpy.props.EnumProperty(name="Resolution", default="1024",
                                           items=(("128", "128x128", ""),
                                                  ("256", "256x256", ""),
                                                  ("512", "512x512", ""),
                                                  ("1024", "1024x1024", ""),
                                                  ("2048", "2048x2048", ""),
                                                  ("4096", "4096x4096", ""))
                                           )
    antialiasing: bpy.props.EnumProperty(name="Anti-aliasing", description="Anti-aliasing", default="1",
                                         items=(("1", "None", ""),
                                                ("2", "2x", ""),
                                                ("4", "4x", "")))

    dilation: bpy.props.FloatProperty(name="Dilation", description="Width of the dilation post-process (in percent of image size). Default = 0.007.",
                                      default=0.007, min=0.00, max=0.01, subtype='PERCENTAGE')

    output: bpy.props.StringProperty(name='File path',
                                     description='The path of the output image.',
                                     default='//textures/',
                                     subtype='FILE_PATH')
    name: bpy.props.StringProperty(name='name', description="Output texture name", default='bake')

    relativeToBbox: bpy.props.BoolProperty(
        name="Relative to bbox", description="Interpret the max distances as a factor of the mesh bounding box.", default=True)
    frontDistance: bpy.props.FloatProperty(name="Max frontal distance",
                                           description="Max ray scan distance outside lowpoly", default=0.02, min=0.0, max=1.0, precision=3)

    bake_pairs_list: bpy.props.CollectionProperty(type=CyclesBakePair)
    bake_pass_list: bpy.props.CollectionProperty(type=CyclesBakePass)

    def get_filepath(self):
        outPath = bpy.path.abspath(self.output)
        return outPath  # [:-1]remove last \ - this is how sd works



class CyclesBakeSettings(bpy.types.PropertyGroup):
    bake_job_queue: bpy.props.CollectionProperty(type=CyclesBakeJob)


def convertCurveToGeo(curve, scene):
    if curve.type == 'CURVE' and (curve.data.bevel_depth != 0 or curve.data.bevel_object is not None):  # for converting curve to geo
        mesh = curve.to_mesh(scene, True, 'RENDER')
        ObjMeshFromCurve = bpy.data.objects.new('DupaBla', mesh)
        scene.objects.link(ObjMeshFromCurve)
        ObjMeshFromCurve.matrix_world = curve.matrix_world
        return ObjMeshFromCurve
    return None


class CB_OT_CyclesBakeOp(bpy.types.Operator):
    bl_idname = "cycles.bake"
    bl_label = "Cycles Bake"
    bl_options = {'REGISTER', 'UNDO'}

    job_index: bpy.props.IntProperty()
    pass_index: bpy.props.IntProperty()
    pair_index: bpy.props.IntProperty()
    bake_all: bpy.props.BoolProperty()
    bake_target: bpy.props.StringProperty()
    normalPassIndex = 99  # helper for curvature map - bake curv ony if curvPassIndex>normalPassIndex
    startTime = None
    baseMaterialList = []

    def create_temp_node(self):
        CyclesBakeSettings = bpy.context.scene.cycles_baker_settings
        pair = CyclesBakeSettings.bake_job_queue[self.job_index].bake_pairs_list[self.pair_index]
        # add an image node to the lowpoly model's material
        bake_mat = bpy.data.scenes["MD_TEMP"].objects[pair.lowpoly + "_MD_TMP"].active_material

        bake_mat.use_nodes = True
        if "MDtarget" not in bake_mat.node_tree.nodes:
            imgnode = bake_mat.node_tree.nodes.new(type="ShaderNodeTexImage")
            imgnode.image = bpy.data.images["MDtarget"]
            imgnode.name = 'MDtarget'
            imgnode.label = 'MDtarget'
        else:
            imgnode = bake_mat.node_tree.nodes['MDtarget']
            imgnode.image = bpy.data.images["MDtarget"]

        bake_mat.node_tree.nodes.active = imgnode

    def create_render_target(self):
        CyclesBakeSettings = bpy.context.scene.cycles_baker_settings
        bj = CyclesBakeSettings.bake_job_queue[self.job_index]
        aa = int(bj.antialiasing)
        bpy.ops.image.new(name="MDtarget", width=int(bj.bakeResolution)*aa,
                          height=int(bj.bakeResolution)*aa,
                          color=(0.0, 0.0, 0.0, 0.0), alpha=True, generated_type='BLANK', float=False)

    def pass_material_id_prep(self):
        for material in bpy.data.materials:
            if material not in self.baseMaterialList:
                material.use_nodes = True

                for node in material.node_tree.nodes:
                    if node.label == 'MDtarget':
                        continue
                    material.node_tree.nodes.remove(node)

                tree = material.node_tree

                tree.nodes.new(type="ShaderNodeBsdfDiffuse")
                tree.nodes.new(type="ShaderNodeOutputMaterial")
                output = tree.nodes["Diffuse BSDF"].outputs["BSDF"]
                input = tree.nodes["Material Output"].inputs["Surface"]
                tree.links.new(output, input)

                material.node_tree.nodes["Diffuse BSDF"].inputs["Color"].default_value = \
                    [material.diffuse_color[0], material.diffuse_color[1], material.diffuse_color[2], 1]

    def compo_nodes_mergePassImgs(self):
        CyclesBakeSettings = bpy.context.scene.cycles_baker_settings
        bj = CyclesBakeSettings.bake_job_queue[self.job_index]
        bakepass = bj.bake_pass_list[self.pass_index]
        # job = mds.bake_job_queue[self.job]
        if bakepass.activated:
            targetimage = bpy.data.images["MDtarget"]
            targetimage.scale(int(bj.bakeResolution), int(bj.bakeResolution))
            # bpy.ops.render.render(write_still=True, scene="MD_COMPO")
            targetimage.filepath_raw = bj.get_filepath() + bakepass.get_filename(bj) + ".png"  # blender needs slash at end
            targetimage.save()

            aa = int(bj.antialiasing)  # revert image size to original size for next bake
            targetimage.scale(int(bj.bakeResolution) * aa, int(bj.bakeResolution) * aa)

    def copyModifierParentsSetup(self, groupObjs, CloneObjPointer):  # same as GroupPro.py
        # copy modifier obj, parents etc from source do clones
        import copy
        for groupObj in groupObjs:  # search donors
            if groupObj.parent:  # for array
                if groupObj.parent.name not in CloneObjPointer.keys():  # what if parent /modifier obj is not inside group????
                    pass  # then probably parent is outside group. So just skip it
                else:
                    backupMatrixWorld = copy.deepcopy(CloneObjPointer[groupObj.name].matrix_world)
                    backupMatrixParentInv = copy.deepcopy(CloneObjPointer[groupObj.name].matrix_parent_inverse)
                    CloneObjPointer[groupObj.name].parent = CloneObjPointer[groupObj.parent.name]  # set parent to new child
                    # this is needed cos changing parent may zero MatrixParentInverted. So just restore it
                    CloneObjPointer[groupObj.name].matrix_parent_inverse = backupMatrixParentInv
                    CloneObjPointer[groupObj.name].matrix_world = backupMatrixWorld
            for index, modifier in enumerate(groupObj.modifiers):
                if hasattr(modifier, 'object'):
                    if modifier.object and modifier.object.name in CloneObjPointer.keys():
                        # set modifier to target
                        CloneObjPointer[groupObj.name].modifiers[index].object = CloneObjPointer[modifier.object.name]
                if hasattr(modifier, 'offset_object'):  # for array
                    if modifier.offset_object and modifier.offset_object.name in CloneObjPointer.keys():
                        # set modifier to target
                        CloneObjPointer[groupObj.name].modifiers[index].offset_object = CloneObjPointer[modifier.offset_object.name]
                if hasattr(modifier, 'origin'):  # for simple deform
                    if modifier.origin and modifier.origin.name in CloneObjPointer.keys():
                        # set modifier to target
                        CloneObjPointer[groupObj.name].modifiers[index].origin = CloneObjPointer[modifier.origin.name]

    def make_duplicates_real(self, empty, oldMatrix, highPolyGroupName, depth=0):
        new_matrix = oldMatrix @ empty.matrix_world  # to store combination of parent collections matrices
        CloneObjPointer = {}
        for dupObj in empty.instance_collection.objects:
            if dupObj.type == 'EMPTY' and dupObj.instance_collection:
                self.make_duplicates_real(dupObj, new_matrix, highPolyGroupName, depth + 1)
            else:
                copyGroupObj = dupObj.copy()
                copyGroupObj.name += "_MD_TMP"
                CloneObjPointer[dupObj.name] = copyGroupObj
                if copyGroupObj.type == 'CURVE':  # try clone and convert curve to mesh.
                    curveMeshClone = convertCurveToGeo(copyGroupObj, bpy.data.scenes['MD_TEMP'])
                    if curveMeshClone is not None:
                        bpy.data.collections[highPolyGroupName].objects.link(curveMeshClone)
                        curveMeshClone.matrix_world = new_matrix @ curveMeshClone.matrix_world
                    # copyGroupObj.select_set( True

                bpy.data.scenes["MD_TEMP"].collection.objects.link(copyGroupObj)
                if not copyGroupObj.hide_render:  # to not add hidden render obj's to export
                    bpy.data.collections[highPolyGroupName].objects.link(copyGroupObj)
                copyGroupObj.matrix_world = new_matrix @ copyGroupObj.matrix_world
        self.copyModifierParentsSetup(empty.instance_collection.objects, CloneObjPointer)
        if depth == 0:  # remove only first empty, cos deeper empties are still used in other groupInstances on scene.
            # bpy.data.scenes["MD_TEMP"].collection.objects.unlink(empty)  # remove empty from baking
            empty.user_clear()
            bpy.data.objects.remove(empty)

    def scene_copy(self):

        CyclesBakeSettings = bpy.context.scene.cycles_baker_settings
        bj = CyclesBakeSettings.bake_job_queue[self.job_index]

        # store the original names of things in the scene so we can easily identify them later
        for object in bpy.context.scene.objects:
            object.sd_orig_name = object.name

        for group in bpy.data.collections:
            group.sd_orig_name = group.name
        for world in bpy.data.worlds:
            world.sd_orig_name = world.name
        for material in bpy.data.materials:
            material.sd_orig_name = material.name
            self.baseMaterialList.append(material)

        # duplicate the scene
        bpy.ops.scene.new(type='FULL_COPY')
        bpy.context.scene.name = "MD_TEMP"
        # tag the copied object names with _MD_TMP
        for object in bpy.data.scenes["MD_TEMP"].objects:
            object.name = object.sd_orig_name + "_MD_TMP"
        for world in bpy.data.worlds:
            if world.name != world.sd_orig_name:
                world.name = "MD_TEMP"
        for material in bpy.data.materials:
            print('material name is: ' + material.name)
            if material.use_nodes:
                material.use_nodes = False
            if material.name != material.sd_orig_name:
                material.name = material.sd_orig_name + "_MD_TMP"
        # error before here
        for group in bpy.data.collections:
            if group.name != group.sd_orig_name:
                group.name = group.sd_orig_name + "_MD_TMP"
        for object in bpy.data.scenes["MD_TEMP"].objects:
            if object.parent:  # set parent and modifiers obj to temp objs
                if bpy.data.objects.get(object.parent.name + "_MD_TMP") is not None:
                    object.parent = bpy.data.objects[object.parent.name + "_MD_TMP"]

        bpy.ops.object.select_all(action='DESELECT')

        for pair in bj.bake_pairs_list:
            if pair.activated:
                if pair.hp_obj_vs_group == "OBJ":  # create group for hipoly
                    bpy.data.collections.new(pair.highpoly + "_MD_TMP")

                # link obj's from group pro to hipoly group if obj is member of hipoly bake group
                if pair.hp_obj_vs_group == "GRP":
                    # search for empties in hipoly group and convert them to geo
                    for obj in bpy.data.collections[pair.highpoly + "_MD_TMP"].objects:
                        if obj.type == 'CURVE':
                            ObjMeshFromCurve = convertCurveToGeo(obj, bpy.data.scenes['MD_TEMP'])
                            if ObjMeshFromCurve is not None:
                                bpy.data.collections[pair.highpoly + "_MD_TMP"].objects.link(ObjMeshFromCurve)
                        if obj.type == 'EMPTY' and obj.instance_collection:
                            self.make_duplicates_real(obj, Matrix.Identity(4), pair.highpoly +
                                                      "_MD_TMP")  # create and add obj to hipolyGroupName
                else:
                    hipolyTemp = bpy.data.scenes["MD_TEMP"].objects[pair.highpoly + "_MD_TMP"]
                    if hipolyTemp.type == 'EMPTY' and hipolyTemp.instance_collection:
                        self.make_duplicates_real(hipolyTemp, Matrix.Identity(4), pair.highpoly +
                                                  "_MD_TMP")  # TODO: whant if there is curve in group?
                    else:
                        if hipolyTemp.type == 'CURVE':  # try clone, convert to mesh, and add to highpoly if possible
                            ObjMeshFromCurve = convertCurveToGeo(hipolyTemp, bpy.data.scenes['MD_TEMP'])
                            if ObjMeshFromCurve is not None:
                                bpy.data.collections[pair.highpoly + "_MD_TMP"].objects.link(ObjMeshFromCurve)
                        isAlreadyInHP_Group = False
                        for objGroup in bpy.data.scenes["MD_TEMP"].objects[pair.highpoly + "_MD_TMP"].users_collection:
                            if objGroup == bpy.data.collections[pair.highpoly + "_MD_TMP"]:
                                isAlreadyInHP_Group = True
                                break
                        if not isAlreadyInHP_Group:
                            bpy.data.collections[pair.highpoly +
                                            "_MD_TMP"].objects.link(bpy.data.scenes["MD_TEMP"].objects[pair.highpoly + "_MD_TMP"])

    def select_hi_low(self):
        CyclesBakeSettings = bpy.context.scene.cycles_baker_settings
        bj = CyclesBakeSettings.bake_job_queue[self.job_index]

        pair = bj.bake_pairs_list[self.pair_index]
        bpy.ops.object.select_all(action="DESELECT")
        if pair.activated == True:
            # enviro group export
            for obj in bpy.data.scenes["MD_TEMP"].objects:
                obj.hide_render = True
            # make selections, ensure visibility
            enviroGroupName = ''
            # bpy.data.scenes["MD_TEMP"].layers[0] = True
            bpy.ops.object.select_all(action='DESELECT')
            for bakepass in bj.bake_pass_list:
                if bakepass.environment_group != "":  # bake enviro objects too
                    enviroGroupName = bakepass.environment_group
                    if bakepass.environment_obj_vs_group == "GRP":
                        for obj in bpy.data.collections[bakepass.environment_group + "_MD_TMP"].objects:
                            obj.hide_render = False
                            obj.select_set(True)
                            # obj.layers[0] = True
                    else:
                        EnviroObj = bpy.data.scenes["MD_TEMP"].objects[bakepass.environment_group + "_MD_TMP"]
                        EnviroObj.hide_render = False
                        EnviroObj.select_set( True)
                        # EnviroObj.layers[0] = True

            print("selected  enviro group " + pair.lowpoly)

            # highpoly export
            print("exporting pair " + pair.lowpoly)
            # bpy.data.scenes["MD_TEMP"].layers[0] = True

            # make selections, ensure visibility
            # bpy.ops.object.select_all(action='DESELECT')
            if pair.highpoly != "":
                for obj in bpy.data.collections[pair.highpoly + "_MD_TMP"].objects:
                    obj.hide_render = False
                    # obj.layers[0] = True
                    obj.select_set( True)

            # Lowpoly export
            print("exporting pair " + pair.highpoly)
            # lowpoly visibility
            bpy.data.scenes["MD_TEMP"].objects[pair.lowpoly + "_MD_TMP"].hide_viewport = False
            bpy.data.scenes["MD_TEMP"].objects[pair.lowpoly + "_MD_TMP"].hide_select = False
            bpy.data.scenes["MD_TEMP"].objects[pair.lowpoly + "_MD_TMP"].hide_render = False
            # bpy.data.scenes["MD_TEMP"].objects[pair.lowpoly + "_MD_TMP"].layers[0] = True
            bpy.data.scenes["MD_TEMP"].objects[pair.lowpoly + "_MD_TMP"].select_set( True)
            # bpy.data.scenes["MD_TEMP"].objects.active = bpy.data.scenes["MD_TEMP"].objects[pair.lowpoly + "_MD_TMP"]
            bpy.data.scenes['MD_TEMP'].view_layers[0].objects.active = bpy.data.scenes["MD_TEMP"].objects[pair.lowpoly + "_MD_TMP"]

            # cage visibility
            if pair.use_cage and pair.cage != "":
                bpy.ops.object.select_all(action='DESELECT')
                bpy.data.scenes["MD_TEMP"].objects[pair.cage + "_MD_TMP"].hide_viewport = False
                bpy.data.scenes["MD_TEMP"].objects[pair.cage + "_MD_TMP"].hide_select = False
                bpy.data.scenes["MD_TEMP"].objects[pair.cage + "_MD_TMP"].hide_render = False
                # bpy.data.scenes["MD_TEMP"].objects[pair.cage + "_MD_TMP"].layers[0] = True
                bpy.data.scenes["MD_TEMP"].objects[pair.cage + "_MD_TMP"].select_set( True)

    def bake_pair_pass(self):
        CyclesBakeSettings = bpy.context.scene.cycles_baker_settings
        bakepass = CyclesBakeSettings.bake_job_queue[self.job_index].bake_pass_list[self.pass_index]
        bj = CyclesBakeSettings.bake_job_queue[self.job_index]
        pair = bj.bake_pairs_list[self.pair_index]
        aa = int(bj.antialiasing)
        # ipdb.set_trace()
        if pair.activated == True:
            self.create_temp_node()
            self.startTime = datetime.now()  # time debug
            dilation = int(bj.dilation * int(bj.bakeResolution))
            # common params first
            if bakepass.pass_name == "MAT_ID":
                self.pass_material_id_prep()
            bpy.data.scenes["MD_TEMP"].render.engine = "CYCLES"
            # bpy.data.scenes["MD_TEMP"].cycles.bake_type = bakepass.pass_name
            if bakepass.pass_name == "AO":
                bpy.data.scenes["MD_TEMP"].cycles.samples = bakepass.samples
                bpy.data.worlds["MD_TEMP"].light_settings.distance = bakepass.ao_distance
            clear = True
            if self.pair_index > 0:
                clear = False

            front = sorted([0, bj.frontDistance * pair.front_distance_modulator, 1])[1]  # clamp to 0-1 range trick
            pass_name = bakepass.pass_name
            passFilter = {'NONE'}
            if pass_name == "MAT_ID":
                pass_name = 'DIFFUSE'
                passFilter = {'COLOR'}

            bpy.ops.object.bake(type=pass_name, filepath="", pass_filter=passFilter,
                                width=int(bj.bakeResolution)*aa, height=int(bj.bakeResolution)*aa, margin=dilation,
                                use_selected_to_active=True, cage_extrusion=front, cage_object=pair.cage,
                                normal_space=bakepass.nm_space,
                                normal_r="POS_X", normal_g=bakepass.nm_invert, normal_b='POS_Z',
                                save_mode='INTERNAL', use_clear=clear, use_cage=pair.use_cage,
                                use_split_materials=False, use_automatic_name=False)
            # self.report({'INFO'},command)

            print("Baking set " + pair.lowpoly + " " + bakepass.pass_name + "  time: " + str(datetime.now() - self.startTime))

    def remove_object(self, obj):
        if bpy.data.objects[obj.name]:
            if obj.type == "MESH":
                if obj.name in bpy.data.scenes["MD_TEMP"].collection.objects.keys():
                    bpy.data.scenes["MD_TEMP"].collection.objects.unlink(obj) #TODO: remake this to remove from master cool??
                mesh_to_remove = obj.data
                obj.user_clear()
                bpy.data.objects.remove(obj)
                if mesh_to_remove.users == 0:  # if multi user skip removing mesh
                    bpy.data.meshes.remove(mesh_to_remove)
            else:
                if obj.name in bpy.data.scenes["MD_TEMP"].collection.objects.keys():
                    bpy.data.scenes["MD_TEMP"].collection.objects.unlink(obj)
                obj.user_clear()
                bpy.data.objects.remove(obj)

    def cleanup(self):

        for obj in bpy.data.scenes["MD_TEMP"].objects:
            self.remove_object(obj)

        for material in bpy.data.materials:
            if material.name.endswith("_MD_TMP"):
                bpy.data.materials.remove(material, do_unlink=True)

        for group in bpy.data.collections:
            if group.name.endswith("_MD_TMP"):
                bpy.data.collections.remove(group, do_unlink=True)

        bpy.ops.scene.delete()

    # empty mat search function
    def is_empty_mat(self, context):
        CyclesBakeSettings = bpy.context.scene.cycles_baker_settings
        for bj in CyclesBakeSettings.bake_job_queue:
            for pair in bj.bake_pairs_list:
                if pair.activated == True:
                    if pair.highpoly != "":
                        if pair.hp_obj_vs_group == "GRP":
                            for obj in bpy.data.collections[pair.highpoly].objects:
                                if obj.type == 'EMPTY' and obj.instance_collection:
                                    continue
                                if obj.type == "MESH" and (len(obj.material_slots) == 0 or obj.material_slots[0].material is None):
                                    self.report({'INFO'}, 'Object: ' + obj.name + ' has no Material!')
                                    return True
                        else:
                            try:
                                hipolyObj = bpy.data.objects[pair.highpoly]
                            except:
                                print("No highpoly " + pair.highpoly + " object on scene found! Cancelling")
                                pair.activated = False
                                continue
                            if hipolyObj.type == 'EMPTY' and hipolyObj.instance_collection:
                                # return False  #prevent detecting empty mat on duplifaces
                                emptyMatInGroup = self.checkEmptyMaterialForGroup(hipolyObj, context)
                                if emptyMatInGroup:
                                    self.MaterialCheckedGroupNamesList.clear()
                                    return emptyMatInGroup
                            # non empty objs
                            elif hipolyObj.type == "MESH" and (len(hipolyObj.material_slots) == 0 or hipolyObj.material_slots[0].material is None):
                                self.report({'INFO'}, 'Object: ' + hipolyObj.name + ' has no Material!')
                                return True
                    else:  # if highpoly empty
                        print("No highpoly defined. Disabling pair")
                        pair.activated = False
                        continue
                    if pair.lowpoly == "":  # if lowpoly empty
                        print("No highpoly defined. Disabling pair")
                        pair.activated = False
                        continue
                    try:
                        bpy.data.objects[pair.lowpoly]
                    except:
                        print("No lowpoly " + pair.lowpoly + " object on scene found! Disabling pair")
                        pair.activated = False
                        continue

        self.MaterialCheckedGroupNamesList.clear()
        return False

    MaterialCheckedGroupNamesList = []

    def checkEmptyMaterialForGroup(self, empty, context):
        if empty.instance_collection.name in self.MaterialCheckedGroupNamesList:
            print(empty.instance_collection.name + " was already checked for empty mat. Skipping!")
            return False
        for obj in bpy.data.collections[empty.instance_collection.name].objects:
            if obj.instance_collection and obj.type == 'EMPTY':
                return self.checkEmptyMaterialForGroup(obj, context)
            elif obj.type == "MESH" and (len(obj.material_slots) == 0 or obj.material_slots[0].material is None) and not obj.hide_render:
                self.report({'INFO'}, 'Object: ' + obj.name + ' has no Material!')
                self.MaterialCheckedGroupNamesList.append(empty.instance_collection.name)  # add to check list if there was obj with empty mat
                return True
        self.MaterialCheckedGroupNamesList.append(empty.instance_collection.name)  # or no empty mat in group
        return False

    def execute(self, context):
        TotalTime = datetime.now()
        CyclesBakeSettings = context.scene.cycles_baker_settings
        if self.is_empty_mat(context):
            return {'CANCELLED'}

        for i_job, bj in enumerate(CyclesBakeSettings.bake_job_queue):
            if bj.activated == True:
                self.job_index = i_job
                self.create_render_target()
                # ensure save path exists
                if not os.path.exists(bpy.path.abspath(bj.output)):
                    os.makedirs(bpy.path.abspath(bj.output))

                for pair in bj.bake_pairs_list:  # disable hipoly lowpoly pairs that are not definied
                    if pair.lowpoly == "" or pair.lowpoly not in bpy.data.objects.keys():
                        self.report({'INFO'}, 'Lowpoly not found ' + pair.lowpoly)
                        pair.activated = False
                    if pair.lowpoly == "" or pair.highpoly not in bpy.data.objects.keys():
                        self.report({'INFO'}, 'highpoly not found ' + pair.highpoly)
                        pair.activated = False

                import sys
                # try:
                self.scene_copy()  # we export temp scene copy
                # except: # to prevent addon crash while temp_scene is created, delete tems scene after exeption
                #     e = sys.exc_info()[0]
                #     self.report({'ERROR'},"Error while exporting scene: "+str(e)+"\n") #or 'Error' ?
                #     print( "Error while exporting scene: "+str(e)+"\n" )
                #     self.cleanup() #delete scene
                #     return
                comp_pass_sum_time = timedelta(microseconds=-1)
                for i_pass, bakepass in enumerate(bj.bake_pass_list):
                    if bakepass.activated == True:
                        self.pass_index = i_pass
                        for i_pair, pair in enumerate(bj.bake_pairs_list):
                            self.pair_index = i_pair
                            self.select_hi_low()
                            self.bake_pair_pass()
                            # self.cleanup_render_target()
                        if len(bj.bake_pass_list) > 0 and len(bj.bake_pairs_list) > 0:
                            comp_passes_sub = datetime.now()
                            self.compo_nodes_mergePassImgs()
                            comp_pass_sum_time += datetime.now() - comp_passes_sub
                self.cleanup()  # delete scene
        # merge images from pass into one

        bpy.data.images["MDtarget"].user_clear()
        bpy.data.images.remove(bpy.data.images["MDtarget"])
        self.playFinishSound()
        print("Cycles Total baking time: " + str((datetime.now() - TotalTime).seconds))
        print("Cycles Comping pases time: " + str(comp_pass_sum_time.seconds))

        return {'FINISHED'}

    @staticmethod
    def playFinishSound():
        return
        script_file = os.path.realpath(__file__)
        directory = os.path.dirname(script_file)
        device = aud.device()
        sound = aud.Factory.file(directory + "\\finished.mp3") #Broken in 2.8
        handle = device.play(sound)
        # omit this to play full sound file
        # sleep(1)
        handle.stop()
        return


class CB_PT_SDPanel(bpy.types.Panel):
    bl_label = "Cycles Baking Tool"
    bl_idname = "cycles_bake"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cycles Baking"

    # @classmethod
    # def poll(cls, context):
    #     return bpy.context.scene.render.engine == "CYCLES"

    def draw(self, context):
        layout = self.layout
        CyclesBakeSettings = context.scene.cycles_baker_settings

        row = layout.row(align=True)
        row.alignment = 'EXPAND'
        row.operator("cycles.bake", text='Bake', icon="SCENE")

        row = layout.row(align=True)
        row.alignment = 'EXPAND'
        row.separator()

        for job_i, bj in enumerate(CyclesBakeSettings.bake_job_queue):

            row = layout.row(align=True)
            row.alignment = 'EXPAND'

            if bj.expand == False:
                row.prop(bj, "expand", icon="TRIA_RIGHT", icon_only=True, text=bj.name, emboss=False)

                if bj.activated:
                    row.prop(bj, "activated", icon_only=True, icon="COLOR_GREEN", emboss=False)
                else:
                    row.prop(bj, "activated", icon_only=True, icon="COLOR_RED", emboss=False)

                oper = row.operator("cyclesbaker.texture_preview", text="", icon="TEXTURE")
                oper.bj_i = job_i
                rem = row.operator("cyclesbake.rem_job", text="", icon="X")
                rem.job_index = job_i
            else:
                row.prop(bj, "expand", icon="TRIA_DOWN", icon_only=True, text=bj.name, emboss=False)

                if bj.activated:
                    row.prop(bj, "activated", icon_only=True, icon="COLOR_GREEN", emboss=False)
                else:
                    row.prop(bj, "activated", icon_only=True, icon="COLOR_RED", emboss=False)

                oper = row.operator("cyclesbaker.texture_preview", text="", icon="TEXTURE")
                oper.bj_i = job_i
                rem = row.operator("cyclesbake.rem_job", text="", icon="X")
                rem.job_index = job_i

                row = layout.row(align=True)
                row.alignment = 'EXPAND'
                row.prop(bj, 'bakeResolution', text="Resolution")

                row = layout.row(align=True)
                row.prop(bj, 'antialiasing', text="AA")

                row = layout.row(align=True)
                row.prop(bj, 'frontDistance', text="Front Distance")
                row.prop(bj, 'relativeToBbox', text="", icon="GRID")

                row = layout.row(align=True)
                row.alignment = 'EXPAND'
                row.prop(bj, 'dilation', text="Dilation")

                row = layout.row(align=True)
                row.alignment = 'EXPAND'
                row.prop(bj, 'output', text="Path")

                row = layout.row(align=True)
                row.alignment = 'EXPAND'
                row.prop(bj, 'name', text="Name")

                row = layout.row(align=True)
                row.alignment = 'EXPAND'
                masterBox = layout.box()
                row = masterBox.row(align=True)
                row.label(text='Helpers')
                row.prop(bj, "refreshRender", icon_only=True, icon="RESTRICT_RENDER_OFF", emboss=False)
                for pair_i, pair in enumerate(bj.bake_pairs_list):
                    row = layout.column(align=True).row(align=True)
                    row.alignment = 'EXPAND'
                    box = row.box().column(align=True)

                    subrow = box.row(align=True)
                    subrow.prop_search(pair, "lowpoly", bpy.context.scene, "objects")
                    oper = subrow.operator("cyclesbake.objectpicker", text="", icon="EYEDROPPER")
                    oper.bj_i = job_i
                    oper.pair_i = pair_i
                    oper.gr_obj = "object"
                    oper.prop = "lowpoly"

                    subrow = box.row(align=True)
                    subrow.prop(pair, 'hp_obj_vs_group', expand=True)
                    if pair.hp_obj_vs_group == 'OBJ':
                        subrow.prop_search(pair, "highpoly", bpy.context.scene, "objects")
                        oper = subrow.operator("cyclesbake.objectpicker", text="", icon="EYEDROPPER")
                        oper.bj_i = job_i
                        oper.pair_i = pair_i
                        oper.gr_obj = "object"
                        oper.prop = "highpoly"
                    else:
                        subrow.prop_search(pair, "highpoly", bpy.data, "collections")
                        oper = subrow.operator("cyclesbake.objectpicker", text="", icon="EYEDROPPER")
                        oper.bj_i = job_i
                        oper.pair_i = pair_i
                        oper.gr_obj = "group"
                        oper.prop = "highpoly"
                    subrow = box.row(align=True)

                    subrow.prop(pair, 'use_cage', icon_only=True, icon="OUTLINER_OB_LATTICE")
                    if not pair.use_cage:
                        subrow.prop(pair, 'front_distance_modulator', expand=True)
                        subrow.prop(pair, 'front_distance_modulator_draw', icon='FORWARD', icon_only=True, expand=True)
                    else:
                        subrow.prop_search(pair, "cage", bpy.context.scene, "objects")
                        oper = subrow.operator("cyclesbake.cage_maker", text="", icon="OBJECT_DATAMODE")
                        oper.lowpoly = pair.lowpoly
                        oper.bj_i = job_i
                        oper.pair_i = pair_i
                        oper = subrow.operator("cyclesbake.objectpicker", text="", icon="EYEDROPPER")
                        oper.bj_i = job_i
                        oper.pair_i = pair_i
                        oper.gr_obj = "object"
                        oper.prop = "cage"

                    col = row.column()
                    row = col.row()
                    rem = row.operator("cyclesbake.rem_pair", text="", icon="X")
                    rem.pair_index = pair_i
                    rem.job_index = job_i

                    row = col.row()
                    if pair.activated:
                        row.prop(pair, "activated", icon_only=True, icon="RESTRICT_RENDER_OFF", emboss=False)
                    else:
                        row.prop(pair, "activated", icon_only=True, icon="RESTRICT_RENDER_ON", emboss=False)
                    row = col.row()

                row = layout.row(align=True)
                row.alignment = 'EXPAND'
                addpair = row.operator("cyclesbake.add_pair", icon="DISCLOSURE_TRI_RIGHT")
                addpair.job_index = job_i

                for pass_i, bakepass in enumerate(bj.bake_pass_list):
                    row = layout.row(align=True)
                    row.alignment = 'EXPAND'
                    box = row.box().column(align=True)

                    # box = layout.box().column(align=True)
                    subrow = box.row(align=True)
                    subrow.alignment = 'EXPAND'
                    # subrow.label(text=bj.get_filepath())

                    # rem = row.operator("cyclesbake.rem_pass", text = "", icon = "X")
                    # rem.pass_index = pass_i
                    # rem.job_index = job_i

                    subrow = box.row(align=True)
                    subrow.alignment = 'EXPAND'
                    subrow.prop(bakepass, 'pass_name')

                    subrow = box.row(align=True)
                    subrow.alignment = 'EXPAND'
                    subrow.prop(bakepass, 'suffix')

                    if len(bakepass.props()) > 0:
                        subrow = box.row(align=True)
                        subrow.alignment = 'EXPAND'
                        subrow.separator()

                        if "ray_distrib" in bakepass.props():
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'ray_distrib', text="Ray distribution")

                        if "ao_distance" in bakepass.props():
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'ao_distance', text="Maximum Occluder Distance")

                        if "nm_space" in bakepass.props():
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'nm_space', text="Type")

                        if "position_mode" in bakepass.props():
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'position_mode', text="Mode")

                        if "position_mode_axis" in bakepass.props() and bakepass.position_mode == '1':
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'position_mode_axis', text="Axis")

                        if "nm_invert" in bakepass.props():
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'nm_invert', text="Flip G")

                        if "bit_depth" in bakepass.props():
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'bit_depth', text="Bit Depth")

                        if "samples" in bakepass.props():
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'samples', text="Samples")

                        if "environment_group" in bakepass.props():
                            subrow = box.row(align=True)
                            subrow.alignment = 'EXPAND'
                            subrow.prop(bakepass, 'environment_obj_vs_group', expand=True)
                            if bakepass.environment_obj_vs_group == 'OBJ':
                                subrow.prop_search(bakepass, "environment_group", bpy.context.scene, "objects")
                            else:
                                subrow.prop_search(bakepass, "environment_group", bpy.data, "collections")

                    col = row.column()
                    row = col.row()
                    rem = row.operator("cyclesbake.rem_pass", text="", icon="X")
                    rem.pass_index = pass_i
                    rem.job_index = job_i

                    row = col.row()
                    if bakepass.activated:
                        row.prop(bakepass, "activated", icon_only=True, icon="RESTRICT_RENDER_OFF", emboss=False)
                    else:
                        row.prop(bakepass, "activated", icon_only=True, icon="RESTRICT_RENDER_ON", emboss=False)

                row = layout.row(align=True)
                row.alignment = 'EXPAND'
                addpass = row.operator("cyclesbake.add_pass", icon="DISCLOSURE_TRI_RIGHT")
                addpass.job_index = job_i

                row = layout.row(align=True)
                row.alignment = 'EXPAND'
                row.separator()

        row = layout.row(align=True)
        row.alignment = 'EXPAND'
        row.operator("cyclesbake.add_job", icon="ADD")


class CB_OT_SDAddPairOp(bpy.types.Operator):
    '''add pair'''

    bl_idname = "cyclesbake.add_pair"
    bl_label = "Add Pair"

    job_index: bpy.props.IntProperty()

    def execute(self, context):
        scene_name = bpy.context.scene.name
        pair = bpy.data.scenes[scene_name].cycles_baker_settings.bake_job_queue[self.job_index].bake_pairs_list.add()
        return {'FINISHED'}


class CB_OT_SDRemPairOp(bpy.types.Operator):
    '''delete pair'''

    bl_idname = "cyclesbake.rem_pair"
    bl_label = "Remove Pair"

    pair_index: bpy.props.IntProperty()
    job_index: bpy.props.IntProperty()

    def execute(self, context):
        scene_name = bpy.context.scene.name
        bpy.data.scenes[scene_name].cycles_baker_settings.bake_job_queue[self.job_index].bake_pairs_list.remove(self.pair_index)

        return {'FINISHED'}


class CB_OT_SDAddPassOp(bpy.types.Operator):
    bl_idname = "cyclesbake.add_pass"
    bl_label = "Add Pass"

    job_index: bpy.props.IntProperty()

    def execute(self, context):
        addonPref = bpy.context.preferences.addons['cycles_baker'].preferences
        newpass = context.scene.cycles_baker_settings.bake_job_queue[self.job_index].bake_pass_list.add()
        newpass.suffix = addonPref.NORMAL  # cos normal seems be default pass when added
        return {'FINISHED'}


class CB_OT_SDPassOp(bpy.types.Operator):
    bl_idname = "cyclesbake.rem_pass"
    bl_label = "Remove Pass"

    pass_index: bpy.props.IntProperty()
    job_index: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.cycles_baker_settings.bake_job_queue[self.job_index].bake_pass_list.remove(self.pass_index)
        return {'FINISHED'}


class CB_OT_SDAddJobOp(bpy.types.Operator):
    bl_idname = "cyclesbake.add_job"
    bl_label = "Add Bake Job"

    def execute(self, context):
        context.scene.cycles_baker_settings.bake_job_queue.add()
        return {'FINISHED'}


class CB_OT_SDRemJobOp(bpy.types.Operator):
    bl_idname = "cyclesbake.rem_job"
    bl_label = "Remove Bake Job"

    job_index: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.cycles_baker_settings.bake_job_queue.remove(self.job_index)
        return {'FINISHED'}


class CB_OT_ObjectPicker(bpy.types.Operator):
    bl_idname = "cyclesbake.objectpicker"
    bl_label = "Pick Obj"
    bj_i: bpy.props.IntProperty()
    pair_i: bpy.props.IntProperty()
    prop: bpy.props.StringProperty()
    gr_obj: bpy.props.StringProperty()

    def execute(self, context):
        if context.active_object and context.active_object.select_get():
            if self.gr_obj == "group" and self.prop == "highpoly":
                context.scene.cycles_baker_settings.bake_job_queue[self.bj_i].bake_pairs_list[self.pair_i][self.prop] = context.active_object.users_collection[0].name
            else:
                context.scene.cycles_baker_settings.bake_job_queue[self.bj_i].bake_pairs_list[self.pair_i][self.prop] = context.active_object.name
        return {'FINISHED'}


class CB_OT_CageMaker(bpy.types.Operator):
    bl_idname = "cyclesbake.cage_maker"
    bl_label = "Create Cage"
    bl_description = "Create new Cage object"

    bj_i: bpy.props.IntProperty()
    pair_i: bpy.props.IntProperty()
    lowpoly: bpy.props.StringProperty()

    def execute(self, context):
        cageName = 'cage_' + self.lowpoly

        if bpy.data.objects[self.lowpoly] is not None:
            lowObj = bpy.data.objects[self.lowpoly]
            cageObj = bpy.data.objects.new(cageName, lowObj.data.copy())  # duplicate obj but not instance
            cageObj.matrix_world = lowObj.matrix_world
            cageObj.show_wire = True
            cageObj.show_all_edges = True
            cageObj.draw_type = 'WIRE'
            PushPullModifier = cageObj.modifiers.new('Push', 'SHRINKWRAP')
            PushPullModifier.target = lowObj
            PushPullModifier.offset = 0.1
            PushPullModifier.offset = 0.1
            PushPullModifier.use_keep_above_surface = True
            PushPullModifier.wrap_method = 'PROJECT'
            PushPullModifier.use_negative_direction = True

            vg = cageObj.vertex_groups.new(cageName + '_weigh')

            weight = 1
            for vert in lowObj.data.vertices:
                vg.add([vert.index], weight, "ADD")

            PushPullModifier.vertex_group = vg.name

            context.scene.objects.link(cageObj)
            context.scene.cycles_baker_settings.bake_job_queue[self.bj_i].bake_pairs_list[self.pair_i].cage = cageObj.name
        return {'FINISHED'}


import time

loadStart = 0


from pathlib import Path
from os.path import exists


def abs_file_path(filePath):
    """Retuns absolute file path, using resolve (removes //..//"""
    abspathToFix = Path(bpy.path.abspath(filePath))  # crappy format like c:\\..\\...\\ddada.fbx
    if not exists(str(abspathToFix)):
        return filePath
    outputPathStr = str(abspathToFix.resolve())
    if abspathToFix.is_dir():
        outputPathStr += '\\'
    return outputPathStr


class CB_OT_CyclesTexturePreview(bpy.types.Operator):
    bl_idname = "cyclesbaker.texture_preview"
    bl_label = "Cycles Bake preview"
    bl_description = "Preview texture on model, by assigning bake result to lowpoly objects \n" \
                     "Press Shift - to preview multiple bake jobs"

    bj_i: bpy.props.IntProperty()
    shiftClicked = False

    def invoke(self, context, event):
        if event.shift:
            self.shiftClicked = True
        return self.execute(context)

    def attachCyclesmaterial(self, obj, bj_i):
        cycles_bake_settings = bpy.context.scene.cycles_baker_settings
        bj = cycles_bake_settings.bake_job_queue[bj_i]
        mat = bpy.data.materials.get(bj.name)
        if len(obj.material_slots) == 0:
            obj.data.materials.append(mat)
        else:
            obj.material_slots[0].material = mat
        if bpy.context.scene.render.engine == 'CYCLES':
            obj.data.materials[0].use_nodes = True

    def execute(self, context):
        cycles_bake_settings = bpy.context.scene.cycles_baker_settings
        if self.shiftClicked:
            bjList = [(i, bj) for i, bj in enumerate(cycles_bake_settings.bake_job_queue) if bj.activated]
        else:
            bjList = [(self.bj_i, cycles_bake_settings.bake_job_queue[self.bj_i])]
        addon_prefs = bpy.context.preferences.addons['sd_baker'].preferences
        imagesFromBakePasses = []
        for bj_i, bj in bjList:
            imagesFromBakePasses.clear()
            for bakepass in bj.bake_pass_list:  # refresh or load images from bakes
                if bakepass.activated:
                    bakedImgPath = bj.get_filepath()[:-1] + '\\' + bakepass.get_filename(bj) + '.' + addon_prefs.outputFormat
                    imgAlreadyExist = False
                    oldImg = []
                    for img in bpy.data.images:  # find if bake is already loaded into bi images
                        if abs_file_path(bakedImgPath) == abs_file_path(img.filepath):
                            if bakepass.activated:
                                img.reload()
                            imgAlreadyExist = True
                            oldImg = img
                            break
                    if imgAlreadyExist:
                        imagesFromBakePasses.append(oldImg)
                        # print("found bake in bi images")
                    else:
                        try:
                            im = bpy.data.images.load(bakedImgPath)
                            imagesFromBakePasses.append(im)
                        except:
                            print("Skipping loading image, because it can't be loaded: " + bakedImgPath)
                            imagesFromBakePasses.append(None)  # to preserve bakePass indexes
                else:
                    imagesFromBakePasses.append(None)  # to preserve breaking bakePass indexes
            bpy.context.space_data.viewport_shade = 'MATERIAL'
            mat = bpy.data.materials.get(bj.name)
            if mat is None:
                mat = bpy.data.materials.new(name=bj.name)
                mat.diffuse_color = (0.609125, 0.0349034, 0.8)
            obList = []
            for i_pair, pair in enumerate(bj.bake_pairs_list):
                if pair.lowpoly in bpy.data.objects.keys():  # create group for hipoly
                    obList.append(bpy.data.objects[pair.lowpoly])

                for obj in obList:
                    # if obj.type == "MESH" and (len(obj.material_slots) == 0 or obj.material_slots[0].material is None):
                    if obj.type == "MESH":
                        self.attachCyclesmaterial(obj, bj_i)
            if len(obList) == 0:
                continue

            matNodeTree = bpy.data.materials.get(bj.name).node_tree
            for node in matNodeTree.nodes:
                matNodeTree.nodes.remove(node)
            links = matNodeTree.links
            principledNode = matNodeTree.nodes.new('ShaderNodeBsdfPrincipled')

            principledNode.inputs['Roughness'].default_value = 0.4
            principledNode.location = 1300, 200
            outputNode = matNodeTree.nodes.new('ShaderNodeOutputMaterial')
            outputNode.location = 1500, 200
            links.new(principledNode.outputs[0], outputNode.inputs[0])
            previousID_AO_Node = None
            for bakeIndex, bakeImg in enumerate(imagesFromBakePasses):
                if bakeImg == None:  # skip imgs that could not be loaded
                    continue

                bakepass = bj.bake_pass_list[bakeIndex]
                # if not foundTextureSlotWithBakeImg: # always true in loop above wass not continued
                if bakepass.pass_name == "MAT_ID" or bakepass.pass_name == "AO":
                    imgNode = matNodeTree.nodes.new('ShaderNodeTexImage')
                    imgNode.name = bakepass.suffix
                    imgNode.label = bakepass.suffix
                    imgNode.location = bakeIndex * 100, bakeIndex * 10 + 380
                    imgNode.image = bakeImg

                    if previousID_AO_Node:
                        mixNode = matNodeTree.nodes.new('ShaderNodeMixRGB')
                        mixNode.location = bakeIndex * 100+200, bakeIndex * 10 + 100
                        mixNode.inputs[0].default_value = 1
                        if bakepass.pass_name == "AO":
                            mixNode.blend_type = 'MULTIPLY'

                        links.new(previousID_AO_Node.outputs[0], mixNode.inputs[2])
                        links.new(imgNode.outputs[0], mixNode.inputs[1])
                        links.new(mixNode.outputs[0], principledNode.inputs['Base Color'])
                        previousID_AO_Node = mixNode
                    else:  # first id or AO texture
                        links.new(imgNode.outputs[0], principledNode.inputs['Base Color'])
                        previousID_AO_Node = imgNode

                elif bakepass.pass_name == "NORMAL":
                    imgNormalNode = matNodeTree.nodes.new('ShaderNodeTexImage')
                    imgNormalNode.name = bakepass.suffix
                    imgNormalNode.label = bakepass.suffix
                    bakeImg.colorspace_settings.name = 'Linear'
                    imgNormalNode.image = bakeImg
                    imgNormalNode.color_space = 'NONE'
                    imgNormalNode.location = bakeIndex + 400, -400

                    normalMapNode = matNodeTree.nodes.new('ShaderNodeNormalMap')
                    normalMapNode.location = bakeIndex + 600, -400

                    if bakepass.nm_invert == 'NEG_Y':  # use DX normal == flip green chanel
                        if "FlipGreenChannel" not in bpy.data.node_groups.keys():
                            script_file = os.path.realpath(__file__)
                            filepath = os.path.dirname(script_file)+"\\baker_library.blend"
                            # read node group
                            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                                data_to.node_groups = ["FlipGreenChannel"]
                        flipGreenNode = matNodeTree.nodes.new('ShaderNodeGroup')
                        flipGreenNode.node_tree = bpy.data.node_groups['FlipGreenChannel']
                        flipGreenNode.location = bakeIndex + 600, -400
                        normalMapNode.location[0] += 200
                        links.new(imgNormalNode.outputs[0], flipGreenNode.inputs[0])
                        links.new(flipGreenNode.outputs[0], normalMapNode.inputs[1])
                        links.new(normalMapNode.outputs[0], principledNode.inputs['Normal'])
                    else:
                        links.new(imgNormalNode.outputs[0], normalMapNode.inputs[1])
                        links.new(normalMapNode.outputs[0], principledNode.inputs['Normal'])

                elif bakepass.pass_name == "OPACITY":
                    imgOpacityNode = matNodeTree.nodes.new('ShaderNodeTexImage')
                    imgOpacityNode.name = bakepass.suffix
                    imgOpacityNode.label = bakepass.suffix
                    imgOpacityNode.location = bakeIndex + 400, -200
                    imgOpacityNode.image = bakeImg
                    invert = matNodeTree.nodes.new('ShaderNodeInvert')
                    invert.location = bakeIndex + 600, -200
                    links.new(imgOpacityNode.outputs[0], invert.inputs[1])
                    links.new(invert.outputs[0], principledNode.inputs['Transmission'])
                    bpy.data.objects[pair.lowpoly].show_transparent = True
                    bpy.data.objects[pair.lowpoly].material_slots[0].material.game_settings.alpha_blend = 'ALPHA_ANTIALIASING'

                else:  # for all other just create img and do not link
                    imgNormalNode = matNodeTree.nodes.new('ShaderNodeTexImage')
                    imgNormalNode.name = bakepass.suffix
                    imgNormalNode.label = bakepass.suffix
                    imgNormalNode.image = bakeImg
                    imgNormalNode.location = bakeIndex*(-400), bakeIndex * 200

        return {'FINISHED'}


classes = (
    CyclesBakePair,
    CyclesBakePass,
    CyclesBakeJob,
    CyclesBakeSettings,
    CB_OT_CyclesBakeOp,
    CB_OT_SDAddPairOp,
    CB_OT_SDRemPairOp,
    CB_OT_SDAddPassOp,
    CB_OT_SDPassOp,
    CB_OT_SDAddJobOp,
    CB_OT_SDRemJobOp,
    CB_OT_ObjectPicker,
    CB_OT_CageMaker,
    CB_OT_CyclesTexturePreview,
    CB_PT_SDPanel,
    CyclesBakerPreferences,
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.cycles_baker_settings= bpy.props.PointerProperty(type=CyclesBakeSettings)
    bpy.types.Object.sd_orig_name = bpy.props.StringProperty(name="Original Name")
    bpy.types.Collection.sd_orig_name = bpy.props.StringProperty(name="Original Name")
    bpy.types.World.sd_orig_name = bpy.props.StringProperty(name="Original Name")
    bpy.types.Material.sd_orig_name = bpy.props.StringProperty(name="Original Name")


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Object.sd_orig_name
    del bpy.types.Collection.sd_orig_name
    del bpy.types.World.sd_orig_name
    del bpy.types.Material.sd_orig_name
    del bpy.types.Scene.cycles_baker_settings

if __name__ == "__main__":
    register()
