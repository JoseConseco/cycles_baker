#  (c) 2025 Bartosz Styperek based on - by Piotr Adamowicz work 2014 (MadMinstrel)

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


import bpy
from pathlib import Path
from .bake import draw_cage_callback


class CyclesBakerPreferences(bpy.types.AddonPreferences):
    bl_idname = 'cycles_baker'

    Info: bpy.props.StringProperty(name="Info", description="", default="")
    DIFFUSE: bpy.props.StringProperty(name="Mat ID suffix", description="", default='id')
    AO: bpy.props.StringProperty(name="AO suffix", description="", default='ao')
    NORMAL: bpy.props.StringProperty(name="NORMAL suffix", description="", default="nrm")
    # HEIGHT: bpy.props.StringProperty(name="Height map suffix", description="", default="heig")
    OPACITY: bpy.props.StringProperty(name="Opacity map suffix", description="", default="opacity")
    COMBINED: bpy.props.StringProperty(name="Combined map suffix", description="", default="combined")

    def draw(self, context):
        layout = self.layout
        layout.label(text=self.Info)
        layout.prop(self, "DIFFUSE")
        layout.prop(self, "AO")
        layout.prop(self, "NORMAL")
        # layout.prop(self, "HEIGHT")
        # layout.prop(self, "COMBINED")
        layout.prop(self, "OPACITY")


handleDrawRayDistance = []

class CyclesBakePair(bpy.types.PropertyGroup):
    def drawCage(self, context):

        global handleDrawRayDistance
        if self.draw_front_dist:
            # disable all other draw_front_dist
            bjobs = context.scene.cycles_baker_settings.bake_job_queue
            for bj in bjobs: # disable all draw_front_dist
                for pair in bj.bake_pairs_list:
                    if pair != self:
                        pair['draw_front_dist'] = False

            if handleDrawRayDistance:
                for h in handleDrawRayDistance:
                    bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
                handleDrawRayDistance.clear()

            args = (self, context)  # u can pass arbitrary class as first param  Instead of (self, context)
            handleDrawRayDistance.append(bpy.types.SpaceView3D.draw_handler_add(draw_cage_callback, args, 'WINDOW', 'POST_VIEW'))
        else:

            if handleDrawRayDistance:
                for h in handleDrawRayDistance:
                    bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
                handleDrawRayDistance.clear()

    activated: bpy.props.BoolProperty( name="Activated", description="Enable/Disable baking this pair of objects. Old bake result will be used if disabled", default=True)
    lowpoly: bpy.props.StringProperty(name="", description="Lowpoly mesh", default="")
    highpoly: bpy.props.StringProperty(name="", description="Highpoly mesh", default="")
    hp_type: bpy.props.EnumProperty( name="Object vs Group", description="", default="OBJ",
        items=[
            ('OBJ', '', 'Object', 'MESH_CUBE', 0,),
            ('GROUP', '', 'Group', 'GROUP', 1,),
        ]
    )
    use_cage: bpy.props.BoolProperty(name="Use Cage", description="Use cage object", default=False)
    cage: bpy.props.StringProperty(name="", description="Cage mesh", default="")
    ray_dist: bpy.props.FloatProperty( name="Ray distance", description="", default=1.0, min=0, max=10, subtype='FACTOR')
    draw_front_dist: bpy.props.BoolProperty( name="Draw Front distance", description="Draw Front Distance Overlay", default=False, update=drawCage)
    no_materials: bpy.props.BoolProperty(name="No Materials", default=False)




class CyclesBakePass(bpy.types.PropertyGroup):
    def upSuffix(self, context):
        addon_prefs = bpy.context.preferences.addons['cycles_baker'].preferences
        if self.pass_type == "AO":
            self.suffix = addon_prefs.AO
        if self.pass_type == "NORMAL":
            self.suffix = addon_prefs.NORMAL
        if self.pass_type == "DIFFUSE":
            self.suffix = addon_prefs.DIFFUSE
        # if self.pass_type == "HEIGHT":
        #     self.suffix = addon_prefs.HEIGHT
        if self.pass_type == "COMBINED":
            self.suffix = addon_prefs.COMBINED
        if self.pass_type == "OPACITY":
            self.suffix = addon_prefs.OPACITY

    activated: bpy.props.BoolProperty(name="Activated", default=True)

    pass_type: bpy.props.EnumProperty(name="Pass", default="NORMAL",
                                      items=(
                                           ("DIFFUSE", "Diffuse Color", ""),
                                           ("AO", "Ambient Occlusion", ""),
                                           ("NORMAL", "Normal", ""),
                                        #    ("HEIGHT", "Height", ""),
                                           ("OPACITY", "Opacity mask", ""),
                                        #    ("COMBINED", "Combined", ""),
                                      ), update=upSuffix)

    ao_distance: bpy.props.FloatProperty(name="Maximum Occluder Distance", description="Maximum Occluder Distance", default=0.1, min=0.0, max=1.0)
    samples: bpy.props.IntProperty(name="Samples", description="", default=32, min=8, max=512)
    suffix: bpy.props.StringProperty(name="Suffix", description="", default="")  # addon_prefs.NORMAL

    bake_all_highpoly: bpy.props.BoolProperty(name="Highpoly", default=False)
    environment_obj_vs_group: bpy.props.EnumProperty(name="Object vs Group", description="", default="OBJ", items=[
                                                     ('OBJ', '', 'Object', 'MESH_CUBE', 0), ('GROUP', '', 'Group', 'GROUP', 1)])
    environment_group: bpy.props.StringProperty(name="", description="Additional environment occluding object(or group)", default="")

    bit_depth: bpy.props.EnumProperty(name="Color Depth", description="", default="0",
                                      items=(("0", "8 bit(default)", ""),
                                             ("1", "16 bit", "")
                                             ))

    nm_space: bpy.props.EnumProperty(name='Type',description="Normal map space", default="TANGENT",
                                     items=(("TANGENT", "Tangent Space", ""),
                                            ("OBJECT", "World Space", "")))
    nm_invert: bpy.props.EnumProperty(name="Flip G", description="Invert green channel", default="POS_Y",
                                      items=(("POS_Y", "OpenGL", "Blender Compatible"),
                                             ("NEG_Y", "DirectX", "")))

    ray_distrib: bpy.props.EnumProperty(name="Ray distribution", description="", default="1",
                                        items=(("0", "Uniform", ""),
                                               ("1", "Cosine", "")
                                               ))

    def props(self):
        props = set()
        if self.pass_type == "AO":
            props = {"ao_distance", "samples", "environment_group", "ray_distrib"}
        if self.pass_type == "NORMAL":
            props = {"nm_space", "nm_invert", "bit_depth"}
        if self.pass_type == "NORMAL":
            props = {"nm_space", "nm_invert", "bit_depth"}
        # if self.pass_type == "HEIGHT":
        #     props = {'bit_depth'}

        return props


    def get_filename(self, bj):
        name = bj.name
        if len(self.suffix) > 0:
            name += "_" + self.suffix
        return name



class CyclesBakeJob(bpy.types.PropertyGroup):

    def update_pad(self, context):
        if self.padding_mode == 'AUTO':
            self['padding_size'] = int(int(self.bakeResolution)/64)


    activated: bpy.props.BoolProperty(name="Activated", description="Disable baking set of high-low pairs", default=True)
    expand: bpy.props.BoolProperty(name="Expand", default=True)
    bakeResolution: bpy.props.EnumProperty(name="Resolution", default="1024",
                                           items=(("128", "128x128", ""),
                                                  ("256", "256x256", ""),
                                                  ("512", "512x512", ""),
                                                  ("1024", "1024x1024", ""),
                                                  ("2048", "2048x2048", ""),
                                                  ("4096", "4096x4096", ""))
                                           )
    antialiasing: bpy.props.EnumProperty(name="Anti-aliasing", description="It will render image n times bigger than render target, to create the AA affect", default="1",
                                         items=(("1", "None", ""),
                                                ("2", "2x", ""),
                                                ("4", "4x", "")))

    padding_mode: bpy.props.EnumProperty(name='Padding Mode', description='Prevents background color beeding into texture',
                                               items=[
                                                   ('AUTO', 'Automatic Padding', 'Padding will be set to around 1% of texture size'),
                                                   ('FIXED', 'Fixed Padding', 'Set padding size by hand')
                                               ], default='AUTO', update=update_pad)
    padding_size: bpy.props.IntProperty(name='Padding', description='Add color margin around bake result (0 - disabled)\nWarning - may be slow on big images', default = 10, min=0, soft_max=40)

    output: bpy.props.StringProperty(name='File path',
                                     description='The path of the output image.',
                                     default='//textures/',
                                     subtype='FILE_PATH')
    name: bpy.props.StringProperty(name='name', description="Output texture name", default='bake_name')


    bake_pairs_list: bpy.props.CollectionProperty(type=CyclesBakePair)
    bake_pass_list: bpy.props.CollectionProperty(type=CyclesBakePass)

    def get_out_dir_path(self):
        return Path(bpy.path.abspath(self.output)).resolve()



class CyclesBakeSettings(bpy.types.PropertyGroup):
    bake_job_queue: bpy.props.CollectionProperty(type=CyclesBakeJob)


def register_props():
    bpy.types.Scene.cycles_baker_settings= bpy.props.PointerProperty(type=CyclesBakeSettings)


def unregister_props():
    del bpy.types.Scene.cycles_baker_settings

