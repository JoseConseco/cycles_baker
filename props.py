import bpy
from bpy.props import *
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
        if self.draw_front_dist:
            if handleDrawRayDistance:
                bpy.types.SpaceView3D.draw_handler_remove(handleDrawRayDistance[0], 'WINDOW')

            args = (self, bpy.context)  # u can pass arbitrary class as first param  Instead of (self, context)
            handleDrawRayDistance[:] = [bpy.types.SpaceView3D.draw_handler_add(draw_cage_callback, args, 'WINDOW', 'POST_VIEW')]
        else:
            if handleDrawRayDistance:
                bpy.types.SpaceView3D.draw_handler_remove(handleDrawRayDistance[0], 'WINDOW')
                handleDrawRayDistance[:] = []

    activated: bpy.props.BoolProperty(
        name="Activated", description="Enable/Disable baking this pair of objects. Old bake result will be used if disabled", default=True)
    lowpoly: bpy.props.StringProperty(name="", description="Lowpoly mesh", default="")
    highpoly: bpy.props.StringProperty(name="", description="Highpoly mesh", default="")
    hp_type: bpy.props.EnumProperty(name="Object vs Group", description="", default="OBJ", items=[
                                            ('OBJ', '', 'Object', 'MESH_CUBE', 0), ('GROUP', '', 'Group', 'GROUP', 1)])
    use_cage: bpy.props.BoolProperty(name="Use Cage", description="Use cage object", default=False)
    cage: bpy.props.StringProperty(name="", description="Cage mesh", default="")
    front_distance_modulator: bpy.props.FloatProperty( name="Front distance modulator", description="", default=1.0, min=0, max=10, subtype='FACTOR')
    draw_front_dist: bpy.props.BoolProperty( name="Draw Front distance", description="", default=False, update=drawCage)
    no_materials: bpy.props.BoolProperty(name="No Materials", default=False)



class CyclesBakePass(bpy.types.PropertyGroup):
    def upSuffix(self, context):
        addon_prefs = bpy.context.preferences.addons['cycles_baker'].preferences
        if self.pass_name == "AO":
            self.suffix = addon_prefs.AO
        if self.pass_name == "NORMAL":
            self.suffix = addon_prefs.NORMAL
        if self.pass_name == "DIFFUSE":
            self.suffix = addon_prefs.DIFFUSE
        # if self.pass_name == "HEIGHT":
        #     self.suffix = addon_prefs.HEIGHT
        if self.pass_name == "COMBINED":
            self.suffix = addon_prefs.COMBINED
        if self.pass_name == "OPACITY":
            self.suffix = addon_prefs.OPACITY

    activated: bpy.props.BoolProperty(name="Activated", default=True)

    pass_name: bpy.props.EnumProperty(name="Pass", default="NORMAL",
                                      items=(
                                           ("DIFFUSE", "Diffuse Color", ""),
                                           ("AO", "Ambient Occlusion", ""),
                                           ("NORMAL", "Normal", ""),
                                        #    ("HEIGHT", "Height", ""),
                                           ("OPACITY", "Opacity mask", ""),
                                        #    ("COMBINED", "Combined", ""),
                                      ), update=upSuffix)

    material_override: bpy.props.StringProperty(name="Material Override", description="", default="")
    ao_distance: bpy.props.FloatProperty(name="Distance", description="Maximum Occluder Distance", default=0.1, min=0.0, max=1.0)
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
        if self.pass_name == "NORMAL":
            props = {"nm_space", "nm_invert", "bit_depth"}
        # if self.pass_name == "HEIGHT":
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
    antialiasing: bpy.props.EnumProperty(name="Anti-aliasing", description="Anti-aliasing", default="1",
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


def register_props():
    bpy.types.Scene.cycles_baker_settings= bpy.props.PointerProperty(type=CyclesBakeSettings)


def unregister_props():
    del bpy.types.Scene.cycles_baker_settings

