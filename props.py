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
from .utils import get_addon_preferences
from .bake import draw_cage_handle, set_ao_mod, set_depth_mod, get_ao_mod, set_curvature_mod, get_depth_mod, get_curvature_mod,  ht_channel_mixing


class CyclesBakePair(bpy.types.PropertyGroup):

    def drawCage(self, context):
        draw_cage_handle(context, self)

    activated: bpy.props.BoolProperty( name="Activated", description="Enable/Disable baking this pair of objects. Old bake result will be used if disabled", default=True)
    expand: bpy.props.BoolProperty(name="Expand", default=True)
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


AO_PARAM_MAP = {
    "gn_ao_flip_normals": "Input_16",
    "gn_ao_samples": "Input_3",
    "gn_ao_spread_angle": "Input_4",
    "gn_ao_environment": "Socket_1",
    "gn_ao_blur_steps": "Input_8",
    "gn_ao_use_additional_mesh": "Input_10",
    "gn_ao_extra_object": "Input_11",
    "gn_ao_max_ray_dist": "Socket_0"
}

# Mapping for Depth pass parameters
DEPTH_PARAM_MAP = {
    "depth_reference_object": "Socket_2",
    "depth_low_offset": "Socket_3",
    "depth_high_offset": "Socket_4"
}

# Mapping for Curvature pass parameters
CURVATURE_PARAM_MAP = {
    "curvature_mode": "Socket_4",
    "curvature_contrast": "Socket_2",
    "curvature_brightness": "Socket_6",
    "curvature_blur": "Socket_3"
}


class CyclesBakePass(bpy.types.PropertyGroup):
    activated: bpy.props.BoolProperty(name="Activated", default=True)
    expand: bpy.props.BoolProperty(name="Expand", default=False)

    pass_type: bpy.props.EnumProperty(name="Pass",
                                      items=(
                                      ("DIFFUSE", "Diffuse Color", ""),
                                      ("AO", "Ambient Occlusion", ""),
                                      ("AO_GN", "Ambient Occlusion (GeoNodes)", ""),
                                      ("NORMAL", "Normal", ""),
                                      ("OPACITY", "Opacity mask", ""),
                                      ("DEPTH", "Depth (GeoNodes)", ""),
                                      ("CURVATURE", "Curvature (GeoNodes)", "")),
                                      default="NORMAL",)

    # NORMAL  - baked from cycles
    # bit_depth: bpy.props.EnumProperty(name="Color Depth", description="", default="0",
    #                                   items=(("0", "8 bit(default)", ""),
    #                                          ("1", "16 bit", "")
    #                                          ))
    nm_space: bpy.props.EnumProperty(name='Type',description="Normal map space",
                                     items=(("TANGENT", "Tangent Space", ""),
                                            ("OBJECT", "World Space", "")),
                                     default="TANGENT",)
    nm_invert: bpy.props.EnumProperty(name="Flip G", description="Invert green channel",
                                      items=(("POS_Y", "OpenGL", "Blender Compatible"),
                                             ("NEG_Y", "DirectX", "")),
                                      default="POS_Y",)

    def update_gn_modifier(self, context):
        if context.scene.name == "MD_PREVIEW":
            proxy_obj = bpy.data.objects.get("HighProxy_Preview")

            # Curvature modifier update
            ao_mod = get_ao_mod(proxy_obj)
            if ao_mod:
                set_ao_mod(proxy_obj, self)
                return

            # Depth modifier update
            depth_mod = get_depth_mod(proxy_obj)
            if depth_mod:
                low_proxy = bpy.data.objects.get("LowProxy_Preview")
                set_depth_mod(proxy_obj, low_proxy, self)
                return

            # AO modifier update
            curvature_mod = get_curvature_mod(proxy_obj)
            if curvature_mod:
                set_curvature_mod(proxy_obj, self)
                return

    # AO - cycles based
    ao_distance: bpy.props.FloatProperty(name="Maximum Occluder Distance", description="Maximum Occluder Distance", default=0.1, min=0.0, max=1.0)
    samples: bpy.props.IntProperty(name="Samples", description="", default=32, min=8, max=512)

    occluder_obj: bpy.props.StringProperty(name="Occluder Object", description="Additional occluding object", default="")

    # ray_distribution: bpy.props.EnumProperty(name="Ray distribution", description="", default="1",
    #                                     items=(("0", "Uniform", ""),
    #                                            ("1", "Cosine", "")
    #                                            ))


    # AO_GN   - geo nodes base
    gn_ao_samples: bpy.props.IntProperty(name="Samples", description="Increase AO ray samples (higher quality by slower)", default=8, min=1, max=200, update=update_gn_modifier)
    gn_ao_environment: bpy.props.EnumProperty(name="Environment", description="",
                                              items=(("0", "Uniform", "Light comes uniformly in all directions"),
                                                     ("1", "Top Lit", "Light comes from above")),
                                              default="0", update=update_gn_modifier)
    gn_ao_spread_angle: bpy.props.FloatProperty(name="Spread Angle", description="0 - spread: only shoot rays along surface normal", default=3.141599, min=0.0, max=3.141592, subtype='ANGLE', update=update_gn_modifier)
    gn_ao_max_ray_dist: bpy.props.FloatProperty(name="Max Ray Distance", description="Maximum raycast distance", default=1.0, min=0.1, update=update_gn_modifier)
    gn_ao_blur_steps: bpy.props.IntProperty(name="Blur Steps", description="Number of blur iterations", default=1, min=0, max=6, update=update_gn_modifier)
    gn_ao_flip_normals: bpy.props.BoolProperty(name="Flip Normals", description="Can be used for thickness map", default=False, update=update_gn_modifier)
    gn_ao_use_additional_mesh: bpy.props.BoolProperty(name="Use Additional Mesh", description="Use additional occluder object", default=False, update=update_gn_modifier)
    gn_ao_extra_object: bpy.props.PointerProperty(name="Extra Object", description="Adds extra occluder object", type=bpy.types.Object, update=update_gn_modifier)


    # DEPTH
    depth_low_offset: bpy.props.FloatProperty(name="Low Offset", description="Black level offset (for valleys)", default=0.0, min=-1000.0, max=1000.0, subtype='DISTANCE', update=update_gn_modifier)
    depth_high_offset: bpy.props.FloatProperty(name="High Offset", description="White level offset (for hills)", default=0.0, min=-1000.0, max=1000.0, subtype='DISTANCE', update=update_gn_modifier)


    # CURVATURE
    curvature_mode: bpy.props.EnumProperty(name="Mode", description="",
                                           items=(("0", "Smooth", ""),
                                                  ("1", "Sharp", "")),
                                           default="0", update=update_gn_modifier)
    curvature_contrast: bpy.props.FloatProperty(name="Contrast", description="", default=1.0, min=0.01, soft_max=1.0, subtype='FACTOR', update=update_gn_modifier)
    curvature_brightness: bpy.props.FloatProperty(name="Brightness", description="", default=0.0, min=-1.0, max=1.0, subtype='FACTOR', update=update_gn_modifier)
    curvature_blur: bpy.props.IntProperty(name="Blur", description="", default=3, min=0, soft_max=16, update=update_gn_modifier)


    def props(self):
        prop_configs = {
            "AO": {
                "ao_distance": None,
                "samples": None,
                "occluder_obj": {"type": "prop_search", "search_data": "objects"},
            },
            "AO_GN": {
                "gn_ao_samples": None,
                "gn_ao_environment": None,
                "gn_ao_spread_angle": None,
                "gn_ao_max_ray_dist": None,
                "gn_ao_blur_steps": None,
                "gn_ao_flip_normals": {"type": "toggle"},
                "gn_ao_use_additional_mesh": {"type": "toggle"},
                "gn_ao_extra_object": None
            },
            "NORMAL": {
                "nm_space": None,
                "nm_invert": None
            },
            "DEPTH": {
                "depth_low_offset": None,
                "depth_high_offset": None
            },
            "CURVATURE": {
                "curvature_mode": None,
                "curvature_contrast": None,
                "curvature_brightness": None,
                "curvature_blur": None
            }
        }
        return prop_configs.get(self.pass_type, {})

    def get_pass_suffix(self):
        addon_prefs = get_addon_preferences()
        suffix_map = {
            "AO": addon_prefs.AO,
            "AO_GN": addon_prefs.AO,
            "NORMAL": addon_prefs.NORMAL,
            "DIFFUSE": addon_prefs.DIFFUSE,
            "COMBINED": addon_prefs.COMBINED,
            "OPACITY": addon_prefs.OPACITY,
            "DEPTH": addon_prefs.DEPTH,
            "CURVATURE": addon_prefs.CURVATURE,
        }
        return suffix_map.get(self.pass_type, "")

    def get_filename(self, bj):
        name = bj.name
        suffix = self.get_pass_suffix()
        if len(suffix) > 0:
            name += "_" + suffix
        return name



class CyclesBakeJob(bpy.types.PropertyGroup):

    def update_pad(self, context):
        if self.padding_mode == 'AUTO':
            self['padding_size'] = int(int(self.bakeResolution)/64)


    def update_compose(self, context):
        if self.hair_bake_composite:
            ht_channel_mixing(context, self)

    activated: bpy.props.BoolProperty(name="Activated", description="Disable baking set of high-low pairs", default=True)
    expand: bpy.props.BoolProperty(name="Expand", default=True)
    use_channel_packing: bpy.props.BoolProperty(name="Use channel packing", description="Mix RGBA channels of baked textures in 'Texture Channel Mixing' nodes editor (for now this works only if you have Hair Tool)", default=False, update=update_compose)

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
    name: bpy.props.StringProperty(name='name', description="Output texture name", default='bake')


    bake_pairs_list: bpy.props.CollectionProperty(type=CyclesBakePair)
    bake_pass_list: bpy.props.CollectionProperty(type=CyclesBakePass)


    def get_out_dir_path(self):
        return Path(bpy.path.abspath(self.output)).resolve()



class CyclesBakeSettings(bpy.types.PropertyGroup):
    bake_job_queue: bpy.props.CollectionProperty(type=CyclesBakeJob)
    pair_spacing_distance: bpy.props.FloatProperty(name="Pair Spread Distance", description="Offset added between high-low pairs during bake, to prevent object pairs affecting each other (per scene option)", default=10.0, min=0.01, soft_max=100.0)


def register_props():
    bpy.types.Scene.cycles_baker_settings= bpy.props.PointerProperty(type=CyclesBakeSettings)


def unregister_props():
    del bpy.types.Scene.cycles_baker_settings

