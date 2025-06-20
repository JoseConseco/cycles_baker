#  (c) 2025 Bartosz Styperek based on - by Piotr Adamowicz addon (from 2014 -MadMinstrel)

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
import os
import aud
import math
from mathutils import Vector
from datetime import datetime

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from .utils import abs_file_path, get_addon_preferences, add_geonodes_mod, import_mat
from . import ui as globa_ui

# CB_AOPass
# Input_16 > {'name': 'Flip Normals', 'type': 'NodeSocketBool', 'default_value': False, 'description': 'Can be used for thickness map'}
# Input_3 > {'name': 'Samples', 'type': 'NodeSocketInt', 'default_value': 8, 'min_value': 1, 'max_value': 200, 'subtype': 'NONE', 'description': 'Increase AO ray samples (higher quality by slower)'}
# Input_4 > {'name': 'Spread Ange', 'type': 'NodeSocketFloat', 'default_value': 3.1415998935699463, 'min_value': 0.0, 'max_value': 3.1415927410125732, 'subtype': 'ANGLE', 'description': '0 - spread: only shoot rays along surface normal'}
# Socket_1 > {'name': 'Environment', 'type': 'NodeSocketMenu', 'default_value': 'Uniform Environment', 'description': ''}
# Input_8 > {'name': 'Blur Steps', 'type': 'NodeSocketInt', 'default_value': 1, 'min_value': 0, 'max_value': 6, 'subtype': 'NONE', 'description': ''}
# Input_10 > {'name': 'Use Additional Mesh', 'type': 'NodeSocketBool', 'default_value': False, 'description': 'Use additional occluder object'}
# Input_11 > {'name': 'Extra Object', 'type': 'NodeSocketObject', 'default_value': None, 'description': 'Adds extra occluder object'}
# Socket_0 > {'name': 'Max Ray Dist', 'type': 'NodeSocketFloat', 'default_value': 1.0, 'min_value': 0.10000000149011612, 'max_value': 3.4028234663852886e+38, 'subtype': 'NONE', 'description': 'Maximum raycast distance'}
# CB_DepthPass
# Socket_2 > {'name': 'Object (for Distance)', 'type': 'NodeSocketObject', 'default_value': None, 'description': 'Object for calculating distance from'}
# Socket_3 > {'name': 'Low Offset', 'type': 'NodeSocketFloat', 'default_value': 0.0, 'min_value': -10000.0, 'max_value': 10000.0, 'subtype': 'DISTANCE', 'description': 'Black level offset (for valleys)'}
# Socket_4 > {'name': 'High Offset', 'type': 'NodeSocketFloat', 'default_value': 0.0, 'min_value': -10000.0, 'max_value': 10000.0, 'subtype': 'DISTANCE', 'description': 'White level offset (for hils)'}
# CB_CurvaturePass
# Socket_4 > {'name': 'Menu', 'type': 'NodeSocketMenu', 'default_value': 'Smooth', 'description': ''}
# Socket_2 > {'name': 'Contrast', 'type': 'NodeSocketFloat', 'default_value': 1.0, 'min_value': 0.009999999776482582, 'max_value': 1.0, 'subtype': 'FACTOR', 'description': ''}
# Socket_3 > {'name': 'Blur', 'type': 'NodeSocketInt', 'default_value': 3, 'min_value': 0, 'max_value': 2147483647, 'subtype': 'NONE', 'description': ''}

def add_split_extrude_mod(obj, displace_val):
    gn_displce = add_geonodes_mod(obj, "Cage CBaker", "CycBaker_SplitExtrude")
    #  'Socket_2' > 'Offset'
    gn_displce['Socket_2'] = displace_val  # set extrusion distance
    return gn_displce

def add_collection_to_mesh_mod(obj, coll):
    coll_to_mesh = add_geonodes_mod(obj, "Collection To Mesh CBaker", "CB_CollectionToMesh")
    #  'Socket_2' > 'Collection'
    coll_to_mesh['Socket_2'] = coll  # set collection name
    # coll_to_mesh['Socket_3'] = material
    return coll_to_mesh

def get_ao_mod(obj):
    return obj.modifiers.get("AO CBaker")

def get_depth_mod(obj):
    return obj.modifiers.get("Depth CBaker")

def get_curvature_mod(obj):
    return obj.modifiers.get("Curvature CBaker")

def set_ao_mod(obj, pass_settings):
    ao_mod = get_ao_mod(obj)
    if not ao_mod:
        ao_mod = add_geonodes_mod(obj, "AO CBaker", "CB_AOPass")
    # Set AO modifier properties from pass settings
    ao_mod['Input_16'] = pass_settings.gn_ao_flip_normals        # Flip Normals
    ao_mod['Input_3'] = pass_settings.gn_ao_samples              # Samples
    ao_mod['Input_4'] = pass_settings.gn_ao_spread_angle         # Spread Angle
    ao_mod['Socket_1'] = int(pass_settings.gn_ao_environment)         # Environment Mode
    ao_mod['Input_8'] = pass_settings.gn_ao_blur_steps          # Blur Steps
    ao_mod['Input_10'] = pass_settings.gn_ao_use_additional_mesh # Use Additional Mesh
    ao_mod['Input_11'] = pass_settings.gn_ao_extra_object       # Extra Object
    ao_mod['Socket_0'] = pass_settings.gn_ao_max_ray_dist       # Max Ray Dist
    obj.modifiers.update()  # Update modifier to apply changes
    obj.update_tag()
    # obj.data.update()
    # context.scene.update_tag()
    # context.view_layer.update()
    return ao_mod

def set_depth_mod(obj, ref_obj, pass_settings):
    """Calculate depth based on distance from reference (usually lowpoly) object."""
    depth_mod = get_depth_mod(obj)
    if not depth_mod:
        depth_mod = add_geonodes_mod(obj, "Depth CBaker", "CB_DepthPass")
    depth_mod['Socket_2'] = ref_obj                          # Object for Distance
    depth_mod['Socket_3'] = pass_settings.depth_low_offset   # Low Offset
    depth_mod['Socket_4'] = pass_settings.depth_high_offset  # High Offset
    obj.modifiers.update()  # Update modifier to apply changes
    obj.update_tag()
    return depth_mod

def set_curvature_mod(obj, pass_settings):
    curvature_mod = get_curvature_mod(obj)
    if not curvature_mod:
        curvature_mod = add_geonodes_mod(obj, "Curvature CBaker", "CB_CurvaturePass")
    curvature_mod['Socket_4'] = int(pass_settings.curvature_mode)     # Menu (Smooth/Sharp)
    curvature_mod['Socket_2'] = pass_settings.curvature_contrast # Contrast
    curvature_mod['Socket_3'] = pass_settings.curvature_blur     # Blur
    obj.modifiers.update()  # Update modifier to apply changes
    obj.update_tag()
    return curvature_mod

def import_attrib_bake_mat():
    bake_mat = import_mat("CBaker_AttribMaterial")
    return bake_mat


def get_raycast_distance(bj, pair):
    low_poly = bpy.data.objects[pair.lowpoly]
    objBBoxSize = 0.2*Vector(low_poly.dimensions[:]).length
    return pair.ray_dist * objBBoxSize

AO_NODES = "CB_AOPass"
DEPTH_NODES = "CB_DepthPass"
CURVATURE_NODES = "CB_CurvaturePass"

BG_color = {
    "NORMAL": np.array([0.5, 0.5, 1.0, 1.0]),
    "AO": np.array([1.0, 1.0, 1.0, 1.0]),
    "AO_GN": np.array([1.0, 1.0, 1.0, 1.0]),
    "DIFFUSE": np.array([0.0, 0.0, 0.0, 0.0]),
    "OPACITY": np.array([0.0, 0.0, 0.0, 0.0]),
    "DEPTH": np.array([0.0, 0.0, 0.0, 1.0]),
    "CURVATURE": np.array([0.5, 0.5, 0.5, 1.0]),
}

shader_uniform = gpu.shader.from_builtin('UNIFORM_COLOR')
Verts = None
Normals = None
Indices = None
def draw_cage_callback(self, context):
    low_poly = bpy.data.objects.get( self.lowpoly )
    if not self.draw_front_dist or not self.lowpoly or not low_poly:
        return
    global Verts, Normals, Indices
    if low_poly.type == 'MESH' and context.mode == 'OBJECT':
        for bj in bpy.data.scenes['Scene'].cycles_baker_settings.bake_job_queue:
            for pair in bj.bake_pairs_list:
                if pair == self:
                    parentBakeJob = bj
                    break

        ray_dist = get_raycast_distance(parentBakeJob, self)

        mesh = low_poly.data
        vert_count = len(mesh.vertices)
        mesh.calc_loop_triangles()

        # mat_np = np.array(low_poly.matrix_world, 'f')
        if Verts is None or Verts.shape[0] != vert_count:
            Vertices = np.empty((vert_count, 3), 'f')
            Normals = np.empty((vert_count, 3), 'f')
            Indices = np.empty((len(mesh.loop_triangles), 3), 'i')

        mesh.vertices.foreach_get( "co", Vertices.ravel())
        mesh.vertices.foreach_get("normal", Normals.ravel())
        mesh.loop_triangles.foreach_get( "vertices", Indices.ravel())

        Vertices = Vertices + Normals * ray_dist  # 0,86 to match substance ray distance

        # old way to transform vertices with matrix
        # coords_4d = np.ones((vert_count, 4), 'f')
        # coords_4d[:, :-1] = Vertices
        # vertices = np.einsum('ij,aj->ai', mat_np, coords_4d)[:, :-1]

        gpu.state.blend_set('ALPHA')
        gpu.state.face_culling_set('BACK')

        face_color = (0, 0.8, 0, 0.5) if self.draw_front_dist else (0.8, 0, 0, 0.5)

        with gpu.matrix.push_pop():
            gpu.matrix.multiply_matrix(low_poly.matrix_world)
            shader_uniform.bind()
            shader_uniform.uniform_float("color", face_color)
            batch = batch_for_shader(shader_uniform, 'TRIS', {"pos": Vertices}, indices=Indices)
            batch.draw(shader_uniform)

        # restore gpu defaults
        gpu.state.blend_set('NONE')
        gpu.state.face_culling_set('NONE')


# Process baked texture - add padding, to it
#############################################

vert_out = gpu.types.GPUStageInterfaceInfo("my_bake_padding")
vert_out.smooth('VEC2', "uvInterp")

shader_info = gpu.types.GPUShaderCreateInfo()

shader_info.push_constant('VEC2', "img_size") # or 'INT' or 'UINT' ?
shader_info.push_constant('INT', "radius") # or vec2 ?
shader_info.push_constant('VEC3', "bg_color")
shader_info.push_constant('BOOL', "solid_bg")


shader_info.sampler(0, 'FLOAT_2D', "image")
shader_info.vertex_in(0, 'VEC2', "position")
shader_info.vertex_in(1, 'VEC2', "uv")
shader_info.vertex_out(vert_out)
shader_info.fragment_out(0, 'VEC4', "FragColor")


shader_info.vertex_source("""
// in vec2 position;
// in vec2 uv;

// out vec2 uvInterp;

void main()
{
    uvInterp = uv;
    gl_Position = vec4(position, 0.0, 1.0);
}
""")

shader_info.fragment_source("""
// uniform sampler2D image;
// uniform vec2 img_size;
// uniform int radius;
// uniform vec3 bg_color;
// uniform bool solid_bg;

// in vec2 uvInterp;

// out vec4 FragColor;

vec4 blend_premultiplied(vec4 imgA, vec4 imgB){
  vec4 out_img = vec4(imgA.rgb * imgA.a, imgA.a) + (1.-imgA.a)*vec4(imgB.rgb * imgB.a, imgB.a); //* pretultiplied ver
  out_img.rgb = out_img.rgb / out_img.a;
  return out_img;
}

// vec2[] offset = vec2[](vec2(-1,0), vec2(1,0), vec2(0,1), vec2(0,-1), vec2(1,1), vec2(-1,1), vec2(1,-1), vec2(-1,-1));

void main() {
    vec2 uv = uvInterp;
    vec4 img = texture(image, uv);
    vec4 out_img = vec4(0,0,0,0);

    if (img.a > 0.9){
        out_img = img;
    }else{
        float sample_weight = 0.;
        for (int r=1; r<=radius; r++){ //sample circle - with sample count ~ rad
          float sample_cnt = 9.*pow(float(r), .3);
          for(int i = 0; i < int(sample_cnt); i++){
            float alpha = 6.28 * float(i) /sample_cnt;
            vec2 uv_offset = uv + vec2(cos(alpha), sin(alpha))/img_size*float(r);
            if (uv_offset.x<0. || uv_offset.x>1. || uv_offset.y<0. || uv_offset.y>1.) // skip borders
                break;
            vec4 img_sample = texture(image, uv_offset);
            if (img_sample.a > 0.8) {
                out_img += img_sample*img_sample.a;
                sample_weight += img_sample.a;
            }
          }
          if (sample_weight > 0.2*sample_cnt) // skip outer rad if got enough samples
              break;
        }
        if (sample_weight > 0.)
            out_img = out_img/float(sample_weight);

        out_img.a = step(0.1, out_img.a);
        out_img.rgb = mix(bg_color, out_img.rgb, out_img.a); // use original background in black places
        out_img.rgb = mix(out_img.rgb, img.rgb, img.a); // overlay original image for using alpha
        if (solid_bg)
            out_img.a = 1.;
    }
    FragColor = vec4(pow(out_img.rgb, vec3(0.4545)), out_img.a);

}
""")

shader = gpu.shader.create_from_info(shader_info)
del vert_out
del shader_info

# shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
batch = batch_for_shader(
    shader, 'TRI_FAN',
    {
        "position": [ [-1,-1], [1,-1], [1,1], [-1,1] ],
#        "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1)),
        "uv": [ [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0] ],
    },
)


def add_padding_offscreen(img, img_x, img_y, padding_size, avg_col=(0, 0, 0)):
    pad_time = datetime.now()
    offscreen = gpu.types.GPUOffScreen(img_x, img_y)
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(avg_col[0], avg_col[1], avg_col[2], 1.0), depth=1)

        # Generate texture
        # buffer = img.pixels[:] # works?
        # pixels = gpu.types.Buffer('FLOAT', len(buffer), buffer)
        # tex= gpu.types.GPUTexture((img_x, img_y), format='RGBA8', data=pixels)
        tex = gpu.texture.from_image(img)

        shader.bind()
        shader.uniform_float("img_size", [img_x, img_y])
        shader.uniform_int("radius", padding_size)
        shader.uniform_sampler("image", tex) # should work okk
        shader.uniform_float("bg_color", np.power(avg_col, 2.2))
        shader.uniform_bool("solid_bg", [False])

        gpu.state.blend_set("ALPHA") # nicer result
        gpu.state.depth_test_set("LESS") # nicer result for whatever rease
        batch.draw(shader)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")

        color = np.array(fb.read_color(0, 0, img_x, img_y, 4, 0, 'UBYTE').to_list())
    offscreen.free()
    img.pixels = np.array(color).ravel()/255
    # img.pixels = np.power(np.array(out_buffer)/255, 0.454).ravel().tolist()
    print("Padding time: ", datetime.now() - pad_time)



def eval_mesh_from_obj(self, obj, deps):
    '''Old Not used anymore- will not work on hair system on curve anyway (old curves with bevel worked? )'''
    parentCurve_eval = obj.evaluated_get(deps) # wont work on emptyies.., for hair curves
    mesh = bpy.data.meshes.new_from_object(parentCurve_eval) # XXX: errors out on new Hair Curves (no geo data)

    new_obj = bpy.data.objects.new(obj.name + "_mesh", mesh)
    new_obj.matrix_world = obj.matrix_world
    return new_obj



class CB_OT_CyclesBakeOps(bpy.types.Operator):
    bl_idname = "cycles.bake"
    bl_label = "Bake Objects"
    bl_description = "Bake selected pairs of highpoly-lowpoly objects using blender Bake 'Selected to Active' feature"
    bl_options = {'REGISTER', 'UNDO'}

    bake_pair_index: bpy.props.IntProperty(name="Bake Pair Index", default=-1, description="Index of the bake pair to process")


    @classmethod
    def description(cls, context, properties):
        if properties.bake_pair_index > -1:
            return "Bake ONLY selected pair: of highpoly-lowpoly objects"
        else:
            return "Bake selected pairs of highpoly-lowpoly objects using blender Bake 'Selected to Active' feature"

    @staticmethod
    def get_set_first_material_slot(obj):
        first_slot_mat = obj.material_slots[0].material if len(obj.material_slots) > 0 else None
        if first_slot_mat:
            first_slot_mat.use_nodes = True # be sure it has nodes for setting active img texture
            return first_slot_mat
        low_bake_mat = bpy.data.materials.get("CyclesBakeMat_MD_TEMP")
        if low_bake_mat is None:
            low_bake_mat = bpy.data.materials.new(name="CyclesBakeMat_MD_TEMP")
            # low_bake_mat.diffuse_color = (0.609125, 0.0349034, 0.8, 1.0)
        low_bake_mat.use_nodes = True
        # If no material slots exist, create one
        if len(obj.material_slots) == 0:
            obj.data.materials.append(None)
        # Assign bake material to first slot
        obj.material_slots[0].material = low_bake_mat
        return low_bake_mat

    @staticmethod
    def create_bake_mat_and_node():
        low_obj = bpy.data.objects.get("LOWPOLY_MD_TMP")
        bake_mat = CB_OT_CyclesBakeOps.get_set_first_material_slot(low_obj)
        imgnode = bake_mat.node_tree.nodes.get('MDtarget')
        if not imgnode:
            imgnode = bake_mat.node_tree.nodes.new(type="ShaderNodeTexImage")
            imgnode.name = 'MDtarget'
            imgnode.label = 'MDtarget'

        imgnode.image = bpy.data.images["MDtarget"]
        bake_mat.node_tree.nodes.active = imgnode


    @staticmethod
    def obj_to_mesh(context, src_obj, coll):
        cp = src_obj.copy()
        coll.objects.link(cp)  # link all objects from hipoly group to highpoly collection
        with context.temp_override(selected_editable_objects=[cp], active_object=cp, selected_objects=[cp]):
            bpy.ops.object.convert(target='MESH')
            cp['tmp'] = True
        return cp  # return converted object, so it can be linked to highpoly collection

    @staticmethod
    def make_inst_real(context, src_obj, coll):
        cp = src_obj.copy()
        coll.objects.link(cp)  # link all objects from hipoly group to highpoly collection
        cp_name = cp.name
        cp['tmp'] = True  # mark as tmp, so it can be deleted later
        with context.temp_override(selected_editable_objects=[cp], active_object=cp, selected_objects=[cp], object=cp):
            bpy.ops.object.duplicates_make_real(use_base_parent=True, use_hierarchy=True)
            root_empty_after_split = bpy.data.objects[cp_name]  # this is the root empty after split
            # go over all children and  makr as tmp
            for child  in root_empty_after_split.children_recursive:
                child['tmp'] = True
                if child.type in ('CURVE', 'CURVES', 'FONT'):
                    CB_OT_CyclesBakeOps.obj_to_mesh(context, child, coll)
            return root_empty_after_split

    @staticmethod
    def get_phyllotaxis_offset(index, spacing=5.0):
        """Calculate phyllotaxis pattern offset for given index for even distribution of baked objects pairs."""
        ### TODO: expose spacing as a property
        golden_angle = 2.39996  # Approximately 137.5 degrees in radians
        r = spacing * math.sqrt(index)
        theta = index * golden_angle
        return Vector(( r * math.cos(theta), r * math.sin(theta), 0))

    def scene_copy(self, context, bj):
        orig_world = context.scene.world

        temp_scn = bpy.data.scenes.new("MD_TEMP")
        context.window.scene = temp_scn  # set new scene as active
        temp_scn.render.engine = "CYCLES"
        temp_scn.cycles.samples = 1 # main thing that affects bake speed..?
        temp_scn.cycles.use_adaptive_sampling = False
        temp_scn.cycles.device = 'GPU' if context.preferences.addons['cycles'].preferences.compute_device_type == 'CUDA' else 'CPU'
        temp_scn.cycles.sampling_pattern = 'BLUE_NOISE'
        temp_scn.cycles.max_bounces = 4
        temp_scn.cycles.caustics_reflective = False
        temp_scn.cycles.caustics_refractive = False
        temp_scn.cycles.transmission_bounces = 2
        temp_scn.cycles.transparent_max_bounces = 2

        temp_scn.world = orig_world  # copy world to temp scene

        # depsgraph = context.evaluated_depsgraph_get()
        context.view_layer.update()  # update depsgraph to get all objects evaluated

        cage_objs = []  # List to store cage objects
        lowpoly_objs = []
        out_hi_collection = bpy.data.collections.new('HIGHPOLY_MD_TMP')  # collection for highpoly objects
        temp_scn.collection.children.link(out_hi_collection)  # link highpoly collection to temp scene collection

        if self.bake_pair_index > -1 and self.bake_pair_index < len(bj.bake_pairs_list):
            active_pairs = [bj.bake_pairs_list[self.bake_pair_index]]
        else:
            active_pairs = [pair for pair in bj.bake_pairs_list if pair.activated]  # get only activated pairs


        addon_prefs = get_addon_preferences()

        for i,pair in enumerate(active_pairs):
            offset = self.get_phyllotaxis_offset(i, spacing=addon_prefs.pair_spacing_distance)

            low_obj = bpy.data.objects[pair.lowpoly]

            low_cp = low_obj.copy()
            low_cp.matrix_world.translation += offset
            lowpoly_objs.append(low_cp)  # To merge them later
            temp_scn.collection.objects.link(low_cp)  # unlink all other lowpoly objects

            cage = bpy.data.objects.get(pair.cage, None)
            if pair.use_cage and cage:
                # Copy existing cage object
                temp_cage = cage.copy()
                temp_cage.data = temp_cage.data.copy() # to not affect original cage object
                temp_cage['tmp'] = True  # mark as tmp, so it can be deleted later
                temp_cage.matrix_world.translation += offset
            else:
                # Create temporary cage by displacing vertices along normals
                temp_cage = low_cp.copy()
                temp_cage.data = low_cp.data.copy()  # copy mesh data to not affect original
                temp_cage['tmp'] = True  # mark as tmp, so it can be deleted later
                temp_cage.name = f"TEMP_CAGE_{low_cp.name}"
                dist = get_raycast_distance(bj, pair)
                displace_mod = add_split_extrude_mod(temp_cage, dist)

            temp_scn.collection.objects.link(temp_cage)
            with context.temp_override(selected_editable_objects=[temp_cage], active_object=temp_cage, selected_objects=[temp_cage]):
                bpy.ops.object.convert(target='MESH')
            cage_objs.append(temp_cage)


            if pair.hp_type == "GROUP":
                hi_collection = bpy.data.collections.get(pair.highpoly)
                for obj in hi_collection.objects:
                    cp = obj.copy()
                    cp['tmp'] = True  # mark as tmp, so it can be deleted later
                    out_hi_collection.objects.link(cp)  # link all objects from hipoly group to highpoly collection
                    cp.matrix_world.translation += offset  # move to offset position

                    # old 'manual' way > handled by Real Instances in gn
                    # if obj.type in ('CURVE', 'CURVES', 'FONT'):
                    #     hi_obj = self.obj_to_mesh(context, obj, out_hi_collection)
                    #     hi_obj.matrix_world.translation += offset
                    # elif obj.type == 'EMPTY' and obj.instance_collection:
                    #     root_empty = self.make_inst_real(context, obj, out_hi_collection)
                    #     root_empty.matrix_world.translation += offset
                    # else:
                    #     cp = obj.copy()
                    #     cp.matrix_world.translation += offset
                    #     cp['tmp'] = True  # mark as tmp, so it can be deleted later
                    #     out_hi_collection.objects.link(cp)

            else:  # pair.hp_type == "OBJ"
                hi_poly_obj = bpy.data.objects[pair.highpoly]
                cp = hi_poly_obj.copy()
                cp['tmp'] = True  # mark as tmp, so it can be deleted later
                out_hi_collection.objects.link(cp)  # link all objects from hipoly group to highpoly collection
                cp.matrix_world.translation += offset  # move to offset position

                # old 'manual' way > handled by Real Instances in gn
                # if hi_poly_obj.type == 'EMPTY' and hi_poly_obj.instance_collection:
                #     root_empty = self.make_inst_real(context, hi_poly_obj, out_hi_collection)
                #     root_empty.matrix_world.translation += offset
                # elif hi_poly_obj.type in ('CURVE', 'CURVES', 'FONT'):
                #     hi_obj = self.obj_to_mesh(context, hi_poly_obj, out_hi_collection)
                #     hi_obj.matrix_world.translation += offset
                # else:
                #     cp = hi_poly_obj.copy()
                #     cp.matrix_world.translation += offset
                #     cp['tmp'] = True  # mark as tmp, so it can be deleted later
                #     out_hi_collection.objects.link(cp)

        # create temp helper obj with no geometry, for geo nodes mod
        tmp_mesh = bpy.data.meshes.new("Tmp_MD_TMP")
        high_proxy = bpy.data.objects.new("HighProxy_MD_TMP", tmp_mesh)
        high_proxy['tmp'] = True  # mark as tmp, so it can be deleted later
        temp_scn.collection.objects.link(high_proxy)
        add_collection_to_mesh_mod(high_proxy, out_hi_collection)

        lowpoly_objs[0].data = lowpoly_objs[0].data.copy()
        lowpoly_objs[0].name = "LOWPOLY_MD_TMP"

        if len(lowpoly_objs) > 1:
            with bpy.context.temp_override(selected_editable_objects=lowpoly_objs, active_object=lowpoly_objs[0], selected_objects=lowpoly_objs):
                bpy.ops.object.join()

        # Merge cage objects if they exist
        if cage_objs:
            cage_objs[0].name = "CAGE_MD_TMP"
            if len(cage_objs) > 1:
                with bpy.context.temp_override(selected_editable_objects=cage_objs, active_object=cage_objs[0], selected_objects=cage_objs):
                    bpy.ops.object.join()


    def select_hi_low(self, bj):
        tmp_scn = bpy.data.scenes["MD_TEMP"]

        def select_obj(obj, select=True):
            obj.hide_render = not select
            obj.hide_set(not select)
            obj.select_set(select)

        for obj in tmp_scn.objects:
            obj.hide_viewport = False  # slow - since it unloads obj from memory, thus just reveal all
            select_obj(obj, False)

        # make selections, ensure visibility
        # for bakepass in bj.bake_pass_list:
        #     if bakepass.environment_group != "":  # bake enviro objects too
        #         if bakepass.environment_obj_vs_group == "GROUP": # XXX: add it
        #             for obj in bpy.data.collections[bakepass.environment_group].objects:
        #                 select_obj(obj)
        #         else:
        #             enviro_obj = tmp_scn.objects[bakepass.environment_group + "_MD_TMP"]
        #             select_obj(enviro_obj)

        # print("selected  enviro group " + pair.lowpoly)

        # high_coll = bpy.data.collections['HIGHPOLY_MD_TMP']
        # for obj in high_coll.objects:
        #     if obj.type == 'MESH':
        #         select_obj(obj)
        high_obj = tmp_scn.objects.get("HighProxy_MD_TMP")
        select_obj(high_obj)

        lowpoly_obj = tmp_scn.objects["LOWPOLY_MD_TMP"]
        select_obj(lowpoly_obj)
        tmp_scn.view_layers[0].objects.active = lowpoly_obj

        # XXX: restore  cage - in tmp scene setup
        # if pair.use_cage and pair.cage != "":
        #     cage_obj = tmp_scn.objects[pair.cage + "_MD_TMP"]
        #     select_obj(cage_obj)

    def bake_pair_pass(self, context, bake_job, bakepass):
        self.create_bake_mat_and_node()
        startTime = datetime.now()  # time debug
        scn = bpy.data.scenes["MD_TEMP"]
        low_obj = bpy.data.objects.get("LOWPOLY_MD_TMP")
        high_obj = bpy.data.objects.get("HighProxy_MD_TMP")
        if bakepass.pass_type == "AO":
            scn.cycles.samples = bakepass.samples
            scn.world.light_settings.distance = bakepass.ao_distance

        # front_dist = get_raycast_distance(bake_job)

        pass_type = bakepass.pass_type
        pass_components = {'NONE'}
        if bakepass.pass_type in ("DIFFUSE"):
            pass_components = {'COLOR'}
        elif bakepass.pass_type in ("AO_GN", "DEPTH" , "CURVATURE"):
            if bakepass.pass_type == "CURVATURE":
                set_curvature_mod(high_obj, bakepass)
            elif bakepass.pass_type == "DEPTH":
                set_depth_mod(high_obj, low_obj, bakepass)
            elif bakepass.pass_type == "AO_GN":
                set_ao_mod(high_obj, bakepass)
            attrib_mat = import_attrib_bake_mat()
            context.view_layer.material_override = attrib_mat
            # print(f"Overriding material for {bakepass.pass_type} pass with {attrib_mat.name}")
            # print(context.view_layer)
            context.view_layer.update()
            pass_type = 'DIFFUSE'
            pass_components = {'COLOR'}

        elif bakepass.pass_type == "OPACITY":
            pass_type = 'EMIT'
        elif bakepass.pass_type == 'COMBINED':
            pass_components = {'AO', 'EMIT', 'DIRECT', 'INDIRECT', 'COLOR', 'DIFFUSE', 'GLOSSY'}

        aa = int(bake_job.antialiasing)
        res = int(bake_job.bakeResolution)

        padding = bake_job.padding_size if bake_job.padding_mode == 'FIXED' else int(res/64)
        if bakepass.pass_type == "OPACITY":
            padding = 0

        # cage_obj = bpy.data.objects.get("CAGE_MD_TMP", None)
        bpy.ops.object.bake(type=pass_type, filepath="", pass_filter=pass_components,
                            width=res*aa, height=res*aa,
                            margin=padding*aa,
                            use_selected_to_active=True,
                            cage_extrusion=0,
                            normal_space=bakepass.nm_space,
                            normal_r="POS_X", normal_g=bakepass.nm_invert, normal_b='POS_Z',
                            save_mode='INTERNAL',
                            use_clear=False,
                            use_cage=True,
                            cage_object="CAGE_MD_TMP",
                            target='IMAGE_TEXTURES',
                            use_split_materials=False, use_automatic_name=False)

        print("Baking set " + bakepass.pass_type + "  time: " + str(datetime.now() - startTime))

        context.view_layer.material_override = None  # clear material override

    @staticmethod
    def remove_object(obj):
        mesh = None
        if obj.type == "MESH":
            mesh = obj.data

        # obj.user_clear()
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    @staticmethod
    def cleanup():
        scn_tmp = bpy.data.scenes["MD_TEMP"]
        for obj in scn_tmp.objects:
            if obj.get('tmp', False):
                CB_OT_CyclesBakeOps.remove_object(obj)

        low_obj = bpy.data.objects.get("LOWPOLY_MD_TMP")
        if low_obj:
            bake_mat = CB_OT_CyclesBakeOps.get_set_first_material_slot(low_obj)
            imgnode = bake_mat.node_tree.nodes.get('MDtarget')
            if imgnode:
                bake_mat.node_tree.nodes.remove(imgnode)  # remove bake image node
            CB_OT_CyclesBakeOps.remove_object(low_obj)  # remove lowpoly object

        node_groups_to_remove = [ "CB_AOPass", "CB_DepthPass", "CB_CurvaturePass", "CycBaker_SplitExtrude" ]
        for node_group_name in node_groups_to_remove:
            node_group = bpy.data.node_groups.get(node_group_name)
            if node_group:
                bpy.data.node_groups.remove(node_group)

        bake_mat = bpy.data.materials.get("CyclesBakeMat_MD_TEMP")
        if bake_mat:
            bpy.data.materials.remove(bake_mat, do_unlink=True)

        attrib_mat = bpy.data.materials.get("CBaker_AttribMaterial")
        if attrib_mat:
            bpy.data.materials.remove(attrib_mat, do_unlink=True)

        bpy.data.collections.remove(bpy.data.collections["HIGHPOLY_MD_TMP"], do_unlink=True)  # remove highpoly collection
        bpy.data.scenes.remove(scn_tmp)


    def assign_pink_mat(self, obj):
        pink_mat = bpy.data.materials.get("TMP_MissingMaterial")
        if obj.type == 'MESH':
            if len(obj.material_slots) == 0 or obj.material_slots[0].material is None:
                self.report({'WARNING'}, "Object: " + obj.name + " has no Material! Assigning pink mat")
                print("Object: " + obj.name + " has no Material! Assigning pink mat")
                obj.data.materials.append(pink_mat)

    # empty mat search function
    def is_empty_mat(self, context):
        pink_mat = bpy.data.materials.get("TMP_MissingMaterial")
        if not pink_mat:
            pink_mat = bpy.data.materials.new(name="TMP_MissingMaterial")
            pink_mat.diffuse_color = (1, 0.0, 0.2, 1.0)
            pink_mat.use_nodes = False


        self.checked_groups.clear()
        bake_settings = context.scene.cycles_baker_settings
        for bj in bake_settings.bake_job_queue:
            active_pair = [pair for pair in bj.bake_pairs_list if pair.activated]
            for pair in active_pair:
                if pair.highpoly != "":
                    if pair.hp_type == "GROUP":
                        for obj in bpy.data.collections[pair.highpoly].objects:
                            if obj.type == 'EMPTY' and obj.instance_collection:
                                continue
                            self.assign_pink_mat(obj)
                    else:
                        hipolyObj = bpy.data.objects.get(pair.highpoly)
                        if not hipolyObj:
                            print("No highpoly " + pair.highpoly + " object on scene found! Cancelling")
                            pair.activated = False
                            continue
                        if hipolyObj.type == 'EMPTY' and hipolyObj.instance_collection:
                            emptyMatInGroup = self.checkEmptyMaterialForGroup(hipolyObj, context)

                        self.assign_pink_mat(hipolyObj)
                else:  # if highpoly empty
                    print("No highpoly defined. Disabling pair")
                    pair.activated = False
                    continue
                if pair.lowpoly == "":  # if lowpoly empty
                    print("No highpoly defined. Disabling pair")
                    pair.activated = False
                    continue
                low = bpy.data.objects.get(pair.lowpoly)
                if not low:
                    print("No lowpoly " + pair.lowpoly + " object on scene found! Disabling pair")
                    pair.activated = False
                    continue


    checked_groups = []

    def checkEmptyMaterialForGroup(self, empty, context):
        if empty.instance_collection.name in self.checked_groups:
            # print(empty.instance_collection.name + " was already checked for empty mat. Skipping!")
            return False
        for obj in bpy.data.collections[empty.instance_collection.name].objects:
            if obj.instance_collection and obj.type == 'EMPTY':
                return self.checkEmptyMaterialForGroup(obj, context)
            elif obj.type == "MESH" and (len(obj.material_slots) == 0 or obj.material_slots[0].material is None) and not obj.hide_render:
                self.assign_pink_mat(obj)
                self.checked_groups.append(empty.instance_collection.name)  # add to check list if there was obj with empty mat
                return True
        self.checked_groups.append(empty.instance_collection.name)  # or no empty mat in group
        return False


    def execute(self, context):
        TotalTime = datetime.now()
        cycles_bake_settings = context.scene.cycles_baker_settings
        if self.is_empty_mat(context):
            return {'CANCELLED'}

        wm = context.window_manager

        active_bj = [bj for bj in cycles_bake_settings.bake_job_queue if bj.activated]

        total_steps = sum(len([p for p in bj.bake_pass_list if p.activated]) for bj in active_bj)
        wm.progress_begin(0, total_steps)
        current_step = 0
        for bj in active_bj:
            # ensure save path exists
            if not os.path.exists(bpy.path.abspath(bj.output)):
                os.makedirs(bpy.path.abspath(bj.output))


            for pair in bj.bake_pairs_list:  # disable hipoly lowpoly pairs that are not defined
                if pair.lowpoly == "" or not bpy.data.objects.get(pair.lowpoly):
                    self.report({'INFO'}, 'Lowpoly not found ' + pair.lowpoly)
                    pair.activated = False
                if pair.highpoly == "" or (pair.hp_type=="OBJ" and  not bpy.data.objects.get(pair.highpoly)) or (pair.hp_type=="GROUP" and  not bpy.data.collections.get(pair.highpoly)):
                    self.report({'INFO'}, 'highpoly not found ' + pair.highpoly)
                    pair.activated = False

            self.scene_copy(context, bj)  # we export temp scene copy

            aa = int(bj.antialiasing)
            img_res = int(bj.bakeResolution)
            # padding = bj.padding_size if bj.padding_mode == 'FIXED' else int(img_res/64)

            active_bake_passes = [bakepass for bakepass in bj.bake_pass_list if bakepass.activated and len(bj.bake_pass_list) > 0 and len(bj.bake_pairs_list) > 0]

            for bakepass in active_bake_passes:
                wm.progress_update(current_step)
                current_step += 1
                self.report({'INFO'}, f"Baking {bakepass.pass_type} ({current_step}/{total_steps})")
                render_target = bpy.data.images.new("MDtarget", width=img_res*aa, height=img_res*aa, alpha=True, float_buffer=False)
                bg =  BG_color[bakepass.pass_type]
                if self.bake_pair_index != -1: # set alpha to 0 for single pair bake
                    bg[3] = 0.0

                render_target.generated_color = bg
                self.select_hi_low(bj)
                self.bake_pair_pass(context, bj, bakepass)

                if bakepass.pass_type != "OPACITY":  # opacity is not saved to image
                    pass
                    # add_padding_offscreen(render_target, img_res*aa, img_res*aa, padding_size=aa*padding)
                else: # copy alpha to color
                    pixels = np.array(render_target.pixels, dtype='f')
                    alpha = pixels.reshape(-1, 4)[:, 3]  # get alpha channel
                    # duplicate alpha to RGB channels and recombine
                    channel = np.repeat(alpha[:, np.newaxis], 3, axis=1)  # make RGB channels
                    render_target.pixels = np.concatenate((channel, alpha[:, np.newaxis]), axis=1).ravel()

                render_target.scale(img_res, img_res)
                imgPath = str(bj.get_out_dir_path() / f"{bakepass.get_filename(bj)}.png")

                single_pair_bake = self.bake_pair_index > -1
                if single_pair_bake and os.path.exists(imgPath): # for bake of single pair  was enabled:
                    old_img = bpy.data.images.load(filepath=imgPath, check_existing=True)
                    if old_img and old_img.size[0]==old_img.size[1]==img_res: # mix with old bake if exists
                        old_img.reload()
                        old_pixels = np.array(old_img.pixels, dtype='f')
                        new_pixels = np.array(render_target.pixels, dtype='f')

                        # reshape to (width*height, 4) - assign the result back to variables
                        old_pixels = old_pixels.reshape(-1, 4)
                        new_pixels = new_pixels.reshape(-1, 4)

                        # get alpha channel from new pixels
                        new_alpha = new_pixels[:, 3:4]  # keep as column vector for broadcasting
                        blended_pixels = new_pixels * new_alpha + old_pixels * (1 - new_alpha)

                        render_target.pixels = blended_pixels.ravel()  # set mixed pixels to render target

                render_target.filepath_raw = imgPath
                render_target.save()

                # render_target.user_clear()
                bpy.data.images.remove(render_target)

                # if path.isfile(imgPath):  # load bake from disk
                img_users = (img for img in bpy.data.images if abs_file_path(img.filepath) == imgPath)
                if img_users:
                    for img in img_users:
                        img.reload()  # reload done in baking
                else:
                    img = bpy.data.images.load(filepath=imgPath)

            self.cleanup()  # delete scene
            wm.progress_end()


        print(f"Cycles Total baking time: {(datetime.now() - TotalTime).seconds} sec")
        addon_prefs = get_addon_preferences()
        if addon_prefs.play_finish_sound:
            self.playFinishSound()

        return {'FINISHED'}

    @staticmethod
    def playFinishSound():
        script_file = os.path.realpath(__file__)
        directory = os.path.dirname(script_file)
        sound_path = os.path.join(directory, "finished.mp3")

        device = aud.Device()
        sound = aud.Sound(sound_path)

        handle = device.play(sound)

        # stop the sounds (otherwise they play until their ends)
        # handle.stop()


OLD_SHADING = None
OLD_RENDER_PASS = None

class CB_OT_PreviewPassOps(bpy.types.Operator):
    bl_idname = "cycles.preview_pass"
    bl_label = "Preview Pass"
    bl_description = "Preview selected pass type on highpoly objects"
    bl_options = {'REGISTER', 'UNDO'}

    pass_type: bpy.props.EnumProperty(name="Pass Type",
                                      items=[
                                          ('CURVATURE', "Curvature", "Preview Curvature Pass"),
                                          ('DEPTH', "Depth", "Preview Depth Pass"),
                                          ('AO_GN', "AO (Geometry Nodes)", "Preview AO Pass with Geometry Nodes"),
                                      ],
                                      default='CURVATURE',
                                      description="Type of pass to preview")
    job_index: bpy.props.IntProperty(name="Bake Job Index", default=-1, description="Index of the bake job to preview")
    pass_index: bpy.props.IntProperty(name="Bake Pass Index", default=-1, description="Index of the bake pass to preview")
    orig_scene_name: bpy.props.StringProperty(name="Original Scene Name", default="", description="Name of the original scene before preview")

    def execute(self, context):
        cycles_bake_settings = context.scene.cycles_baker_settings
        active_bj = cycles_bake_settings.bake_job_queue[self.job_index]
        active_pairs = [pair for pair in active_bj.bake_pairs_list if pair.activated]
        if not active_pairs:
            self.report({'WARNING'}, "No active low/high-poly bake pairs found for preview.")
            return {'CANCELLED'}

        # Create temp scene
        print("Creating preview \"MD_PREVIEW\" scene for pass: " + self.pass_type)
        temp_scn = bpy.data.scenes.new("MD_PREVIEW")
        context.window.scene = temp_scn
        temp_scn.render.engine = "CYCLES"
        temp_scn.cycles.device = 'GPU' if context.preferences.addons['cycles'].preferences.compute_device_type == 'CUDA' else 'CPU'
        temp_scn.world = context.scene.world

        # Create collection for highpoly objects
        out_hi_collection = bpy.data.collections.new('HIGHPOLY_PREVIEW')
        temp_scn.collection.children.link(out_hi_collection)

        lowpoly_objs = []
        # Copy highpoly objects from all active bake pairs
        for pair in active_pairs:
            if self.pass_type == "DEPTH":
                low_obj = bpy.data.objects.get(pair.lowpoly)
                low_cp = low_obj.copy()
                lowpoly_objs.append(low_cp)  # To merge them later
                temp_scn.collection.objects.link(low_cp)  # link all lowpoly objects to temp scene

            if pair.hp_type == "GROUP":
                hi_collection = bpy.data.collections.get(pair.highpoly)
                for obj in hi_collection.objects:
                    cp = obj.copy()
                    # cp['tmp'] = True
                    out_hi_collection.objects.link(cp)
            else:  # pair.hp_type == "OBJ"
                hi_poly_obj = bpy.data.objects[pair.highpoly]
                cp = hi_poly_obj.copy()
                # cp['tmp'] = True
                out_hi_collection.objects.link(cp)

        # Create proxy object with geometry nodes
        tmp_mesh = bpy.data.meshes.get("Tmp_Preview")
        if not tmp_mesh:
            tmp_mesh = bpy.data.meshes.new("Tmp_Preview")
        high_proxy = bpy.data.objects.new("HighProxy_Preview", tmp_mesh)
        # high_proxy['tmp'] = True
        temp_scn.collection.objects.link(high_proxy)

        # Add collection to mesh modifier
        add_collection_to_mesh_mod(high_proxy, out_hi_collection)

        out_hi_collection.hide_viewport = True

        # Add pass-specific modifier and material override
        attrib_mat = import_attrib_bake_mat()
        context.view_layer.material_override = attrib_mat

        if lowpoly_objs:
            lowpoly_objs[0].name = "LowProxy_Preview"

            if len(lowpoly_objs) > 1:
                lowpoly_objs[0].data = lowpoly_objs[0].data.copy() # do not affect original lowpoly object
                with bpy.context.temp_override(selected_editable_objects=lowpoly_objs, active_object=lowpoly_objs[0], selected_objects=lowpoly_objs):
                    bpy.ops.object.join()
            lowpoly_objs[0].display_type = 'BOUNDS'
            lowpoly_objs[0].hide_viewport = True

        # Add specific pass modifier based on pass_type
        bake_pass = active_bj.bake_pass_list[self.pass_index]
        if self.pass_type == "CURVATURE":
            set_curvature_mod(high_proxy, bake_pass)
        elif self.pass_type == "DEPTH":
            set_depth_mod(high_proxy, lowpoly_objs[0], bake_pass)
        elif self.pass_type == "AO_GN":
            set_ao_mod(high_proxy, bake_pass)

        # set areas[2].spaces[0].shading.type to Rendred, and preview only Color from diffuse pass

        global OLD_SHADING, OLD_RENDER_PASS
        OLD_SHADING = context.space_data.shading.type
        context.space_data.shading.type = 'RENDERED'
        OLD_RENDER_PASS = context.space_data.shading.render_pass
        context.space_data.shading.cycles.render_pass = 'DIFFUSE_COLOR'

        temp_scn['preview_bj_idx'] = self.job_index
        temp_scn['preview_pass_idx'] = self.pass_index
        temp_scn['orig_scene_name'] = self.orig_scene_name  # store original scene name

        temp_scn.view_settings.view_transform = 'Standard'  # set view transform to Standard
        temp_scn.cycles.preview_samples = 1
        temp_scn.cycles.use_adaptive_sampling = False

        return {'FINISHED'}

class CB_OT_ClosePreviewOps(bpy.types.Operator):
    bl_idname = "cycles.close_preview"
    bl_label = "Close Preview"
    bl_description = "Close preview scene and return to original scene"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        preview_scene = bpy.data.scenes.get("MD_PREVIEW")
        if preview_scene:
            # Clear material override
            context.view_layer.material_override = None

            # Remove temporary objects
            for obj in preview_scene.objects:
                bpy.data.objects.remove(obj)

            # Remove collections
            preview_collection = bpy.data.collections.get("HIGHPOLY_PREVIEW")
            if preview_collection:
                bpy.data.collections.remove(preview_collection, do_unlink=True)

            # Remove materials
            attrib_mat = bpy.data.materials.get("CBaker_AttribMaterial")
            if attrib_mat:
                bpy.data.materials.remove(attrib_mat, do_unlink=True)

            # Remove preview scene
            original_scene = bpy.data.scenes.get("Scene")
            if original_scene:
                context.window.scene = original_scene
            bpy.data.scenes.remove(preview_scene)

        context.space_data.shading.type = OLD_SHADING if OLD_SHADING else 'MATERIAL'
        context.space_data.shading.cycles.render_pass = OLD_RENDER_PASS if OLD_RENDER_PASS else 'DIFFUSE_COLOR'

        return {'FINISHED'}

# cleanup operator (for CB_OT_CyclesBakeOps.cleanup())
class CB_OT_CleanupCyclesBake(bpy.types.Operator):
    bl_idname = "cycles.cleanup_cycles_bake"
    bl_label = "Cleanup Cycles Bake"
    bl_description = "Cleanup temporary objects and materials created during Cycles baking - useful for cleanup after failed bake"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        CB_OT_CyclesBakeOps.cleanup()
        return {'FINISHED'}
