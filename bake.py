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
from .utils import abs_file_path, get_addon_preferences, add_geonodes_mod, import_mat, load_baked_images, clear_parent

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

node_groups_to_remove = [ "CB_AOPass", "CB_DepthPass", "CB_CurvaturePass", "CycBaker_SplitExtrude", "CB_CollectionToMesh" ]

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
    curvature_mod['Socket_6'] = pass_settings.curvature_brightness # Brightness
    curvature_mod['Socket_3'] = pass_settings.curvature_blur     # Blur
    obj.modifiers.update()  # Update modifier to apply changes
    obj.update_tag()
    return curvature_mod

def import_attrib_bake_mat():
    bake_mat = import_mat("CBaker_AttribMaterial")
    return bake_mat


def get_raycast_distance(low_poly):
    objBBoxSize = 0.2*Vector(low_poly.dimensions[:]).length
    return objBBoxSize

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
Loops = None
Indices = None
Loop_to_Vert_id = None
Verts_co = None

def draw_cage_callback(bake_pair, context):
    low_poly = bpy.data.objects.get(bake_pair.lowpoly)
    if not bake_pair.draw_front_dist or not bake_pair.lowpoly or not low_poly:
        return
    global Verts, Loops, Indices, Loop_to_Vert_id, Verts_co
    if low_poly.type == 'MESH' and context.mode == 'OBJECT':
        ray_dist = bake_pair.ray_dist * get_raycast_distance(low_poly)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = low_poly.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()

        mesh.calc_loop_triangles()
        loop_count = len(mesh.loops)
        tri_count = len(mesh.loop_triangles)

        # Reallocate arrays if mesh changed
        if Loops is None or Loops.shape[0] != loop_count:
            Loops = np.empty((loop_count, 3), 'f')  # Normal per loop
            Verts = np.empty((loop_count, 3), 'f')  # One vertex per loop
            Verts_co = np.empty((loop_count, 3), 'f')  # One vertex per loop
            Indices = np.empty((tri_count, 3), 'i')
            Loop_to_Vert_id = np.empty(loop_count, dtype=np.int32)
        #
        # # Get vertex positions for each loop
        # for i, loop in enumerate(mesh.loops):
        #     Verts[i] = mesh.vertices[loop.vertex_index].co
        #     Loops[i] = mesh.corner_normals[i].vector  # Use corner_normals instead of loop.normal
        #
        # # Get triangle indices (need to remap to loop indices)
        # for i, tri in enumerate(mesh.loop_triangles):
        #     Indices[i] = tri.loops

        mesh.loops.foreach_get('vertex_index', Loop_to_Vert_id)
        mesh.corner_normals.foreach_get('vector', Loops.ravel())
        mesh.vertices.foreach_get('co', Verts_co.ravel())
        mesh.loop_triangles.foreach_get('loops', Indices.ravel()) # triangle loops indices

        Verts = Verts_co[Loop_to_Vert_id] # loop will point to vertex co

        # Extrude along split normals
        Verts = Verts + Loops * ray_dist

        gpu.state.blend_set('ALPHA')
        gpu.state.face_culling_set('BACK')

        face_color = (0, 0.8, 0, 0.5) if bake_pair.draw_front_dist else (0.8, 0, 0, 0.5)

        with gpu.matrix.push_pop():
            gpu.matrix.multiply_matrix(low_poly.matrix_world)
            shader_uniform.bind()
            shader_uniform.uniform_float("color", face_color)
            batch = batch_for_shader(shader_uniform, 'TRIS', {"pos": Verts}, indices=Indices)
            batch.draw(shader_uniform)

        # restore gpu defaults
        gpu.state.blend_set('NONE')
        gpu.state.face_culling_set('NONE')

        # old way to transform vertices with matrix
        # coords_4d = np.ones((vert_count, 4), 'f')
        # coords_4d[:, :-1] = Vertices
        # vertices = np.einsum('ij,aj->ai', mat_np, coords_4d)[:, :-1]


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


def is_htool_installed(context):
    return bool(bpy.context.preferences.addons.get('hair_tool'))


def ht_channel_mixing(context, bj):
    ''' Create compositing node tree and load baked imgs'''
    if not is_htool_installed: # no Hair Tool - no channel mixing
        print("Hair Toon addon is not installed. Cannot create channel mixing nodes.")
        return

    mix_name = 'CyclesBaker_ChannelMixing'
    if mix_name not in bpy.data.node_groups.keys():
        bpy.ops.node.new_node_tree(type='TextureChannelMixing', name=mix_name)

    node_tree = bpy.data.node_groups.get(mix_name)
    if node_tree is None:
        print(f"Node tree '{mix_name}' not found. Cannot create channel mixing nodes.")
        return
    node_tree.restart_node_tree()
    # addon_prefs = get_addon_preferences()

    bake_images = load_baked_images(bj)
    for i, (pass_img, bpass) in enumerate(zip(bake_images, bj.bake_pass_list)):
        if not pass_img:
            continue  # skip if no image for this pass
        # imgPath = str(bj.get_out_dir_path() / f"{bakepass.get_filename(bj)}.png")

        img_node = node_tree.nodes.get(bpass.pass_type)
        if not img_node:  # create new node if not found
            img_node = node_tree.nodes.new('TextureMixInputTexture')
            img_node.name = bpass.pass_type
        pass_img.preview_ensure() # or else it will throw error - no:  img.preview.icon_id
        img_node.img = pass_img  # also refreshes img...
        img_node.location[1] = i * 200
        img_node.width = 220

        # bpy.data.images.remove(img)
    # outputImg.filepath_raw = bpy.path.abspath(bake_settings.hair_bake_path + "Hair_compo." + ext)


class CB_OT_CyclesBakeOps(bpy.types.Operator):
    bl_idname = "cycles.bake"
    bl_label = "Bake Objects"
    bl_description = "Bake selected pairs of highpoly-lowpoly objects using blender Bake 'Selected to Active' feature"
    bl_options = {'REGISTER', 'UNDO'}

    bake_pair_index: bpy.props.IntProperty(name="Bake Pair Index", default=-1, description="Index of the bake pair to process")


    @classmethod
    def description(cls, context, properties):
        if properties.bake_pair_index > -1:
            return "Bake ONLY selected pair: of highpoly-lowpoly objects (then compose bake onto existing, previously baked textures)"
        else:
            return "Bake selected pairs of highpoly-lowpoly objects using blender Bake 'Selected to Active' feature"

    @staticmethod
    def get_set_first_material_slot(obj, mat=None):
        """Get or set the first material slot for an object.

        Args:
            obj: The object to process
            mat: Optional material to assign to first slot. If None, returns existing or creates new.

        Returns:
            The material in the first slot
        """
        while len(obj.material_slots) > 1:  # remove mat slots above 1 (there is risk low mesh parts with it won't be baked)
            obj.data.materials.pop(index=-1)

        # Create slot if needed
        if len(obj.material_slots) == 0:
            obj.data.materials.append(None)

        if mat:  # If material specified, assign it
            obj.material_slots[0].material = mat
            mat.use_nodes = True
            return mat

        # Otherwise handle existing or create new
        first_slot_mat = obj.material_slots[0].material
        if first_slot_mat:
            first_slot_mat.use_nodes = True
            return first_slot_mat

        # Create new material if needed
        low_bake_mat = bpy.data.materials.get("CyclesBakeMat_MD_TEMP")
        if low_bake_mat is None:
            low_bake_mat = bpy.data.materials.new(name="CyclesBakeMat_MD_TEMP")
        low_bake_mat.use_nodes = True

        obj.material_slots[0].material = low_bake_mat
        return low_bake_mat

    @staticmethod
    def create_bake_mat_and_node():
        # both low and proxy low should have the same bake material
        proxy_low = bpy.data.objects.get("LowProxy_MD_TMP")
        low_collection = bpy.data.collections.get("LOWPOLY_MD_TMP")
        first_low = low_collection.objects[0]
        bake_mat = CB_OT_CyclesBakeOps.get_set_first_material_slot(first_low)

        # assign first_mat to proxy tol
        CB_OT_CyclesBakeOps.get_set_first_material_slot(proxy_low, bake_mat)

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
        """Create temporary scene with proxy objects for baking.

        Args:
            context: Blender context
            bj: Bake job containing pairs to process
        """
        # Scene setup
        temp_scn = bpy.data.scenes.new("MD_TEMP")
        temp_scn.world = context.scene.world
        context.window.scene = temp_scn

        # Configure render settings
        temp_scn.render.engine = "CYCLES"
        cycles = temp_scn.cycles
        cycles.samples = 1
        cycles.use_adaptive_sampling = False
        cycles.device = 'GPU' if context.preferences.addons['cycles'].preferences.compute_device_type == 'CUDA' else 'CPU'
        cycles.sampling_pattern = 'BLUE_NOISE'
        cycles.max_bounces = 4
        cycles.caustics_reflective = cycles.caustics_refractive = False
        cycles.transmission_bounces = cycles.transparent_max_bounces = 2

        # Create collections once
        collections = {
            'high': bpy.data.collections.new('HIGHPOLY_MD_TMP'),
            'low': bpy.data.collections.new('LOWPOLY_MD_TMP'),
            'cage': bpy.data.collections.new('CAGE_MD_TMP')
        }

        for coll in collections.values():
            temp_scn.collection.children.link(coll)

        # Get active pairs
        if self.bake_pair_index > -1 and self.bake_pair_index < len(bj.bake_pairs_list):
            active_pairs = [bj.bake_pairs_list[self.bake_pair_index]]
        else:
            active_pairs = [pair for pair in bj.bake_pairs_list if pair.activated]  # get only activated pairs

        # Calculate grid layout
        spacing = get_addon_preferences().pair_spacing_distance
        square_cnt = math.ceil(len(active_pairs)**0.5)
        grid_center = (square_cnt - 1) * spacing * Vector((0.5, 0.5, 0))
        deps = context.evaluated_depsgraph_get()

        def create_copy(obj, collection, copy_data=False):
            """Helper to create and setup object copy."""
            cp = obj.copy()
            if copy_data and obj.data:
                cp.data = obj.data.copy()
            cp['tmp'] = True
            clear_parent(cp)
            collection.objects.link(cp)
            return cp

        # Process each pair
        for i, pair in enumerate(active_pairs):
            # Calculate offset
            offset = spacing * Vector((i % square_cnt, i // square_cnt, 0)) - grid_center

            # Track average position
            positions = []

            # Process lowpoly
            low_obj = bpy.data.objects[pair.lowpoly]
            low_cp = create_copy(low_obj, collections['low'])
            positions.append(low_obj.evaluated_get(deps).matrix_world.translation)

            # Process cage
            if pair.use_cage and (cage := bpy.data.objects.get(pair.cage)):
                temp_cage = create_copy(cage, collections['cage'], copy_data=True)
            else:
                temp_cage = create_copy(low_cp, collections['cage'], copy_data=True)
                temp_cage.name = f"TEMP_CAGE_{low_cp.name}"
                ray_dist = pair.ray_dist * get_raycast_distance(low_obj)
                # print(f"Baking cage for {pair.lowpoly} with ray distance: {ray_dist}")
                add_split_extrude_mod(temp_cage, ray_dist)

            # Convert cage to mesh
            # with context.temp_override(selected_editable_objects=[temp_cage], active_object=temp_cage, selected_objects=[temp_cage]):
            #     bpy.ops.object.convert(target='MESH')

            # Create highpoly subcollection
            hi_subcoll = bpy.data.collections.new(f'HIGH{i+1}_TMP')
            hi_subcoll['tmp'] = True
            collections['high'].children.link(hi_subcoll)

            # Process highpoly objects
            source = (bpy.data.collections[pair.highpoly].objects
                     if pair.hp_type == "GROUP"
                     else [bpy.data.objects[pair.highpoly]])

            hi_objs = []
            for obj in source:
                cp = create_copy(obj, hi_subcoll)
                hi_objs.append(cp)
                positions.append(obj.evaluated_get(deps).matrix_world.translation)

            # Apply offset from average position
            avg_pos = sum(positions, Vector((0,0,0))) / len(positions)
            translation = offset - avg_pos
            for obj in [low_cp, temp_cage] + hi_objs:
                obj.matrix_world.translation += translation

        # Create proxy objects for each collection
        for name, collection in collections.items():
            mesh = bpy.data.meshes.new(f"Tmp_{name}_MD_TMP")
            if name == 'low':   # lowpoly has to have at least one quad - or else it will not bake
                mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2), (2, 3, 0)])
                # and placeholder uvs
                mesh.uv_layers.new(name="UVMap")
            proxy = bpy.data.objects.new(f"{name.capitalize()}Proxy_MD_TMP", mesh)
            proxy['tmp'] = True
            temp_scn.collection.objects.link(proxy)
            add_collection_to_mesh_mod(proxy, collection)

            layer_collection = context.view_layer.layer_collection.children.get(collection.name)
            layer_collection.hide_viewport = True  # This controls the 'eye' icon in the outliner

        return temp_scn


    def select_hi_low(self, bj):
        tmp_scn = bpy.data.scenes["MD_TEMP"]

        def select_obj(obj, select=True):
            obj.hide_render = not select
            obj.hide_set(not select)
            obj.select_set(select)

        for obj in tmp_scn.objects:
            # obj.hide_viewport = False  # slow - since it unloads obj from memory, thus just reveal all
            select_obj(obj, False)

        # Select and show proxy objects
        high_proxy = tmp_scn.objects["HighProxy_MD_TMP"]
        low_proxy = tmp_scn.objects["LowProxy_MD_TMP"]
        # cage_proxy = tmp_scn.objects["CageProxy_MD_TMP"]

        select_obj(high_proxy)
        select_obj(low_proxy)
        tmp_scn.view_layers[0].objects.active = low_proxy


    def bake_pair_pass(self, context, bake_job, bakepass):
        self.create_bake_mat_and_node()
        startTime = datetime.now()  # time debug
        scn = bpy.data.scenes["MD_TEMP"]
        low_proxy = scn.objects["LowProxy_MD_TMP"]
        high_proxy = scn.objects["HighProxy_MD_TMP"]
        cage_proxy = scn.objects["CageProxy_MD_TMP"]

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
                set_curvature_mod(high_proxy, bakepass)
            elif bakepass.pass_type == "DEPTH":
                set_depth_mod(high_proxy, low_proxy, bakepass)
            elif bakepass.pass_type == "AO_GN":
                set_ao_mod(high_proxy, bakepass)
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
                            cage_object=cage_proxy.name,
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
        tmp_scn = bpy.data.scenes["MD_TEMP"]
        if not tmp_scn:
            return

        # low_obj = bpy.data.objects.get("LowProxy_MD_TMP")
        low_collection = bpy.data.collections.get("LOWPOLY_MD_TMP")
        if low_collection:
            low_obj = low_collection.objects[0] # first obj has to has proper mat
            if low_obj:
                bake_mat = CB_OT_CyclesBakeOps.get_set_first_material_slot(low_obj)
                imgnode = bake_mat.node_tree.nodes.get('MDtarget')
                if imgnode:
                    bake_mat.node_tree.nodes.remove(imgnode)  # remove bake image node

        for obj in tmp_scn.objects:
            if obj.get('tmp', False):
                CB_OT_CyclesBakeOps.remove_object(obj)


        # Remove all collections marked as temporary
        for coll in bpy.data.collections:
            if coll.get('tmp', False):
                bpy.data.collections.remove(coll, do_unlink=True)

        # Remove main collections
        for coll_name in ('HIGHPOLY_MD_TMP', 'LOWPOLY_MD_TMP', 'CAGE_MD_TMP'):
            if coll := bpy.data.collections.get(coll_name):
                bpy.data.collections.remove(coll, do_unlink=True)

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

        bpy.data.scenes.remove(tmp_scn)


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
        active_bjobs = [bj for bj in cycles_bake_settings.bake_job_queue if bj.activated]
        if len(active_bjobs) == 0:
            self.report({'ERROR'}, "No bake jobs defined. Please add bake job first.")
            return {'CANCELLED'}
        if self.is_empty_mat(context):
            return {'CANCELLED'}

        disable_all_cages_drawing(context)  # disable all cages drawing before preview

        total_steps = sum(len([p for p in bj.bake_pass_list if p.activated]) for bj in active_bjobs)
        wm = context.window_manager
        wm.progress_begin(0, total_steps)
        current_step = 0
        for bj in active_bjobs:
            if len(bj.bake_pairs_list) == 0:
                self.report({'ERROR'}, "No bake pairs defined in job: " + bj.name + ". Please add bake pairs first.")
                wm.progress_end()
                continue
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
                render_target = bpy.data.images.get("MDtarget")
                if render_target:
                    if render_target.size[0] != img_res*aa or render_target.size[1] != img_res*aa:
                        render_target.scale(img_res*aa, img_res*aa)
                else:
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
                bpy.data.images.remove(render_target, do_unlink=True)

                # if path.isfile(imgPath):  # load bake from disk
                img_users = (img for img in bpy.data.images if abs_file_path(img.filepath) == imgPath)
                if img_users:
                    for img in img_users:
                        img.reload()  # reload done in baking
                else:
                    img = bpy.data.images.load(filepath=imgPath)

            self.cleanup()  # delete scene
            wm.progress_end()

            if is_htool_installed(context) and bj.use_channel_packing:
                ht_channel_mixing(context, bj)  # create channel mixing nodes for hair toon

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
handleDrawRayDistance = []
def draw_cage_handle(context, bake_pair):
    global handleDrawRayDistance
    if bake_pair.draw_front_dist:
        # disable all other draw_front_dist
        for bj in context.scene.cycles_baker_settings.bake_job_queue: # disable all draw_front_dist
            for pair in bj.bake_pairs_list:
                if pair != bake_pair:
                    pair['draw_front_dist'] = False

        if handleDrawRayDistance: # remove - so we can fire draw_cage_callback again with current pair
            for handle in handleDrawRayDistance:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
            handleDrawRayDistance.clear()

        args = (bake_pair, context)  # u can pass arbitrary class as first param  Instead of (self, context)
        handleDrawRayDistance.append(bpy.types.SpaceView3D.draw_handler_add(draw_cage_callback, args, 'WINDOW', 'POST_VIEW'))
    else:

        if handleDrawRayDistance:
            for handle in handleDrawRayDistance:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
            handleDrawRayDistance.clear()

def disable_all_cages_drawing(context):
    """Disable drawing of all cages in the scene."""
    for bj in context.scene.cycles_baker_settings.bake_job_queue:
        for pair in bj.bake_pairs_list:
            pair.draw_front_dist = False  # this will disable cage drawing
    context.view_layer.update()  # update view layer to apply changes

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

        disable_all_cages_drawing(context)  # disable all cages drawing before preview

        # Create temp scene
        print(f"Creating preview \"MD_PREVIEW\" scene for pass: {self.pass_type}")
        temp_scn = bpy.data.scenes.new("MD_PREVIEW")
        context.window.scene = temp_scn
        temp_scn.render.engine = "CYCLES"
        temp_scn.cycles.device = 'GPU' if context.preferences.addons['cycles'].preferences.compute_device_type == 'CUDA' else 'CPU'
        temp_scn.world = context.scene.world

        # Create main collections
        collections = {
            'high': bpy.data.collections.new('HIGHPOLY_PREVIEW'),
            'low': bpy.data.collections.new('LOWPOLY_PREVIEW'),
            'cage': bpy.data.collections.new('CAGE_PREVIEW')
        }

        for coll in collections.values():
            coll['tmp'] = True
            temp_scn.collection.children.link(coll)

        def create_copy(obj, collection, copy_data=False):
            """Helper to create and setup object copy."""
            cp = obj.copy()
            if copy_data and obj.data:
                cp.data = cp.data.copy()
            cp['tmp'] = True
            clear_parent(cp)
            collection.objects.link(cp)
            return cp

        # Process each pair
        for i, pair in enumerate(active_pairs):
            # Process lowpoly
            if self.pass_type == "DEPTH":
                low_obj = bpy.data.objects[pair.lowpoly]
                low_cp = create_copy(low_obj, collections['low'])

            # Process highpoly
            hi_subcoll = bpy.data.collections.new(f'HIGH{i+1}_PREVIEW')
            hi_subcoll['tmp'] = True
            collections['high'].children.link(hi_subcoll)

            if pair.hp_type == "GROUP":
                hi_collection = bpy.data.collections.get(pair.highpoly)
                for obj in hi_collection.objects:
                    create_copy(obj, hi_subcoll)
            else:  # pair.hp_type == "OBJ"
                hi_poly_obj = bpy.data.objects[pair.highpoly]
                create_copy(hi_poly_obj, hi_subcoll)

        # Create proxy objects
        for name, collection in collections.items():
            mesh = bpy.data.meshes.new(f"Tmp_{name}_Preview")
            if name == 'low' and self.pass_type == "DEPTH":
                mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2), (2, 3, 0)])
                mesh.uv_layers.new(name="UVMap")
            proxy = bpy.data.objects.new(f"{name.capitalize()}Proxy_Preview", mesh)
            proxy['tmp'] = True
            temp_scn.collection.objects.link(proxy)
            add_collection_to_mesh_mod(proxy, collection)

            layer_collection = context.view_layer.layer_collection.children.get(collection.name)
            if layer_collection:
                layer_collection.hide_viewport = True

        # Get proxy references
        high_proxy = temp_scn.objects["HighProxy_Preview"]
        low_proxy = temp_scn.objects.get("LowProxy_Preview")

        # Add pass-specific modifier and material override
        attrib_mat = import_attrib_bake_mat()
        context.view_layer.material_override = attrib_mat

        # Add specific pass modifier based on pass_type
        bake_pass = active_bj.bake_pass_list[self.pass_index]
        if self.pass_type == "CURVATURE":
            set_curvature_mod(high_proxy, bake_pass)
        elif self.pass_type == "DEPTH":
            set_depth_mod(high_proxy, low_proxy, bake_pass)
        elif self.pass_type == "AO_GN":
            set_ao_mod(high_proxy, bake_pass)

        # Store old shading settings and set new ones
        global OLD_SHADING, OLD_RENDER_PASS
        OLD_SHADING = context.space_data.shading.type
        context.space_data.shading.type = 'RENDERED'
        OLD_RENDER_PASS = context.space_data.shading.render_pass
        context.space_data.shading.cycles.render_pass = 'DIFFUSE_COLOR'

        # Store scene settings
        temp_scn['preview_bj_idx'] = self.job_index
        temp_scn['preview_pass_idx'] = self.pass_index
        temp_scn['orig_scene_name'] = self.orig_scene_name

        # Configure render settings
        temp_scn.view_settings.view_transform = 'Standard'
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
        if not preview_scene:
            return {'CANCELLED'}

        # Clear material override
        context.view_layer.material_override = None

        # Remove all objects marked as temporary
        for obj in preview_scene.objects:
            if obj.get('tmp', False):
                mesh = obj.data if obj.type == "MESH" else None
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)

        # Remove all collections marked as temporary
        for coll in bpy.data.collections:
            if coll.get('tmp', False):
                bpy.data.collections.remove(coll, do_unlink=True)

        # Remove main collections
        for coll_name in ('HIGHPOLY_PREVIEW', 'LOWPOLY_PREVIEW', 'CAGE_PREVIEW'):
            if coll := bpy.data.collections.get(coll_name):
                bpy.data.collections.remove(coll, do_unlink=True)

        # Remove materials
        if attrib_mat := bpy.data.materials.get("CBaker_AttribMaterial"):
            bpy.data.materials.remove(attrib_mat, do_unlink=True)

        # Remove node groups
        for node_group_name in node_groups_to_remove:
            if node_group := bpy.data.node_groups.get(node_group_name):
                bpy.data.node_groups.remove(node_group)

        # Switch back to original scene
        original_scene = bpy.data.scenes.get("Scene")
        if original_scene:
            context.window.scene = original_scene
        bpy.data.scenes.remove(preview_scene)

        # Restore shading settings
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
