import os
from numpy.lib.stride_tricks import as_strided
import bpy
import aud
import math
from os import path
from bpy.props import *
from mathutils import Vector
from datetime import datetime
from pathlib import Path
from os.path import exists

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from .utils import abs_file_path, import_node_group, add_geonodes_mod

def get_raycast_distance(bj):
    # low_poly = bpy.data.objects[pair.lowpoly]
    # objBBoxSize = Vector(low_poly.dimensions[:]).length/2 if bj.relativeToBbox else 1
    # front = bj.frontDistance * pair.front_distance_modulator * objBBoxSize
    return max(bj.frontDistance, 0)


BG_color = {
    "NORMAL": np.array([0.5, 0.5, 1.0, 1.0]),
    "AO": np.array([1.0, 1.0, 1.0, 1.0]),
    "DIFFUSE": np.array([0.0, 0.0, 0.0, 0.0]),
    "OPACITY": np.array([0.0, 0.0, 0.0, 0.0]),
    }


shader_uniform = gpu.shader.from_builtin('UNIFORM_COLOR')
Verts = None
Normals = None
Indices = None
def draw_cage_callback(self, context):
    if not self.draw_front_dist or not self.lowpoly:
        return
    low_poly = bpy.data.objects[self.lowpoly]
    global Verts, Normals, Indices
    if low_poly.type == 'MESH' and context.mode == 'OBJECT':
        for bj in bpy.data.scenes['Scene'].cycles_baker_settings.bake_job_queue:
            for pair in bj.bake_pairs_list:
                if pair == self:
                    parentBakeJob = bj
                    break

        front = get_raycast_distance(parentBakeJob, self)

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

        Vertices = Vertices + Normals * front  # 0,86 to match substance ray distance

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

    def create_bake_mat_and_node(self):
        low_obj = bpy.data.objects["LOWPOLY_MD_TMP"]
        bake_mat = self.get_set_first_material_slot(low_obj)
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
        active_pairs = [pair for pair in bj.bake_pairs_list if pair.activated]  # get only activated pairs
        for i,pair in enumerate(active_pairs):
            offset = self.get_phyllotaxis_offset(i)

            low_poly_obj = bpy.data.objects[pair.lowpoly]

            low_poly_obj_cp = low_poly_obj.copy()
            low_poly_obj_cp.matrix_world.translation += offset
            lowpoly_objs.append(low_poly_obj_cp)  # To merge them later
            temp_scn.collection.objects.link(low_poly_obj_cp)  # unlink all other lowpoly objects

            cage = bpy.data.objects.get(pair.cage, None)
            if pair.use_cage and cage:
                # Copy existing cage object
                cage_obj = cage.copy()
                if i == 0:
                    cage_objs[0].data = cage_objs[0].data.copy() # to not affect original cage object
                cage_obj['tmp'] = True  # mark as tmp, so it can be deleted later
                cage_obj.matrix_world.translation += offset
                temp_scn.collection.objects.link(cage_obj)
                cage_objs.append(cage_obj)
            else:
                # Create temporary cage by displacing vertices along normals
                temp_cage = low_poly_obj_cp.copy()
                temp_cage.data = low_poly_obj_cp.data.copy()  # copy mesh data to not affect original
                temp_cage['tmp'] = True  # mark as tmp, so it can be deleted later
                temp_cage.name = f"TEMP_CAGE_{low_poly_obj_cp.name}"
                temp_scn.collection.objects.link(temp_cage)

                # Add displacement modifier - noo good - does not support split edges
                # displace = temp_cage.modifiers.new(name="CAGE_DISPLACE", type='DISPLACE')
                # displace.strength = get_raycast_distance(bj)
                # displace.mid_level = 0.0
                # I gueess there is no need to split lowpoly mesh too (maybe blender matches cage by face ids?)
                gn_displce = add_geonodes_mod(temp_cage, "CAGE_GEONODES", "CycBaker_SplitExtrude")
                gn_displce['Socket_2'] = get_raycast_distance(bj)  # set extrusion distance                                                 ]
                with context.temp_override(selected_editable_objects=[temp_cage], active_object=temp_cage, selected_objects=[temp_cage]):
                    bpy.ops.object.convert(target='MESH')
                cage_objs.append(temp_cage)


            if pair.hp_type == "GROUP":
                hi_collection = bpy.data.collections.get(pair.highpoly)
                for obj in hi_collection.objects:
                    if obj.type in ('CURVE', 'CURVES', 'FONT'):
                        hi_obj = self.obj_to_mesh(context, obj, out_hi_collection)
                        hi_obj.matrix_world.translation += offset
                    elif obj.type == 'EMPTY' and obj.instance_collection:
                        root_empty = self.make_inst_real(context, obj, out_hi_collection)
                        root_empty.matrix_world.translation += offset
                    else:
                        cp = obj.copy()
                        cp.matrix_world.translation += offset
                        cp['tmp'] = True  # mark as tmp, so it can be deleted later
                        out_hi_collection.objects.link(cp)

            else:  # pair.hp_type == "OBJ"
                hi_poly_obj = bpy.data.objects[pair.highpoly]
                if hi_poly_obj.type == 'EMPTY' and hi_poly_obj.instance_collection:
                    root_empty = self.make_inst_real(context, hi_poly_obj, out_hi_collection)
                    root_empty.matrix_world.translation += offset
                elif hi_poly_obj.type in ('CURVE', 'CURVES', 'FONT'):
                    hi_obj = self.obj_to_mesh(context, hi_poly_obj, out_hi_collection)
                    hi_obj.matrix_world.translation += offset
                else:
                    cp = hi_poly_obj.copy()
                    cp.matrix_world.translation += offset
                    cp['tmp'] = True  # mark as tmp, so it can be deleted later
                    out_hi_collection.objects.link(cp)


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

        high_coll = bpy.data.collections['HIGHPOLY_MD_TMP']
        for obj in high_coll.objects:
            if obj.type == 'MESH':
                select_obj(obj)

        lowpoly_obj = tmp_scn.objects["LOWPOLY_MD_TMP"]
        select_obj(lowpoly_obj)
        tmp_scn.view_layers[0].objects.active = lowpoly_obj

        # XXX: restore  cage - in tmp scene setup
        # if pair.use_cage and pair.cage != "":
        #     cage_obj = tmp_scn.objects[pair.cage + "_MD_TMP"]
        #     select_obj(cage_obj)

    def bake_pair_pass(self, bake_job, bakepass):
        self.create_bake_mat_and_node()
        startTime = datetime.now()  # time debug
        scn = bpy.data.scenes["MD_TEMP"]
        if bakepass.pass_name == "AO":
            scn.cycles.samples = bakepass.samples
            scn.world.light_settings.distance = bakepass.ao_distance

        front_dist = get_raycast_distance(bake_job)

        pass_name = bakepass.pass_name
        pass_components = {'NONE'}
        if pass_name == "DIFFUSE":
            pass_name = 'DIFFUSE'
            pass_components = {'COLOR'}
        elif pass_name == "OPACITY":
            pass_name = 'EMIT'
        elif pass_name == 'COMBINED':
            pass_components = {'AO', 'EMIT', 'DIRECT', 'INDIRECT', 'COLOR', 'DIFFUSE', 'GLOSSY'}

        aa = int(bake_job.antialiasing)
        res = int(bake_job.bakeResolution)

        padding = bake_job.padding_size if bake_job.padding_mode == 'FIXED' else int(res/64)
        if bakepass.pass_name == "OPACITY":
            padding = 0

        # cage_obj = bpy.data.objects.get("CAGE_MD_TMP", None)
        bpy.ops.object.bake(type=pass_name, filepath="", pass_filter=pass_components,
                            width=res*aa, height=res*aa,
                            margin=padding*aa,
                            use_selected_to_active=True,
                            cage_extrusion=front_dist,
                            normal_space=bakepass.nm_space,
                            normal_r="POS_X", normal_g=bakepass.nm_invert, normal_b='POS_Z',
                            save_mode='INTERNAL',
                            use_clear=False,
                            use_cage=True,
                            cage_object="CAGE_MD_TMP",
                            target='IMAGE_TEXTURES',
                            use_split_materials=False, use_automatic_name=False)

        print("Baking set " + bakepass.pass_name + "  time: " + str(datetime.now() - startTime))

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

    def cleanup(self):
        scn_tmp = bpy.data.scenes["MD_TEMP"]
        for obj in scn_tmp.objects:
            if obj.get('tmp', False):
                self.remove_object(obj)

        low_obj = bpy.data.objects.get("LOWPOLY_MD_TMP")
        if low_obj:
            self.remove_object(low_obj)  # remove lowpoly object

        # for material in bpy.data.materials:
        #     if material.name.endswith("_MD_TMP"):
        #         bpy.data.materials.remove(material, do_unlink=True)
        bake_mat = bpy.data.materials.get("CyclesBakeMat_MD_TEMP")
        if bake_mat:
            bpy.data.materials.remove(bake_mat, do_unlink=True)

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

        active_bj = [bj for bj in cycles_bake_settings.bake_job_queue if bj.activated]
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
                render_target = bpy.data.images.new("MDtarget", width=img_res*aa, height=img_res*aa, alpha=True, float_buffer=False)
                render_target.generated_color = BG_color[bakepass.pass_name]
                self.select_hi_low(bj)
                self.bake_pair_pass(bj, bakepass)

                if bakepass.pass_name != "OPACITY":  # opacity is not saved to image
                    pass
                    # add_padding_offscreen(render_target, img_res*aa, img_res*aa, padding_size=aa*padding)
                else: # copy alpha to color
                    pixels = np.array(render_target.pixels, dtype='f')
                    alpha = pixels.reshape(-1, 4)[:, 3]  # get alpha channel
                    # duplicate alpha to RGB channels and recombine
                    channel = np.repeat(alpha[:, np.newaxis], 3, axis=1)  # make RGB channels
                    render_target.pixels = np.concatenate((channel, alpha[:, np.newaxis]), axis=1).ravel()

                render_target.scale(img_res, img_res)
                imgPath = bj.get_filepath() + bakepass.get_filename(bj) + ".png"  # blender needs slash at end
                render_target.filepath_raw = imgPath
                render_target.save()

                # render_target.user_clear()
                bpy.data.images.remove(render_target)

                if path.isfile(imgPath):  # load bake from disk
                    img_users = (img for img in bpy.data.images if abs_file_path(img.filepath) == imgPath)
                    if img_users:
                        for img in img_users:
                            img.reload()  # reload done in baking
                    else:
                        img = bpy.data.images.load(filepath=imgPath)

            self.cleanup()  # delete scene


        print(f"Cycles Total baking time: {(datetime.now() - TotalTime).seconds} sec")
        # self.playFinishSound()

        return {'FINISHED'}

    @staticmethod
    def playFinishSound():
        script_file = os.path.realpath(__file__)
        directory = os.path.dirname(script_file)
        device = aud.Device()

        sound = aud.Sound(os.path.join(directory, "finished.mp3"))

        # play the audio, this return a handle to control play/pause
        handle = device.play(sound)
        # if the audio is not too big and will be used often you can buffer it
        sound_buffered = aud.Sound.buffer(sound, 24000.0)
        handle_buffered = device.play(sound_buffered)

        # stop the sounds (otherwise they play until their ends)
        handle.stop()
        handle_buffered.stop()



