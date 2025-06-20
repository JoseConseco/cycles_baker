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
from .utils import abs_file_path, import_node_group, link_obj_to_same_collections, get_addon_preferences, addon_name_lowercase


class CB_PT_SDPanel(bpy.types.Panel):
    bl_label = "Cycles Baker"
    bl_idname = "CB_PT_SDPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cycles Baking"


    def draw(self, context):
        layout = self.layout
        active_obj = context.active_object

        if context.scene.name == "MD_PREVIEW":
            layout.operator("cycles.close_preview", icon="PANEL_CLOSE")

            temp_scn = context.scene
            preview_pass_idx = temp_scn.get('preview_pass_idx')
            preview_jb_idx = temp_scn.get('preview_bj_idx')
            if preview_jb_idx is not None and preview_pass_idx is not None:
                orig_scene_name = temp_scn['orig_scene_name']
                orig_scene = bpy.data.scenes.get(orig_scene_name)

                bj = orig_scene.cycles_baker_settings.bake_job_queue[preview_jb_idx]
                bakepass = bj.bake_pass_list[preview_pass_idx]
                row = layout.row(align=True)
                box = row.box().column(align=True)

                subrow = box.row(align=True)
                subrow.active = False
                subrow.label(text=bakepass.pass_type)

                for prop_name, config in bakepass.props().items():
                    subrow = box.row(align=True)

                    if config is None:
                        subrow.prop(bakepass, prop_name)
                    elif config["type"] == "prop_search":
                        subrow.prop_search(bakepass, prop_name, bpy.data, config["search_data"])
                    elif config["type"] == "toggle":
                        subrow.prop(bakepass, prop_name, toggle=True)

            return
        elif context.scene.name == "MD_TEMP":
            row.operator("cycles.cleanup_cycles_bake", icon="TRASH")
        else:
            layout.operator("cycles.bake", text='Bake', icon="SCENE")

        layout.separator()

        CyclesBakeSettings = context.scene.cycles_baker_settings
        for job_i, bj in enumerate(CyclesBakeSettings.bake_job_queue):
            row = layout.row(align=True)

            if bj.expand is False:
                row.prop(bj, "expand", icon="TRIA_RIGHT", icon_only=True, text=bj.name, emboss=False)

                icon = "RESTRICT_RENDER_OFF" if bj.activated else "RESTRICT_RENDER_ON"
                row.prop(bj, "activated", icon_only=True, icon=icon, emboss=False)

                oper = row.operator("cyclesbaker.texture_preview", text="", icon="TEXTURE")
                oper.bj_i = job_i
                rem = row.operator("cyclesbake.rem_job", text="", icon="X")
                rem.job_index = job_i
            else:
                row.prop(bj, "expand", icon="TRIA_DOWN", icon_only=True, text=bj.name, emboss=False)

                icon = "RESTRICT_RENDER_OFF" if bj.activated else "RESTRICT_RENDER_ON"
                row.prop(bj, "activated", icon_only=True, icon=icon, emboss=False)

                oper = row.operator("cyclesbaker.texture_preview", text="", icon="TEXTURE")
                oper.bj_i = job_i
                rem = row.operator("cyclesbake.rem_job", text="", icon="X")
                rem.job_index = job_i

                row = layout.row(align=True)
                row.prop(bj, 'bakeResolution', text="Resolution")

                row = layout.row(align=True)
                row.prop(bj, 'antialiasing', text="AA")

                row = layout.row(align=True)
                row.prop(bj, 'output', text="Path")

                row = layout.row(align=True)
                row.prop(bj, 'name', text="Name")

                row = layout.row(align=True)
                split = row.split(factor=0.70, align=True)
                split.prop(bj, 'padding_mode', text='')
                if bj.padding_mode == 'FIXED':
                    split.prop(bj, 'padding_size', text='')
                else:
                    sub_r = split.row(align=True)
                    sub_r.enabled = False
                    sub_r.prop(bj, 'padding_size', text='')

                row = layout.row(align=True)
                for pair_i, pair in enumerate(bj.bake_pairs_list):
                    row = layout.column(align=True).row(align=True)
                    box = row.box().column(align=True)

                    subrow = box.row(align=True)
                    ic = "SNAP_FACE" if bpy.data.objects.get(pair.lowpoly) == active_obj else "OBJECT_DATA"
                    subrow.prop_search(pair, "lowpoly", bpy.context.scene, "objects", icon=ic)
                    oper = subrow.operator("cyclesbake.objectpicker", text="", icon="EYEDROPPER")
                    oper.bj_i = job_i
                    oper.pair_i = pair_i
                    oper.gr_obj = "object"
                    oper.prop = "lowpoly"

                    subrow = box.row(align=True)
                    subrow.prop(pair, 'hp_type', expand=True)
                    if pair.hp_type == 'OBJ':
                        ic = "SNAP_FACE" if bpy.data.objects.get(pair.highpoly) == active_obj else "OBJECT_DATA"
                        subrow.prop_search(pair, "highpoly", bpy.context.scene, "objects", icon=ic)
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
                        subrow.prop(pair, 'ray_dist', expand=True)
                        subrow.prop(pair, 'draw_front_dist', icon='MOD_THICKNESS', icon_only=True, expand=True)
                    else:
                        subrow.prop_search(pair, "cage", bpy.context.scene, "objects")
                        oper = subrow.operator("cyclesbake.cage_maker", text="", icon="FILE_NEW")
                        oper.bj_i = job_i
                        oper.pair_i = pair_i
                        oper = subrow.operator("cyclesbake.objectpicker", text="", icon="EYEDROPPER")
                        oper.bj_i = job_i
                        oper.pair_i = pair_i
                        oper.gr_obj = "object"
                        oper.prop = "cage"

                    col = row.column()
                    rem = col.operator("cyclesbake.rem_pair", text="", icon="X")
                    rem.pair_index = pair_i
                    rem.job_index = job_i

                    col.operator("cycles.bake", text='', icon="SCENE").bake_pair_index = pair_i

                    ic = "RESTRICT_RENDER_ON" if not pair.activated else "RESTRICT_RENDER_OFF"
                    col.prop(pair, "activated", icon_only=True, icon=ic, emboss=False)

                row = layout.row(align=True)
                addpair = row.operator("cyclesbake.add_pair", icon="DISCLOSURE_TRI_RIGHT")
                addpair.job_index = job_i

                for pass_i, bakepass in enumerate(bj.bake_pass_list):
                    row = layout.row(align=True)
                    box = row.box().column(align=True)

                    subrow = box.row(align=True)
                    subrow.prop(bakepass, 'pass_type')

                    for prop_name, config in bakepass.props().items():
                        subrow = box.row(align=True)

                        if config is None:
                            subrow.prop(bakepass, prop_name)
                        elif config["type"] == "prop_search":
                            subrow.prop_search(bakepass, prop_name, bpy.data, config["search_data"])
                        elif config["type"] == "toggle":
                            subrow.prop(bakepass, prop_name, toggle=True)

                    col = row.column()
                    row = col.row()
                    rem = row.operator("cyclesbake.rem_pass", text="", icon="X")
                    rem.pass_index = pass_i
                    rem.job_index = job_i

                    row = col.row()
                    icon = "RESTRICT_RENDER_OFF" if bakepass.activated else "RESTRICT_RENDER_ON"
                    row.prop(bakepass, "activated", icon_only=True, icon=icon, emboss=False)

                    # row for Preview button - with eye icon
                    if bakepass.pass_type in ( "AO_GN", "DEPTH", "CURVATURE"):
                        row = col.row(align=True)
                        op = row.operator("cycles.preview_pass", text="", icon="HIDE_OFF")
                        op.pass_type = bakepass.pass_type
                        op.job_index = job_i
                        op.pass_index = pass_i
                        op.orig_scene_name = context.scene.name

                row = layout.row(align=True)
                addpass = row.operator("cyclesbake.add_pass", icon="DISCLOSURE_TRI_RIGHT")
                addpass.job_index = job_i

                row = layout.row(align=True)
                row.separator()

        row = layout.row(align=True)
        row.operator("cyclesbake.add_job", icon="ADD")


panels = (
    CB_PT_SDPanel,
)

def update_panel(self, context):
    message = "Cycles Baker: Updating Panel locations has failed"
    try:
        for panel in panels:
            if "bl_rna" in panel.__dict__:
                bpy.utils.unregister_class(panel)

        for panel in panels:
            panel.bl_category = get_addon_preferences().category
            bpy.utils.register_class(panel)

    except Exception as e:
        print("\n[{}]\n{}\n\nError:\n{}".format(__name__, message, e))
        pass




class CyclesBakerPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    DIFFUSE: bpy.props.StringProperty(name="Mat ID", description="", default='id')
    AO: bpy.props.StringProperty(name="AO", description="", default='ao')
    NORMAL: bpy.props.StringProperty(name="Normal", description="", default="nrm")
    DEPTH: bpy.props.StringProperty(name="Depth map", description="", default="depth")
    CURVATURE: bpy.props.StringProperty(name="Curvature map", description="", default="curvature")
    OPACITY: bpy.props.StringProperty(name="Opacity map", description="", default="opacity")
    COMBINED: bpy.props.StringProperty(name="Combined map", description="", default="combined")

    tabs: bpy.props.EnumProperty(name="Tabs", items=[("UPDATE", "Update", ""),
                                                     ("SETTINGS", "Settings", ""),
                                                     ("CATEGORY", "Category", ""),], default="UPDATE")

    category: bpy.props.StringProperty(name="Tab Category", description="Choose a name for the category of the panel", default="Tools", update=update_panel)

    update_exist: bpy.props.BoolProperty(name="Update Exist", description="There is new GroupPro update",  default=False)
    update_text: bpy.props.StringProperty(name="Update text",  default='')

    pair_spacing_distance: bpy.props.FloatProperty(name="Pair Spread Distance", description="Offset added between high-low pairs during bake, to prevent object pairs affecting each other", default=5.0, min=0.01, soft_max=10.0)
    play_finish_sound: bpy.props.BoolProperty(name="Play Finish Sound", description="Play sound when baking finished", default=True)


    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.prop(self, "tabs", expand=True)
        box = layout.box()
        if self.tabs == "SETTINGS":
            col = box.column(align=True)
            col.prop(self, "pair_spacing_distance")
            col.prop(self, "play_finish_sound")
            col.separator()
            col.label(text="Texture Suffixes:")
            col.prop(self, "DIFFUSE")
            col.prop(self, "AO")
            col.prop(self, "NORMAL")
            col.prop(self, "DEPTH")
            col.prop(self, "CURVATURE")
            # col.prop(self, "COMBINED")
            col.prop(self, "OPACITY")

        elif self.tabs == "UPDATE":
            col = box.column()
            sub_row = col.row(align=True)
            sub_row.operator(addon_name_lowercase()+".check_for_update")
            split_lines_text = self.update_text.splitlines()
            for line in split_lines_text:
                sub_row = col.row(align=True)
                sub_row.label(text=line)
            sub_row.separator()
            sub_row = col.row(align=True)
            if self.update_exist:
                sub_row.operator(addon_name_lowercase()+".update_addon", text='Install latest version').reinstall = False
            else:
                sub_row.operator(addon_name_lowercase()+".update_addon", text='Reinstall current version').reinstall = True
            sub_row.operator(addon_name_lowercase()+".update_addon", text='Grab Latest Daily Build').daily_build = True
            sub_row.operator(addon_name_lowercase()+".rollback_addon")

        elif self.tabs == "CATEGORY":
            col = box.column()
            col.prop(self, "category")


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
        # addonPref = get_addon_preferences()
        newpass = context.scene.cycles_baker_settings.bake_job_queue[self.job_index].bake_pass_list.add()
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

    def execute(self, context):
        bj = context.scene.cycles_baker_settings.bake_job_queue[self.bj_i]
        pair = bj.bake_pairs_list[self.pair_i]
        lowObj = bpy.data.objects.get(pair.lowpoly)
        if not lowObj:
            self.report({'ERROR'}, f"Lowpoly object '{self.lowpoly}' not found.")
            return {'CANCELLED'}

        # cageObj = bpy.data.objects.new('cage_' + self.lowpoly, lowObj.data.copy())  # duplicate obj but not instance
        cageObj = lowObj.copy()  # linked copy
        cageObj.matrix_world = lowObj.matrix_world
        cageObj.show_wire = True
        cageObj.show_all_edges = True
        cageObj.display_type = 'WIRE'
        push_mod = cageObj.modifiers.new('Push', 'DISPLACE')
        push_mod.strength = 0.1
        push_mod.direction = 'NORMAL'
        push_mod.mid_level = 0  # This ensures displacement starts from the surface
        push_mod.space = 'LOCAL'

        vg = cageObj.vertex_groups.new(name='displace_weight')
        weight = 1
        for vert in lowObj.data.vertices:
            vg.add([vert.index], weight, "ADD")

        push_mod.vertex_group = vg.name

        link_obj_to_same_collections(lowObj, cageObj)
        pair.cage = cageObj.name
        return {'FINISHED'}



class CB_OT_CyclesTexturePreview(bpy.types.Operator):
    bl_idname = "cyclesbaker.texture_preview"
    bl_label = "Preview Baked Texture"
    bl_description = "Preview texture on model, by assigning bake result to lowpoly objects \n" \
                     "Press Shift - to preview multiple bake jobs"

    bj_i: bpy.props.IntProperty()
    shiftClicked = False

    def invoke(self, context, event):
        if event.shift:
            self.shiftClicked = True
        return self.execute(context)

    def attachCyclesmaterial(self, obj, mat):
        if len(obj.material_slots) == 0:
            obj.data.materials.append(mat)
        else:
            obj.material_slots[0].material = mat
        if bpy.context.scene.render.engine == 'CYCLES':
            obj.data.materials[0].use_nodes = True

    def execute(self, context):
        cycles_bake_settings = bpy.context.scene.cycles_baker_settings
        if self.shiftClicked:
            bjList = [bj for bj in cycles_bake_settings.bake_job_queue if bj.activated]
        else:
            bjList = [cycles_bake_settings.bake_job_queue[self.bj_i],]
        imagesFromBakePasses = []
        for bj in bjList:
            imagesFromBakePasses.clear()
            for bakepass in bj.bake_pass_list:  # refresh or load images from bakes
                if bakepass.activated:
                    bakedImgPath = str(bj.get_out_dir_path() / f"{bakepass.get_filename(bj)}.png")
                    imgAlreadyExist = False
                    oldImg = []
                    for img in bpy.data.images:  # find if bake is already loaded into bi images
                        if bakedImgPath == abs_file_path(img.filepath):
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
                        except Exception as e:
                            print(f"Cannot load image {bakedImgPath}: {str(e)}")
                            imagesFromBakePasses.append(None)  # preserve index
                else:
                    imagesFromBakePasses.append(None)  # to preserve breaking bakePass indexes
            bpy.context.space_data.shading.type = 'MATERIAL'

            mat = bpy.data.materials.get(bj.name)
            if not mat:
                mat = bpy.data.materials.new(name=bj.name)
                mat.diffuse_color = (0.609125, 0.0349034, 0.8, 1)
            mat.use_nodes = True

            obj_list = []
            for pair in bj.bake_pairs_list:
                lowpoly = bpy.data.objects.get(pair.lowpoly)
                if lowpoly:  # create group for hipoly
                    obj_list.append(bpy.data.objects[pair.lowpoly])

            for obj in obj_list:
                if obj.type == "MESH":
                    self.attachCyclesmaterial(obj, mat)
            if len(obj_list) == 0:
                continue

            matNodeTree = mat.node_tree
            for node in matNodeTree.nodes:
                matNodeTree.nodes.remove(node)
            links = matNodeTree.links
            principledNode = matNodeTree.nodes.new('ShaderNodeBsdfPrincipled')
            principledNode.location = 1300, 200

            outputNode = matNodeTree.nodes.new('ShaderNodeOutputMaterial')
            outputNode.location = 1700, 200
            links.new(principledNode.outputs[0], outputNode.inputs[0])
            previousID_AO_Node = None
            offset_x = 600
            for bakeIndex, bakeImg in enumerate(imagesFromBakePasses):
                if not bakeImg :  # skip imgs that could not be loaded
                    continue

                y_pos = bakeIndex * 300
                bakepass = bj.bake_pass_list[bakeIndex]
                # if not foundTextureSlotWithBakeImg: # always true in loop above wass not continued
                if bakepass.pass_type in ("DIFFUSE" , "AO"):
                    imgNode = matNodeTree.nodes.new('ShaderNodeTexImage')
                    imgNode.name = bakepass.get_pass_suffix()
                    imgNode.label = bakepass.get_pass_suffix()
                    imgNode.location = offset_x, y_pos
                    imgNode.image = bakeImg

                    if previousID_AO_Node:
                        mixNode = matNodeTree.nodes.new('ShaderNodeMixRGB')
                        mixNode.location = offset_x + 300, y_pos
                        mixNode.inputs[0].default_value = 1
                        if bakepass.pass_type == "AO":
                            mixNode.blend_type = 'MULTIPLY'

                        links.new(previousID_AO_Node.outputs[0], mixNode.inputs[2])
                        links.new(imgNode.outputs[0], mixNode.inputs[1])
                        links.new(mixNode.outputs[0], principledNode.inputs['Base Color'])
                        previousID_AO_Node = mixNode
                    else:  # first id or AO texture
                        links.new(imgNode.outputs[0], principledNode.inputs['Base Color'])
                        previousID_AO_Node = imgNode

                elif bakepass.pass_type == "NORMAL":
                    imgNormalNode = matNodeTree.nodes.new('ShaderNodeTexImage')
                    imgNormalNode.name = bakepass.get_pass_suffix()
                    imgNormalNode.label = bakepass.get_pass_suffix()
                    bakeImg.colorspace_settings.name = 'Non-Color' #or normals
                    imgNormalNode.image = bakeImg
                    imgNormalNode.location = offset_x, y_pos

                    normalMapNode = matNodeTree.nodes.new('ShaderNodeNormalMap')
                    normalMapNode.location = offset_x + 300, y_pos

                    if bakepass.nm_invert == 'NEG_Y':  # use DX normal == flip green chanel
                        flip_ng = import_node_group("FlipGreenChannel")
                        flipGreenNode = matNodeTree.nodes.new('ShaderNodeGroup')
                        flipGreenNode.node_tree = flip_ng
                        flipGreenNode.location = offset_x + 600, y_pos
                        # normalMapNode.location[0] += 200
                        links.new(imgNormalNode.outputs[0], flipGreenNode.inputs[0])
                        links.new(flipGreenNode.outputs[0], normalMapNode.inputs[1])
                        links.new(normalMapNode.outputs[0], principledNode.inputs['Normal'])
                    else:
                        links.new(imgNormalNode.outputs[0], normalMapNode.inputs[1])
                        links.new(normalMapNode.outputs[0], principledNode.inputs['Normal'])

                elif bakepass.pass_type == "OPACITY":
                    imgOpacityNode = matNodeTree.nodes.new('ShaderNodeTexImage')
                    imgOpacityNode.name = bakepass.get_pass_suffix()
                    imgOpacityNode.label = bakepass.get_pass_suffix()
                    imgOpacityNode.location = offset_x, y_pos
                    imgOpacityNode.image = bakeImg
                    links.new(imgOpacityNode.outputs[0], principledNode.inputs['Alpha'])

                else:  # for all other just create img and do not link
                    imgNormalNode = matNodeTree.nodes.new('ShaderNodeTexImage')
                    imgNormalNode.name = bakepass.get_pass_suffix()
                    imgNormalNode.label = bakepass.get_pass_suffix()
                    imgNormalNode.image = bakeImg
                    imgNormalNode.location = 800, y_pos

        return {'FINISHED'}



