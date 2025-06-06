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

def get_addon_name():
    return __package__.split(".")[-1]

def addon_name_lowercase():
    return get_addon_name().lower()

def get_addon_preferences():
    # return bpy.context.preferences.addons[get_addon_name()].preferences
    return bpy.context.preferences.addons[__package__].preferences

def clear_parent(obj):
    parent = obj.parent
    if parent:
        obj.matrix_basis = parent.matrix_world @ obj.matrix_parent_inverse @ obj.matrix_basis
        obj.parent = None


def set_parent(obj, new_parent):
    if obj.parent:
        old_parent_m_w = obj.parent.matrix_world.copy() #backup cos we will change obj.parent below, so backup
        backup_obj_mpi = obj.matrix_parent_inverse.copy()  # bug in blender: obj.parent = new_parent clears obj.matrix_parent_inverse so backup it
        obj.parent = new_parent
        new_m_p_inv = new_parent.matrix_world.inverted() @ old_parent_m_w @ backup_obj_mpi
        obj.matrix_parent_inverse = new_m_p_inv
    else:
        obj.parent = new_parent
        obj.matrix_parent_inverse = new_parent.matrix_world.inverted()

def assign_material(obj, material_name, clear_materials=True):
    '''Add material to obj. If clear_material is True - remove all slots except first.'''
    mat = bpy.data.materials.get(material_name)
    if not mat:
        print('Material %s doesn\'t exist!' % (material_name))
        return
    if clear_materials is True:
        while len(obj.material_slots) > 1:
            obj.data.materials.pop()
    if len(obj.material_slots) == 0:  # make sure first slot is assigned
        obj.data.materials.append(mat)
    else:
        obj.material_slots[0].material = mat


def abs_file_path(filePath):
    path = Path(bpy.path.abspath(filePath))
    return str(Path(path).resolve())


# function to get all node groups in a node tree recursively
def get_nested_node_groups(node_tree):
    node_groups = []
    for node in node_tree.nodes:
        if node.bl_idname in ('GeometryNodeGroup', 'ShaderNodeGroup'):
            node_groups.append(node.node_tree)
            node_groups.extend(get_nested_node_groups(node.node_tree))
    return set(node_groups) # without duplicates


def remap_node_groups(new_ngroups, original_ngroups, data_type='node_groups'):
    for orig_ng_name, old_node_gr in original_ngroups.items(): # old name: Twist
        for new_ng in new_ngroups: # imported name: Twist.001
            if new_ng.name.startswith(orig_ng_name) and abs(len(new_ng.name)-len(orig_ng_name))<=4:   # 'Twist.001'.startswith('Twist')
                # new_data = data_to.node_groups[all_lib_node_groups_names.index(orig_node_gr_name)] # we access by index
                print(f'remapping: {orig_ng_name} to new: {new_ng.name}')
                try:
                    old_node_gr.user_remap(new_ng)
                except Exception as e:
                    print(f'failed to remap: {orig_ng_name} to new: {new_ng.name}')
                else: # remap went ok
                    # due to https://projects.blender.org/blender/blender/issues/119139  in blender 4.1
                    # we have to remove data block that holds the name. Then use that name, on new data block...
                    data_handle = getattr(bpy.data, data_type)
                    data_handle.remove(old_node_gr)
                    new_ng.name = orig_ng_name

def link_obj_to_same_collections(source_obj, clone, force_linking = True):
    ''' Link obj to collections where source_obj is located '''
    was_linked = False
    for source_obj_coll_parent in source_obj.users_collection:
        if clone.name not in source_obj_coll_parent.objects.keys():
            source_obj_coll_parent.objects.link(clone)
            was_linked = True

    if source_obj.name in bpy.context.scene.collection.objects.keys():
        if clone.name not in bpy.context.scene.collection.objects.keys():
            bpy.context.scene.collection.objects.link(clone)
            was_linked = True
    if not was_linked and force_linking: #link to at least one collection
        bpy.context.scene.collection.objects.link(clone)


data_types = ['node_groups']  # 'materials', 'textures' etc from gpro

def import_node_group(node_name, force_update=False):
    ''' import node gr. from hair_geo_nodes.blend '''
    script_file = Path(__file__).resolve()
    filepath = script_file.parent / "baker_library.blend" # modern way
    lib_file = str(filepath)
    old_node_gr = bpy.data.node_groups.get(node_name)

    if not old_node_gr or force_update:
        # store name now before it gets overridden
        original_node_groups = {node_group.name: node_group for node_group in bpy.data.node_groups}
        with bpy.data.libraries.load(lib_file) as (data_from, data_to):
            data_to.node_groups = [node_n for node_n in data_from.node_groups if node_n == node_name]
            imported_gr_names = [node_n for node_n in data_from.node_groups if node_n == node_name]  # stores names
            # all_lib_node_groups_names = data_from.node_groups[:] # part gets imported

        if data_to.node_groups:
            print(f"{imported_gr_names=}")
        else:
            print(f"node group {node_name} not found in {lib_file}")
            return None


        if old_node_gr: #remap main node
            old_node_gr.user_remap(data_to.node_groups[0])
            # due to https://projects.blender.org/blender/blender/issues/119139  in blender 4.1
            # we have to remove data block that holds the name. Then use freed name, on new data block...
            bpy.data.node_groups.remove(old_node_gr)
            data_to.node_groups[0].name = node_name

        # imported node_gr - may contain nested node groups - scan for them and remap old ngroups to new ones
        imported_node_groups = get_nested_node_groups(data_to.node_groups[0])
        remap_node_groups(imported_node_groups, original_node_groups)

        return data_to.node_groups[0]
    else:
        return old_node_gr


def safe_rename(data_blocks, id_data_to_rename, new_name: str):
    ''' renames id_data to new_name if new_name is not taken, else swaps current id_data with taken one'''
    # due to https://projects.blender.org/blender/blender/issues/119139  in blender 4.1
    # we have to free name from data block that holds the name. Then use that name, on new data block...
    old_data =  data_blocks.get(new_name)
    if old_data: # name already taken...  perform swap..
        current_name = id_data_to_rename.name
        old_data.name = 'random_available_tmp_name'  # free up 'a_temp' name (no id is using it now)
        id_data_to_rename.name = new_name
        old_data.name = current_name
    else:
        id_data_to_rename.name = new_name


def add_geonodes_mod(obj, name, node_type):
    geo_nodes_mod = obj.modifiers.new(name,type='NODES')
    # seems to work on hidden objs...
    # modifier_move_to_index(obj, geo_nodes_mod.name, 0)
    # bpy.ops.node.new_geometry_node_group_assign()
    ntree = import_node_group(node_type)
    geo_nodes_mod.node_group = ntree
    return geo_nodes_mod
