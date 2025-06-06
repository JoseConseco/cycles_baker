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

# bl_info = {
#     "name": "Cycles Baker",
#     "author": "Bartosz Styperek",
#     "blender": (4, 2, 0),
#     "version": (2, 0, 0),
#     "location": "(N) Right Sidebar > Cycles Baking (tab)",
#     "description": "UI for baking with Cycles.",
#     "warning": "",
#     "doc_url": "https://joseconseco.github.io/HairTool_3_Documentation/",
#     "tracker_url": "https://discord.gg/cxZDbqH",
#     "category": "Object"}

if "bpy" in locals():
    import importlib

    importlib.reload(utils)
    importlib.reload(bake)
    importlib.reload(ui)
    importlib.reload(props)
else:
    from . import utils
    from . import bake
    from . import ui
    from . import props


# register
##################################
import bpy
from . import auto_load
auto_load.init()



def register():
    # bpy.utils.register_class(hair_workspace_tool.GiGroup)
    auto_load.register()
    print("Registered Cycles Baker")

    addon_prefs = utils.get_addon_preferences()
    addon_prefs.update_text = ''

    ui.update_panel(None, bpy.context)
    props.register_props()


def unregister():
    auto_load.unregister()
    props.unregister_props()
    print("Unregistered Cycles Baker")


if __name__ == "__main__":
    register()
