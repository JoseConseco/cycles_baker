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

bl_info = {
    "name": "Cycles Baker",
    "author": "Bartosz Styperek",
    "version": (2, 0),
    "blender": (4, 4, 0),
    "location": "Npanel -> Tool shelf -> Baking (tab)",
    "description": "Addon for baking with Cycles.",
    "warning": "",
    "wiki_url": "",
    "category": "Object"}

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
    # addon_prefs = update_addon.get_addon_preferences()
    # if addon_prefs:
    #     addon_prefs.update_text = ''
    props.register_props()


def unregister():
    props.unregister_props()
    auto_load.unregister()
    print("Unregistered Cycles Baker")


if __name__ == "__main__":
    register()
