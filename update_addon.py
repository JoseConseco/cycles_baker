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
import ssl
import bpy
import zipfile
import json
import urllib.request
import addon_utils
import os
import shutil
import sys
from io import BytesIO
import socket
TAGS_URL = ''


def is_connected():
    hostname = "www.google.com"
    try:
        # see if we can resolve the host name -- tells us if there is
        # a DNS listening
        host = socket.gethostbyname(hostname)
        s = socket.create_connection((host, 80), 2)
        return True
    except:
        pass
    return False


def get_addon_name():
    return __package__.split(".")[-1]


def addon_name_lowercase():
    return get_addon_name().lower()


def get_addon_preferences():
    return bpy.context.preferences.addons[__package__].preferences


def get_json_from_remonte():
    try:
        gcontext = ssl.SSLContext()  # Only for gangstars this actually works
        with urllib.request.urlopen(TAGS_URL, context=gcontext) as response:
            remonte_json = response.read()
    except urllib.request.HTTPError as err:
        print('Could not read tags from server.')
        print(err)
        return None
    tags_json = json.loads(remonte_json)
    return tags_json


def get_installed_version():
    installed_ver = ''
    # addon_ver = sys.modules[__package__].bl_info.get('version')

    mod = sys.modules[__package__]
    bl_info = addon_utils.module_bl_info(mod)
    min_blender_ver = bl_info.get("blender", (0, 0, 0))
    if bpy.app.version < min_blender_ver:
        print(f'Addon requires Blender {min_blender_ver} or later')
        return

    addon_ver = bl_info.get("version", (0, 0, 0))
    for a in addon_ver:
        installed_ver += str(a)+'.'
    return installed_ver[:-1]


def str_version_to_float(ver_str):
    repi = ver_str.partition('.')
    cc = repi[0]+'.' + repi[2].replace('.', '')
    return float(cc)


def check_update_exist():
    tags_json = get_json_from_remonte()
    remonte_ver_str = tags_json[-1]['version']
    release_notes = tags_json[-1]['release_notes']
    installed_ver_float = str_version_to_float(get_installed_version())
    remonte_ver_float = str_version_to_float(remonte_ver_str)
    return remonte_ver_float > installed_ver_float, remonte_ver_str, release_notes


def get_latest_version_url():
    tags_json = get_json_from_remonte()
    if len(tags_json) == 0:
        print('Remonte releases list is empty')
        return None
    zip_url = tags_json[-1]['download_url']
    return zip_url


def get_current_version_url():
    tags_json = get_json_from_remonte()
    if len(tags_json) == 0:
        print('Remonte releases list is empty')
        return None
    installed_ver = get_installed_version()
    for i, release in enumerate(tags_json):
        if release['version'] == installed_ver:
            return tags_json[i]['download_url']
    return None


def get_previous_version_url():
    tags_json = get_json_from_remonte()
    if len(tags_json) == 0:
        print('Remonte releases list is empty')
        return None, None
    installed_ver = get_installed_version()
    for i, release in enumerate(tags_json):
        if release['version'] == installed_ver:
            if i >= 1:
                return tags_json[i-1]['download_url'], tags_json[i-1]['version']
            else:
                return None, None
    return None, None


def clean_dir(folder):
    for the_file in os.listdir(folder):
        if the_file.startswith(('.vscode', '.vs', '.git')):
            continue
        file_path = os.path.join(folder, the_file)
        print(f'Removing old file: {the_file}')
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def reload_addon():
    print("Reloading addon...")
    addon_utils.modules(refresh=True)
    bpy.utils.refresh_script_paths()
    # not allowed in restricted context, such as register module
    # toggle to refresh
    # mod_name = get_addon_name()
    # bpy.ops.preferences.addon_disable(module=mod_name)
    # bpy.ops.preferences.addon_refresh()
    # bpy.ops.preferences.addon_enable(module=mod_name)


def get_update(zip_url):
    current_dir = os.path.split(__file__)[0]
    context = ssl._create_unverified_context()
    zipdata = BytesIO(urllib.request.urlopen(zip_url, context=context).read())
    #!Wont work on MAC:  zipdata = BytesIO(urllib.request.urlopen(zip_url).read())
    # zipdata.seek(0) #move it back to beggining
    addon_zip_file = zipfile.ZipFile(zipdata)

    clean_dir(current_dir)  # remove old files before update

    # addon_zip_file.extractall(directory_to_extract_to) #we could do this but zip contains weird root folder name
    created_sub_dirs = []
    for name in addon_zip_file.namelist():
        if '/' not in name:
            continue
        top_folder = name[:name.index('/')+1]
        if name == top_folder:
            continue  # skip top level folder
        subpath = name[name.index('/')+1:]
        if subpath.startswith(('.vscode', '.vs', '.git', '__pycache__')):
            continue
        if subpath.endswith(('.pyc')):  # , 'update_addon.py'
            continue
        sub_path, file_name = os.path.split(subpath)  # splits
        #bug - os.path.exists(os.path.join(current_dir, sub_path)) - dosent detect path created in previous iteraiton. Use created_sub_dirs, to store dirs
        if sub_path and sub_path not in created_sub_dirs and not os.path.exists(os.path.join(current_dir, sub_path)):
            try:
                os.mkdir(os.path.join(current_dir, sub_path))
                created_sub_dirs.append(sub_path)
                # print("Extract - mkdir: ", os.path.join(current_dir, subpath))
            except OSError as exc:
                print("Could not create folder from zip")
                return 'Install failed'
        with open(os.path.join(current_dir, subpath), "wb") as outfile:
            data = addon_zip_file.read(name)
            outfile.write(data)
            print("Extracting :", os.path.join(current_dir, subpath))
    addon_zip_file.close()
    reload_addon()
    return


class AddonCheckUpdateExist(bpy.types.Operator):
    bl_idname = addon_name_lowercase()+".check_for_update"
    bl_label = "Check for update"
    bl_description = "Check for update"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        addon_prefs = get_addon_preferences()
        if not is_connected():
            addon_prefs.update_text = 'Make sure you are connected to internet'
            return {'CANCELLED'}
        update_exist, remonte_ver_str, release_notes = check_update_exist()
        curr_ver_f = str_version_to_float(get_installed_version())
        rem_ver_f = str_version_to_float(remonte_ver_str)
        if update_exist:
            self.report({'INFO'}, f'Found new update: {remonte_ver_str}')
            addon_prefs.update_text = f'Found new update: {remonte_ver_str}\nRelease notes:\n{release_notes}'
            addon_prefs.update_exist = True
        elif curr_ver_f == rem_ver_f:
            self.report({'INFO'}, 'You have the latest version')
            addon_prefs.update_text = f'You have the latest version: {get_installed_version()}\nRelease notes:\n{release_notes}'
            addon_prefs.update_exist = False
        elif curr_ver_f > rem_ver_f:
            self.report({'INFO'}, 'You are ahead of official releases')
            addon_prefs.update_text = f'You seem to be ahead of official releases\nYour version: {get_installed_version()}\nRemonte Version: {remonte_ver_str}\nThere is nothing to download'
            addon_prefs.update_exist = False

        return {"FINISHED"}


class AddonUpdate(bpy.types.Operator):
    bl_idname = addon_name_lowercase()+".update_addon"
    bl_label = "Update addon"
    bl_description = "Download and install addon. May require blender restart"
    bl_options = {"REGISTER", "UNDO"}

    reinstall: bpy.props.BoolProperty(name='Reinstall', description='Force reinstalling curernt version', default=False)

    def execute(self, context):
        addon_prefs = get_addon_preferences()
        if not is_connected():
            addon_prefs.update_text = 'Make sure you are connected to internet'
            return {'CANCELLED'}
        if self.reinstall:
            latest_zip_url = get_current_version_url()
        else:
            latest_zip_url = get_latest_version_url()

        if latest_zip_url:
            print('Downloading addon')
            addon_prefs.update_text = 'Downloading addon'
            get_update(latest_zip_url)
            text = 'reinstalled' if self.reinstall else 'updated'
            self.report({'INFO'}, f'Addon {text}')
            addon_prefs.update_text = f'Addon {text}. Consider restarting blender'
            addon_prefs.update_exist = False
        else:
            print('There is nothing to download')
            addon_prefs.update_text = 'There is nothing to download'

        return {"FINISHED"}


class AddonRollBack(bpy.types.Operator):
    bl_idname = addon_name_lowercase()+".rollback_addon"
    bl_label = "Get previous version"
    bl_description = "Download and install previous version (if exist)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        addon_prefs = get_addon_preferences()
        if not is_connected():
            addon_prefs.update_text = 'Make sure you are connected to internet'
            return {'CANCELLED'}
        prevous_version_url, prev_ver_id = get_previous_version_url()
        if prevous_version_url:
            print('Reverting addon')
            addon_prefs.update_text = f'Reverting addon to {prev_ver_id}'
            get_update(prevous_version_url)
            self.report({'INFO'}, 'Addon Reverted')
            addon_prefs.update_text = 'Addon reverted. Consider restarting blender'
            addon_prefs.update_exist = False
        else:
            addon_prefs.update_text = 'No previous version found!'
        return {"FINISHED"}
