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
import requests
import addon_utils
import os
import shutil
import sys
import tempfile
from io import BytesIO
import socket

TAGS_URL = "https://api.github.com/repos/JoseConseco/cycles_baker/releases"

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
    response = requests.get(TAGS_URL)
    releases = response.json()
    latest_release = releases[-1] if releases else {}

    version = latest_release.get('tag_name', '0.0.0')
    download_url = latest_release.get('zipball_url', '')
    release_date = latest_release.get('published_at', '')
    # ignore time get just data
    release_date = release_date[:11]
    release_name = latest_release.get('name', 'No name provided')

    release_notes = latest_release.get('body', 'No release notes provided')

    return [{
        'version': version,
        'release_date': release_date,
        'download_url': download_url,
        'release_notes': release_notes
    }]


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
    installed_ver_float = str_version_to_float(get_installed_version())
    remonte_ver_float = str_version_to_float(tags_json[-1]['version'])
    up_exists = remonte_ver_float > installed_ver_float
    return up_exists, tags_json[-1]['version'], tags_json[-1]['release_notes'], tags_json[-1]['release_date']


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
    # root_dir_name = os.path.basename(current_dir)
    context = ssl._create_unverified_context()

    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download and extract zip
        with urllib.request.urlopen(zip_url, context=context) as response:
            with BytesIO(response.read()) as zipdata:
                with zipfile.ZipFile(zipdata) as zip_file:
                    zip_file.extractall(temp_dir)

        # Get the root folder from the extracted content
        root_folder = next(os.walk(temp_dir))[1][0]
        source_dir = os.path.join(temp_dir, root_folder)

        # Clean current directory before update
        clean_dir(current_dir)

        # Copy files while filtering
        for root, dirs, files in os.walk(source_dir):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if not d.startswith(('.vscode', '.vs', '.git', '__pycache__'))]

            # Calculate relative path
            rel_path = os.path.relpath(root, source_dir)
            if rel_path == '.':
                rel_path = ''

            # Create target directory
            target_dir = os.path.join(current_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)

            # Copy files, skipping unwanted extensions
            for file in files:
                if not file.endswith('.pyc'):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, file)
                    shutil.copy2(src_file, dst_file)
                    print("Extracting:", dst_file)

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
        update_exist, remonte_ver_str, release_notes, release_date = check_update_exist()
        curr_ver_f = str_version_to_float(get_installed_version())
        rem_ver_f = str_version_to_float(remonte_ver_str)
        if update_exist:
            self.report({'INFO'}, f'Found new update: {remonte_ver_str}')
            addon_prefs.update_text = f'Found new update: {remonte_ver_str} - Date {release_date}\nRelease notes:\n{release_notes}'
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
