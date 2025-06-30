Extension for Blender 4.4+ for baking pairs of high/low-poly objects.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/yL7A8PiKmxM/0.jpg)](https://www.youtube.com/watch?v=yL7A8PiKmxM)

Installation:
 * Install Cycles Baker as *Extension** from Blender Preferences > Get Extensions > Install From Disk (drop-down menu at the very top right corner) > find place where you saved CyclesBaker zip file

Location:
* Right Sidebar > Tools > Cycles Baker tab

Usage:
 * use 'Add Pair' button and pick low-poly mesh and high-poly object (or collection)
 * Add needed baking passes (e.g. Diffuse, AO, Opacity, Depth etc)
 * Press 'Bake'

It supported passes:
 * Material (Diffuse),
 * Normal,
 * Opacity.
 * Ambient Occlusion (and thickness),
 * Curvature,
 * Position,
 * Depth (not 100% accurate, but good enough for most cases - like scalp masks),


![CyclesBake](https://github.com/user-attachments/assets/7b58338c-1b76-46c9-b852-6323182fcf47)

It supports baking:
 * Material (Diffuse),
 * Normal,
 * Opacity.
 * Ambient Occlusion (and thickness),
 * Curvature,
 * Depth (not 100% accurate, but good enough for most cases - like scalp masks),

Baking from various types of high-poly objects is supported:
 * Mesh,
 * [Hair](https://bartoszstyperek.gumroad.com/l/hairtool),
 * Collection Instance,
 * [Groups](https://bartoszstyperek.gumroad.com/l/GroupPro) of objects,
 * Text,
 * Metaball,
 * Others...

Other features:
* Ability customize to preview front ray distance
* Option to add and use custom cage object
* AA option (it bakes image in higher resolution, then scale it down to get AA effect)
* Generate preview material with baked textures with one click
* Option for partial re-bake (partial texture update for selected high/low-poly pair) - no need to re-bake all objects
* Built-in auto-updater
* Channel Mixing (requires [Hair Tool](https://bartoszstyperek.gumroad.com/l/hairtool) to work - maybe in the future it will be available without it)
