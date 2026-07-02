Extension for Blender 4.4+ for baking pairs of high/low-poly objects.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/yL7A8PiKmxM/0.jpg)](https://www.youtube.com/watch?v=yL7A8PiKmxM)

Installation:
 * Install Cycles Baker as *Extension** from Blender Preferences > Get Extensions > Install From Disk (drop-down menu at the very top right corner) > find place where you saved CyclesBaker zip file

Location:
* Right Sidebar > Tools > Cycles Baker tab

Usage:
 * use 'Add Pair' button and pick low-high (poly) objects pair (highpoly can be set to collection)
 * Add required baking passes (e.g. Diffuse, AO, Opacity, Depth etc)
 * set output: resolution, AA, padding, bake file path and name
 * Press 'Bake'

![CyclesBake](https://github.com/user-attachments/assets/7b58338c-1b76-46c9-b852-6323182fcf47)

Supported passes:
 * Material (Diffuse),
 * Normal,
 * Opacity.
 * Ambient Occlusion (and thickness),
 * Curvature,
 * Random Color (or greyscale) - per mesh island,
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
* Ability customize to preview baking cage/extrusion offset and ray distance
* support for custom cage object
* AA support (it bakes image of AA*resolution size, then scales it back down AA times to get AA effect)
* Generate 'Preview material', with all baked textures loaded into that material,
* Option for partial re-bake (partial texture update for selected high/low-poly pair) - no need to re-bake all objects
* Built-in auto-updater
* Channel Mixing (requires [Hair Tool](https://bartoszstyperek.gumroad.com/l/hairtool) to work - maybe in the future it will be available without it)
