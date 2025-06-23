Extension for Blender 4.4+ for baking pairs of high/low-poly objects.
It supports baking:
 * Material (Diffuse),
 * Normal,
 * Opacity.
 * Ambient Occlusion (and thickness),
 * Curvature,
 * Depth (not 100% accurate, but good enough for most cases - like scalp masks),
![cycles_baker_ui](https://github.com/user-attachments/assets/d79ce21c-f018-43bb-8713-6871e539e6dc)

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
