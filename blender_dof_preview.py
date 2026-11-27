import math
import os
import bpy
import bmesh
from mathutils import Vector

OUT_DIR = r"e:\betweenthenotes\drone_animation_project\blender_previews"
os.makedirs(OUT_DIR, exist_ok=True)

MUSHROOM_PNG = r"e:\betweenthenotes\drone_animation_project\assets\blender_clean\hero_pagoda_mushroom_master.png"
PINE_WARM = r"e:\betweenthenotes\drone_animation_project\assets\blender_clean\pine_fuller_warm.png"
PINE_DEEP = r"e:\betweenthenotes\drone_animation_project\assets\blender_clean\pine_fuller_deep.png"
PINE_COOL = r"e:\betweenthenotes\drone_animation_project\assets\blender_clean\pine_fuller_cool.png"
DINO_SAURO = r"e:\betweenthenotes\drone_animation_project\assets\blender_clean\pokemon_sauropod.png"
DINO_HADRO = r"e:\betweenthenotes\drone_animation_project\assets\blender_clean\pokemon_hadrosaur.png"
DINO_ANKY = r"e:\betweenthenotes\drone_animation_project\assets\blender_clean\pokemon_ankylosaur.png"


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render(scene, width=1280, height=720):
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 32


def dune_height(x, y):
    """Smooth coastal sand-dune surface height z(x, y)."""
    ridge1 = 0.75 * math.sin(x * 0.28 + 0.4) * math.exp(-((y - 2.0) ** 2) / 14.0)
    ridge2 = 1.15 * math.cos(x * 0.19 - 0.8) * math.exp(-((y - 7.5) ** 2) / 18.0)
    slope = max(0.0, (y + 1.0) * 0.12)
    return slope + ridge1 + ridge2 - 0.35


def build_dune_mesh():
    bm = bmesh.new()
    nx, ny = 120, 60
    x_min, x_max = -22.0, 22.0
    y_min, y_max = -1.0, 18.0
    verts = []
    for j in range(ny):
        row = []
        y = y_min + (y_max - y_min) * (j / (ny - 1))
        for i in range(nx):
            x = x_min + (x_max - x_min) * (i / (nx - 1))
            z = dune_height(x, y)
            row.append(bm.verts.new((x, y, z)))
        verts.append(row)
    for j in range(ny - 1):
        for i in range(nx - 1):
            bm.faces.new((verts[j][i], verts[j][i + 1], verts[j + 1][i + 1], verts[j + 1][i]))
    mesh = bpy.data.meshes.new("DuneMesh")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new("Dunes", mesh)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new("SandMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.38, 0.31, 0.28, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.88
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def build_ocean():
    bpy.ops.mesh.primitive_plane_add(size=80.0, location=(0.0, -18.0, -0.32))
    ocean = bpy.context.active_object
    ocean.name = "Ocean"
    mat = bpy.data.materials.new("OceanMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.012, 0.085, 0.175, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.16
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.65
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 2.5
    wave.inputs["Distortion"].default_value = 3.2
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    ocean.data.materials.append(mat)
    return ocean


def add_sprite_card(name, img_path, x, y, height_world=3.0, sink=0.14, solid_body=False):
    img = bpy.data.images.load(img_path, check_existing=True)
    iw, ih = img.size[0], img.size[1]
    aspect = (iw / ih) if ih > 0 else 1.0
    w = height_world * aspect
    z_ground = dune_height(x, y)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(x, y, z_ground + height_world * 0.5 - sink))
    card = bpy.context.active_object
    card.name = name
    card.scale = (w, height_world, 1.0)
    card.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    mat = bpy.data.materials.new(f"Mat_{name}")
    mat.use_nodes = True
    if hasattr(mat, "blend_method"):
        mat.blend_method = "CLIP"
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.85

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "CLIP"

    # Trim bottom edge so no bounding-box fringe ever shows
    uv = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.03
    ramp.inputs["From Max"].default_value = 0.08
    math_mul = nt.nodes.new("ShaderNodeMath")
    math_mul.operation = "MULTIPLY"

    nt.links.new(uv.outputs["UV"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Y"], ramp.inputs["Value"])

    if solid_body:
        # Force any interior semi-transparent pixels (>0.06) to 100% opaque so sun never bleeds through
        solid_ramp = nt.nodes.new("ShaderNodeMapRange")
        solid_ramp.inputs["From Min"].default_value = 0.05
        solid_ramp.inputs["From Max"].default_value = 0.14
        nt.links.new(tex.outputs["Alpha"], solid_ramp.inputs["Value"])
        nt.links.new(solid_ramp.outputs["Result"], math_mul.inputs[0])
    else:
        nt.links.new(tex.outputs["Alpha"], math_mul.inputs[0])

    nt.links.new(ramp.outputs["Result"], math_mul.inputs[1])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(math_mul.outputs["Value"], bsdf.inputs["Alpha"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    card.data.materials.append(mat)
    return card


def add_sun_and_sky(scene, sun_pos, sky_horizon, sky_zenith, sun_energy=4.0):
    # World gradient sky
    world = bpy.data.worlds.new("SkyWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.15
    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (*sky_horizon, 1.0)
    ramp.color_ramp.elements[1].position = 0.45
    ramp.color_ramp.elements[1].color = (*sky_zenith, 1.0)
    nt.links.new(texcoord.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Visible glowing Sun disc
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.9, location=sun_pos)
    sun_orb = bpy.context.active_object
    sun_orb.name = "SunOrb"
    s_mat = bpy.data.materials.new("SunOrbMat")
    s_mat.use_nodes = True
    s_nt = s_mat.node_tree
    s_nt.nodes.clear()
    s_out = s_nt.nodes.new("ShaderNodeOutputMaterial")
    em = s_nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.42, 0.08, 1.0)
    em.inputs["Strength"].default_value = 8.0
    s_nt.links.new(em.outputs["Emission"], s_out.inputs["Surface"])
    sun_orb.data.materials.append(s_mat)

    # Directional Sun Lamp pointing from sun_pos toward origin
    light_data = bpy.data.lights.new(name="KeySun", type="SUN")
    light_data.energy = sun_energy
    light_data.color = (1.0, 0.86, 0.68)
    light_data.angle = math.radians(3.5)
    light_obj = bpy.data.objects.new(name="KeySun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    direction = Vector((0.0, 4.0, 0.0)) - Vector(sun_pos)
    light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(scene, focus_dist, fstop=2.2):
    cam_data = bpy.data.cameras.new("MainCam")
    cam_data.lens = 42.0
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = focus_dist
    cam_data.dof.aperture_fstop = fstop
    cam_obj = bpy.data.objects.new("MainCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -10.5, 1.45)
    direction = Vector((0.0, 4.0, 1.25)) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam_obj
    return cam_obj


def render_shot(filename, setup_fn):
    clear_scene()
    scene = bpy.context.scene
    setup_render(scene)
    build_dune_mesh()
    build_ocean()
    setup_fn(scene)
    scene.render.filepath = os.path.join(OUT_DIR, filename)
    bpy.ops.render.render(write_still=True)
    print("Saved:", scene.render.filepath)


def shot_mushrooms(scene):
    add_sun_and_sky(scene, sun_pos=(-0.3, 24.0, 6.2), sky_horizon=(0.72, 0.56, 0.42), sky_zenith=(0.24, 0.28, 0.34), sun_energy=4.2)
    # Place 5 mushrooms across near, mid, and far dune depths for clear DoF separation
    add_sprite_card("Mush_1", MUSHROOM_PNG, -5.2, 4.5, height_world=2.7, sink=0.16)
    add_sprite_card("Mush_2", MUSHROOM_PNG, -2.6, 8.2, height_world=2.4, sink=0.16)
    add_sprite_card("Mush_3", MUSHROOM_PNG, 0.2, 5.8, height_world=2.5, sink=0.16)
    add_sprite_card("Mush_4_Hero", MUSHROOM_PNG, 2.7, 2.2, height_world=3.2, sink=0.18)
    add_sprite_card("Mush_5", MUSHROOM_PNG, 5.4, 7.5, height_world=2.3, sink=0.16)
    # Focus on hero mushroom at y=2.2 (cam at y=-10.5 -> ~12.7m)
    add_camera(scene, focus_dist=12.7, fstop=1.8)


def shot_forest(scene):
    add_sun_and_sky(scene, sun_pos=(-7.5, 25.0, 2.6), sky_horizon=(0.76, 0.48, 0.30), sky_zenith=(0.20, 0.25, 0.33), sun_energy=3.8)
    add_sprite_card("Pine_1_FG", PINE_WARM, -5.4, 2.8, height_world=2.4, sink=0.18)
    add_sprite_card("Pine_2_Hero", PINE_DEEP, -2.3, 4.0, height_world=5.6, sink=0.22)
    add_sprite_card("Pine_3_Mid", PINE_COOL, 0.9, 6.8, height_world=3.4, sink=0.18)
    add_sprite_card("Pine_4_Mid", PINE_WARM, 3.6, 5.2, height_world=3.9, sink=0.18)
    add_sprite_card("Pine_5_Far", PINE_COOL, 6.4, 9.5, height_world=2.8, sink=0.18)
    add_camera(scene, focus_dist=14.5, fstop=2.0)


def shot_dinos(scene):
    add_sun_and_sky(scene, sun_pos=(4.8, 24.0, 5.8), sky_horizon=(0.70, 0.54, 0.40), sky_zenith=(0.22, 0.27, 0.34), sun_energy=4.5)
    add_sprite_card("Pine_1", PINE_WARM, -5.6, 3.0, height_world=2.2, sink=0.28)
    add_sprite_card("Pine_2", PINE_DEEP, -2.7, 4.2, height_world=5.4, sink=0.35)
    add_sprite_card("Pine_3", PINE_COOL, 0.5, 7.2, height_world=3.2, sink=0.28)
    add_sprite_card("Pine_4", PINE_WARM, 3.4, 6.0, height_world=3.4, sink=0.28)
    add_sprite_card("Sauropod", DINO_SAURO, 3.1, 3.0, height_world=2.5, sink=0.10, solid_body=True)
    add_sprite_card("Hadrosaur", DINO_HADRO, 5.6, 3.4, height_world=1.7, sink=0.10, solid_body=True)
    add_sprite_card("Ankylosaur", DINO_ANKY, 7.6, 4.0, height_world=1.2, sink=0.08, solid_body=True)
    # Focus on dinosaur caravan at y=3.2 (~13.7m)
    add_camera(scene, focus_dist=13.7, fstop=2.2)


if __name__ == "__main__":
    render_shot("03_dinos_4m22_sun_high_right.png", shot_dinos)
