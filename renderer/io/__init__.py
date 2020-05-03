# renderer.io/__init__.py

# io utils
from pytorch3d.io import load_obj
# datastructures
from pytorch3d.structures import Meshes, Textures
from config import DEVICE


def load_moon_mesh(_obj_filename):
    # Load the object
    verts, faces, aux = load_obj(_obj_filename)
    faces_idx = faces.verts_idx.to(DEVICE)
    verts = verts.to(DEVICE)

    verts_uvs = aux.verts_uvs[None, ...].to(DEVICE)
    faces_uvs = faces.textures_idx[None, ...].to(DEVICE)
    tex_maps = aux.texture_images
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...].to(DEVICE)
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
    moon_mesh = Meshes(verts=[verts], faces=[faces_idx], textures=tex)

    return moon_mesh
