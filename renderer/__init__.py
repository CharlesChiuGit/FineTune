# renderer/__init__.py

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, RasterizationSettings, MeshRenderer, MeshRasterizer,
    DirectionalLights, TexturedSoftPhongShader
)

from config import FOV, EPOCH, DEVICE
from helper import camera_direction, get_cpu_image


def build_renderer(_image_size):
    # Initialize an OpenGL perspective camera.
    cameras = OpenGLPerspectiveCameras(device=DEVICE, degrees=True, fov=FOV, znear=1e-4, zfar=100)

    raster_settings = RasterizationSettings(image_size=_image_size, blur_radius=0.0, faces_per_pixel=1, bin_size=0)

    lights = DirectionalLights(device=DEVICE, direction=((-40, 200, 100),), ambient_color=((0.5, 0.5, 0.5),),
                               diffuse_color=((0.5, 0.5, 0.5),), specular_color=((0.0, 0.0, 0.0),), )

    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                            shader=TexturedSoftPhongShader(device=DEVICE, cameras=cameras, lights=lights))

    return renderer


def render_single_image(mesh, renderer, pos):
    # print(pos)
    R, T = look_at_view_transform(dist=pos[0], elev=pos[1], azim=pos[2],
                                  at=((pos[3], pos[4], pos[5]),),
                                  up=((pos[6], pos[7], pos[8]),),
                                  device=DEVICE)
    image = renderer(meshes_world=mesh.clone(), R=R, T=T)

    image = get_cpu_image(image)  # (size, size, 3)

    return image
