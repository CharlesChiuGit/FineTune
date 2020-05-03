# renderer/__init__.py

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, RasterizationSettings, MeshRenderer, MeshRasterizer,
    DirectionalLights, TexturedSoftPhongShader
)

from config import FOV, EPOCH, SIZE, DEVICE
from helper import camera_direction, get_cpu_image


def build_renderer():
    # Initialize an OpenGL perspective camera.
    cameras = OpenGLPerspectiveCameras(device=DEVICE, degrees=True, fov=FOV, znear=1e-4, zfar=100)

    raster_settings = RasterizationSettings(image_size=SIZE, blur_radius=0.0, faces_per_pixel=1, bin_size=0)

    lights = DirectionalLights(device=DEVICE, direction=((-40, 200, 100),), ambient_color=((0.5, 0.5, 0.5),),
                               diffuse_color=((0.5, 0.5, 0.5),), specular_color=((0.0, 0.0, 0.0),), )

    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                            shader=TexturedSoftPhongShader(device=DEVICE, cameras=cameras, lights=lights))

    return renderer


def render_single_image(mesh, renderer, pos):
    # print(pos)
    u_x, u_y, u_z = camera_direction(pos[0][0], pos[0][1], pos[0][2], 0, 0, 0)
    R, T = look_at_view_transform(eye=pos,
                                  at=((0, 0, 0),),
                                  up=((u_x, u_y, u_z),), device=DEVICE)
    image = renderer(meshes_world=mesh.clone(), R=R, T=T)

    image = get_cpu_image(image)

    return image
