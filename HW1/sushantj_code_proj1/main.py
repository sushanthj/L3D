from utils import *

def render_cow_custom_camera(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    # NOTE: Notice how io.load_objs_as_meshes is different than creating our own mesh using
    #       pytorch3d.structures.Meshes
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    # Redfining R_relative and T_relative according to requirements
    ######## 1. 90 degree counterclock rotation of camera about z ################
    # R_relative = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # T_relative = [0, 0, 0]

    ######## 2. 90 degree counterclock rotation of camera about y ################
    # R_relative = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    # T_relative = [3, 0, 3]

    ######## 3. Move camera away from the cow ################
    # R_relative = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # T_relative = [0, 0, 3]

    ######## 4. Move camera sideways from the cow and lift up ################
    theta = np.radians(10)
    R_relative = [[1, 0, 0], [0, np.cos(-theta), -np.sin(-theta)], [0, np.sin(-theta), np.cos(-theta)]]
    T_relative = [0.5, -0.5, 0]

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)

    fig = plot_scene({
        "figure": {
            "Mesh": meshes,
            "Camera": cameras,
        }
    })
    # fig.show()

    return rend[0, ..., :3].cpu().numpy()

def load_rgbd_data(path="data/rgbd_data.pkl"):
    """
    rgb_data.keys() = dict_keys(['rgb1', 'mask1', 'depth1',
                                'rgb2', 'mask2', 'depth2',
                                'cameras1', 'cameras2'])
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def dolly_zoom(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)

    """
    - FOV and focal length are inversely proportional.
    - As the FOV increases, the focal length decreases.
    - Hence, to maintain the object size in the image, we need to move the camera closer to the object.
    - Formula is distance = distance * (tan(prev_fov/2) / tan(fov/2))
    """
    distance = 50.0

    renders = []
    for i in tqdm(range(1,num_frames)):
        fov = fovs[i].cpu().item()
        prev_fov = fovs[i-1].cpu().item()
        # Calculate the new distance to maintain object size constant
        distance = distance * (np.tan(np.radians(prev_fov/2)) / np.tan(np.radians(fov/2)))
        T = [[0, 0, distance]]
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh.extend(num_frames), cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, fps=(num_frames / duration), loop=3)


def render_first_pointcloud(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color)
    rgb_data = load_rgbd_data()

    points, rgb = unproject_depth_image(rgb_data.get("rgb1"), rgb_data.get("mask1"),
                                        rgb_data.get("depth1"), rgb_data.get("cameras1"))
    # extend points to 12 views
    num_views = 12
    points = points.unsqueeze(0).to(device)
    rgb = rgb.unsqueeze(0).to(device)
    points = points.expand(num_views, -1, -1)
    rgb = rgb.expand(num_views, -1, -1)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 180 degrees about the z-axis
    additional_rotation = torch.tensor([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]], dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/first_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0], points, rgb


def render_second_pointcloud(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color)
    rgb_data = load_rgbd_data()

    points, rgb = unproject_depth_image(rgb_data.get("rgb2"), rgb_data.get("mask2"),
                                        rgb_data.get("depth2"), rgb_data.get("cameras2"))
    # extend points to 12 views
    num_views = 12
    points = points.unsqueeze(0).to(device)
    rgb = rgb.unsqueeze(0).to(device)
    points = points.expand(num_views, -1, -1)
    rgb = rgb.expand(num_views, -1, -1)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 180 degrees about the z-axis
    additional_rotation = torch.tensor([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]], dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/second_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0], points, rgb


def render_composite_cloud(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color)
    _, pts1, features1 = render_first_pointcloud(image_size=image_size, device=device)
    _, pts2, features2 = render_second_pointcloud(image_size=image_size, device=device)

    num_views = 12

    combined_points = torch.cat([pts1, pts2], dim=1)
    combined_features = torch.cat([features1, features2], dim=1)
    combined_point_cloud = pytorch3d.structures.Pointclouds(points=combined_points, features=combined_features)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 180 degrees about the z-axis
    additional_rotation = torch.tensor([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]], dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(combined_point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/composite_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_torus(image_size=256, num_samples=200, device=None):
    """
    Renders a Torus
    """
    start = time.time()
    background_color=(1, 1, 1)
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
    image_size=image_size, background_color=background_color)
    R = 1
    r = 0.5
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta)
    x = (R + r * torch.cos(Phi)) * torch.cos(Theta)
    y = (R + r * torch.cos(Phi)) * torch.sin(Theta)
    z = r * torch.sin(Phi)

    num_views = 12
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    points = points.unsqueeze(0).expand(num_views, -1, -1)
    color = color.unsqueeze(0).expand(num_views, -1, -1)

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=points, features=color,)
    print("Memory occupied by torus_point_cloud in MB: ", sys.getsizeof(torus_point_cloud) / (1024 * 1024))
    torus_point_cloud = torus_point_cloud.to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 20 degrees about the x-axis
    additional_rotation = torch.tensor([[1,  0, 0],
                                        [0, np.cos(np.radians(20)), np.sin(np.radians(20))],
                                        [0,  -np.sin(np.radians(20)), np.cos(np.radians(20))]],
                                        dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(torus_point_cloud, cameras=many_cameras)
    end = time.time()
    print("Time taken to render parametric torus: ", end - start)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/torus_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]

    """
    Memory occupied by torus_point_cloud in MB:  4.57763671875e-05
    Time taken to render parametric torus:  0.31734442710876465
    """


def render_custom_shape(image_size=256, num_samples=200, device=None):
    """
    Renders a new shape
    """
    background_color=(1, 1, 1)
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
    image_size=image_size, background_color=background_color)

    def hyperboloid(u, v, a=1):
        x = a * np.cosh(u) * np.cos(v)
        y = a * np.cosh(u) * np.sin(v)
        z = a * np.sinh(u)
        return x, y, z

    # Generate u and v values
    u = torch.linspace(-2, 2, 100)
    v = torch.linspace(0, 2 * np.pi, 100)
    u, v = torch.meshgrid(u, v)

    # Generate hyperboloid points
    x, y, z = hyperboloid(u, v)

    num_views = 12
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    points = points.unsqueeze(0).expand(num_views, -1, -1)
    color = color.unsqueeze(0).expand(num_views, -1, -1)

    hyperboloid_point_cloud = pytorch3d.structures.Pointclouds(
        points=points, features=color,
    ).to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 10 degrees about the x-axis
    additional_rotation = torch.tensor([[1,  0, 0],
                                        [0, np.cos(np.radians(80)), np.sin(np.radians(80))],
                                        [0,  -np.sin(np.radians(80)), np.cos(np.radians(80))]],
                                        dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(hyperboloid_point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/custom_hyperbolloid_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def render_torus_implicit(image_size=256, voxel_size=64, device=None):
    start = time.time()
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    R = 0.6
    r = 0.4
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - R) ** 2 + Z ** 2 - r ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())

    num_views = 12
    vertices = vertices.unsqueeze(0).expand(num_views, -1, -1)
    faces = faces.unsqueeze(0).expand(num_views, -1, -1)
    textures = textures.unsqueeze(0).expand(num_views, -1, -1)

    textures = pytorch3d.renderer.TexturesVertex(vertices)

    mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures)
    print("Memory occupied by mesh in MB: ", sys.getsizeof(mesh) / (1024 * 1024))
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 10 degrees about the x-axis
    additional_rotation = torch.tensor([[1,  0, 0],
                                        [0, np.cos(np.radians(20)), np.sin(np.radians(20))],
                                        [0,  -np.sin(np.radians(20)), np.cos(np.radians(20))]],
                                        dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(mesh, cameras=many_cameras, lights=lights)
    end = time.time()
    print("Time taken to render implicit torus: ", end - start)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/implicit_torus_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]

    """
    Memory occupied by mesh in MB:  4.57763671875e-05
    Time taken to render implicit torus:  2.4901247024536133
    """


def render_custom_implicit(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    a = 0.6
    b = 0.6
    c = 0.8
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 / a ** 2 + Y ** 2 / b ** 2 - Z ** 2 / c ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())

    num_views = 12
    vertices = vertices.unsqueeze(0).expand(num_views, -1, -1)
    faces = faces.unsqueeze(0).expand(num_views, -1, -1)
    textures = textures.unsqueeze(0).expand(num_views, -1, -1)

    textures = pytorch3d.renderer.TexturesVertex(vertices)

    mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 10 degrees about the x-axis
    additional_rotation = torch.tensor([[1,  0, 0],
                                        [0, np.cos(np.radians(20)), np.sin(np.radians(20))],
                                        [0,  -np.sin(np.radians(20)), np.cos(np.radians(20))]],
                                        dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(mesh, cameras=many_cameras, lights=lights)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/implicit_hyperbolloid_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

def render_cow_multiple_views(
    cow_path="data/cow.obj",
    image_size=256,
    color=[0.7, 0.7, 1],
    device=None):
        if device is None:
            device = get_device()
        # Get the renderer.
        renderer = get_mesh_renderer(image_size=image_size)

        # Get the vertices, faces, and textures.
        vertices, faces = load_cow_mesh(cow_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor(color)  # (1, N_v, 3)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)

        num_views = 12
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3,
            elev=0,
            azim=np.linspace(-180, 180, num_views, endpoint=False),
        )
        many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
        fig, axs = plt.subplots(3, 4)
        axs = axs.flatten()
        for i, image in enumerate(images):
            ax = axs[i]
            ax.imshow(image.cpu())
            ax.axis("off")

        plt.savefig('images/trial_image.png')
        # Optionally, show the figure
        # plt.show()
        images = images.cpu().numpy()
        # img_list = [img.squeeze().astype(np.uint8) for img in images]
        img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
        imageio.mimsave('images/cow_360.gif', img_list, loop=10, duration = 0.05)
        plt.close()

def render_tetrahedron(image_size=256, color=[0.7, 0.7, 1], device=None):
    if device is None:
            device = get_device()
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    vertices = torch.tensor([[3,0,0],[0,0,3],[-2,0,-2],[0,3,0]], dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=torch.int64).unsqueeze(0)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    num_views = 12
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=3,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    fig, axs = plt.subplots(3, 4)
    axs = axs.flatten()
    for i, image in enumerate(images):
        ax = axs[i]
        ax.imshow(image.cpu())
        ax.axis("off")

    # Save the figure to a file (adjust the file format as needed)
    plt.savefig('images/tetrahedron.png')
    # Optionally, show the figure
    # plt.show()
    images = images.cpu().numpy()
    # img_list = [img.squeeze().astype(np.uint8) for img in images]
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]

    imageio.mimsave('images/tetrahedron_360.gif', img_list, loop=10, duration = 0.05)
    plt.close()


def render_cube(image_size=256, color=[0.7, 0.7, 1], device=None):
    if device is None:
            device = get_device()
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # define the vertices for a cube
    cube_coords = [
        [3, 3, 3],
        [3, 3, -3],
        [3, -3, 3],
        [3, -3, -3],
        [-3, 3, 3],
        [-3, 3, -3],
        [-3, -3, 3],
        [-3, -3, -3],
    ]
    cube_faces = [
        [0, 1, 2],
        [1, 2, 3],
        [4, 5, 6],
        [5, 6, 7],
        [0, 1, 4],
        [1, 4, 5],
        [2, 3, 6],
        [3, 6, 7],
        [0, 2, 4],
        [2, 4, 6],
        [1, 3, 5],
        [3, 5, 7],
    ]
    vertices = torch.tensor(cube_coords, dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor(cube_faces, dtype=torch.int64).unsqueeze(0)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    num_views = 12
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=3,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    fig, axs = plt.subplots(3, 4)
    axs = axs.flatten()
    for i, image in enumerate(images):
        ax = axs[i]
        ax.imshow(image.cpu())
        ax.axis("off")

    # Save the figure to a file (adjust the file format as needed)
    plt.savefig('images/cube.png')
    # Optionally, show the figure
    # plt.show()
    images = images.cpu().numpy()
    # img_list = [img.squeeze().astype(np.uint8) for img in images]
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]

    imageio.mimsave('images/cube_360.gif', img_list, loop=10, duration = 0.05)
    plt.close()


def render_bb8(
    cow_path="data/bb8.obj",
    image_size=256,
    color=[0.7, 0.7, 1],
    device=None):
        if device is None:
            device = get_device()
        # Get the renderer.
        renderer = get_mesh_renderer(image_size=image_size, bin_size=0)

        # Get the vertices, faces, and textures.
        vertices, faces = load_cow_mesh(cow_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor(color)  # (1, N_v, 3)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)

        num_views = 12
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=10,
            elev=50,
            azim=np.linspace(-180, 180, num_views, endpoint=False),
        )
        many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
        fig, axs = plt.subplots(3, 4)
        axs = axs.flatten()
        for i, image in enumerate(images):
            ax = axs[i]
            ax.imshow(image.cpu())
            ax.axis("off")

        plt.savefig('images/trial_image.png')
        # Optionally, show the figure
        # plt.show()
        images = images.cpu().numpy()
        # img_list = [img.squeeze().astype(np.uint8) for img in images]
        img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
        imageio.mimsave('images/bb8_360.gif', img_list, loop=10, duration = 0.05)
        plt.close()


def retexture_cow(
    cow_path="data/cow.obj",
    image_size=256,
    color1=[0.0, 0.4, 0.6],
    color2=[0.6, 0.4, 0.0],
    device=None):
        """
        color1=[0.0, 0.4, 0.6],
        color2=[0.6, 0.4, 0.0],

        Both are blended as per the z value of the vertices
        """
        if device is None:
            device = get_device()
        # Get the renderer.
        renderer = get_mesh_renderer(image_size=image_size)

        # Get the vertices, faces, and textures.
        vertices, faces = load_cow_mesh(cow_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

        ######### Blend color according to z value of the vertices #######
        textures = vertices[0].clone()
        z_min = vertices[..., 2].min().cpu().item()
        z_max = vertices[..., 2].max().cpu().item()
        print("Z min:", z_min, "Z max:", z_max)
        color1 = torch.tensor(color1, dtype=torch.float32).to(device)
        color2 = torch.tensor(color2, dtype=torch.float32).to(device)
        alpha = lambda z: (z - z_min) / (z_max - z_min)
        color = lambda alpha: alpha * color2 + (1 - alpha) * color1

        for i in range(textures.shape[0]): # textures shape = (N_v, 3)
            textures[i] = color(alpha(textures[i, 2].cpu().item()))
        textures = textures.unsqueeze(0)
        ##################################################################

        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)

        num_views = 12
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3,
            elev=0,
            azim=np.linspace(-180, 180, num_views, endpoint=False),
        )
        many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
        fig, axs = plt.subplots(3, 4)
        axs = axs.flatten()
        for i, image in enumerate(images):
            ax = axs[i]
            ax.imshow(image.cpu())
            ax.axis("off")

        plt.savefig('images/trial_image.png')
        # Optionally, show the figure
        # plt.show()
        images = images.cpu().numpy()
        # img_list = [img.squeeze().astype(np.uint8) for img in images]
        img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
        imageio.mimsave('images/cow_retextured_360.gif', img_list, loop=10, duration = 0.05)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="images/cow.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    args = parser.parse_args()

    # Mesh Rendering
    image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    plt.imsave(args.output_path, image)

    render_cow_multiple_views(cow_path=args.cow_path, image_size=args.image_size)
    render_tetrahedron(image_size=args.image_size)
    retexture_cow(cow_path=args.cow_path, image_size=args.image_size)

    # Camera Transforms
    plt.imsave(args.output_path, render_cow_custom_camera(cow_path=args.cow_path, image_size=args.image_size))

    # Point Clouds, Parametric, and Implicit Surfaces Rendering
    image = render_bridge(image_size=args.image_size)
    plt.imsave(args.output_path, image)

    image, _, _ = render_first_pointcloud(image_size=args.image_size)
    image, _, _ = render_second_pointcloud(image_size=args.image_size)
    _ = render_composite_cloud(image_size=args.image_size)
    _ = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    _ = render_torus(image_size=args.image_size, num_samples=args.num_samples)
    _ = render_custom_shape(image_size=args.image_size, num_samples=args.num_samples)
    _ = render_sphere_mesh(image_size=args.image_size)
    _ = render_torus_implicit(image_size=args.image_size)
    _ = render_custom_implicit(image_size=args.image_size)


    # Dolly Zoom
    dolly_zoom(
        image_size=args.image_size,
        num_frames=30,
        duration=3,
        output_file="images/dolly_zoom.gif",
    )

    # Render Something For fun (the rendering is not stable due to large/complex geometry)
    # render_bb8(image_size=args.image_size)