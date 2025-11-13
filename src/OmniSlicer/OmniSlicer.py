import os
import torch
import trimesh
import numpy as np
import pyvista as pv
import torchio as tio
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms

__all__ = ["extract_slices"]

def _calculate_rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Compute the cross product (axis of rotation)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # If the vectors are already the same, no rotation is needed
    if axis_norm == 0:
        # print("Vectors are the same!")
        return np.eye(3)

    # Normalize the axis of rotation
    axis = axis / axis_norm

    # Compute the angle between the vectors
    cos_theta = np.dot(v1, v2)
    sin_theta = axis_norm  # This is the norm of the cross product

    # Compute the skew-symmetric matrix K
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    # Rodrigues' rotation formula
    R = np.eye(3) + np.sin(np.arccos(cos_theta)) * K + (1 - cos_theta) * np.dot(K, K)

    return R

def _create_rotation_matrices(vertices: np.ndarray):

    rot_matrices = []
    if vertices.shape[0] == 1:
        pass
    
    elif vertices.shape[0] == 3:
        rot_matrices.append(np.array([[ 0.,  0., -1.],
                                      [ 0.,  1.,  0.],
                                      [ 1.,  0.,  0.]]))
        
        rot_matrices.append(np.array([[ 1.,  0.,  0.],
                                      [ 0.,  0., -1.],
                                      [ 0.,  1.,  0.]]))        
                                    
    else:
        assert vertices.shape[0] >= 8, "Please make sure number of views is greater than 8 or equal to 1 or 3!"
    
        origin = np.array(vertices[0])

        
        for i in range(1, len(vertices)):
            rot_matrices.append(_calculate_rotation_matrix(origin, np.array(vertices[i])))
        
    return rot_matrices

def _find_largest_lesion_slice(mask: torch.Tensor, axis: int) -> int:
    assert mask.ndim == 3, "Mask must be a 3D tensor"
    assert axis in [0, 1, 2], "Axis must be 0, 1, or 2"

    # Move the desired axis to the front
    slices = mask.moveaxis(axis, 0)

    # Compute lesion area (non-zero count) per slice
    areas = torch.sum(slices != 0, dim=(1, 2))

    # Get index of maximum area
    max_index = torch.argmax(areas).item()
    return max_index

def _rotate_3d_tensor_around_center(tensor: torch.Tensor, rotation_matrix: torch.Tensor, order: int = 1, device: str = 'cuda'):
    """
    Rotate a 3D torch tensor around its center using a given rotation matrix.

    Parameters:
    - tensor: 3D torch tensor to rotate.
    - rotation_matrix: 3x3 torch tensor representing the rotation matrix.
    - order: Interpolation order (0: nearest, 1: linear).  Defaults to 1.

    Returns:
    - rotated_tensor: 3D torch tensor after rotation.
    """
    # Validate inputs
    if tensor.ndim != 3:
        raise ValueError("Input tensor must be 3D.")
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if order not in [0, 1]:
        raise ValueError("Order must be 0 (nearest) or 1 (linear).")

    # Compute the center of the tensor
    center = torch.tensor(tensor.shape, dtype=torch.float32) / 2.0

    # Create the affine transformation matrix
    affine_matrix = torch.eye(4)
    affine_matrix[:3, :3] = rotation_matrix

    # Translate to origin, apply rotation, and translate back
    translation_to_origin = torch.eye(4)
    translation_to_origin[:3, 3] = -center

    translation_back = torch.eye(4)
    translation_back[:3, 3] = center

    # Combine transformations: T_back * R * T_origin
    combined_transform = translation_back @ affine_matrix @ translation_to_origin

    # Create a meshgrid of coordinates for the original volume
    d_coords = torch.arange(tensor.shape[0], dtype=torch.float32)
    h_coords = torch.arange(tensor.shape[1], dtype=torch.float32)
    w_coords = torch.arange(tensor.shape[2], dtype=torch.float32)
    grid = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
    coords = torch.stack(grid, dim=-1)  # Shape: (D, H, W, 3)

    # Reshape to (D*H*W, 3) for matrix multiplication
    original_coords_flat = coords.reshape(-1, 3)

    # Add a homogeneous coordinate (1) to each point
    ones = torch.ones(original_coords_flat.shape[0], 1)
    original_coords_homogeneous = torch.cat((original_coords_flat, ones), dim=1)  # (D*H*W, 4)

    # Apply the inverse transformation to get source coordinates
    # We use the inverse because grid_sample samples *from* the input
    # at locations given by the output (transformed) coordinates.
    transformed_coords_homogeneous = original_coords_homogeneous @ torch.inverse(combined_transform).T

    # Extract the spatial coordinates (x, y, z)
    transformed_coords = transformed_coords_homogeneous[:, :3]

    # Normalize to the range [-1, 1] for grid_sample
    normalized_coords_d = 2 * transformed_coords[:, 0] / (tensor.shape[0] - 1) - 1
    normalized_coords_h = 2 * transformed_coords[:, 1] / (tensor.shape[1] - 1) - 1
    normalized_coords_w = 2 * transformed_coords[:, 2] / (tensor.shape[2] - 1) - 1

    # Create the sampling grid for grid_sample
    sampling_grid = torch.stack((normalized_coords_w, normalized_coords_h, normalized_coords_d), dim=-1)
    sampling_grid = sampling_grid.reshape(1, tensor.shape[0], tensor.shape[1], tensor.shape[2], 3)

    # Use grid_sample to perform the rotation
    mode = 'bilinear' if order == 1 else 'nearest'
    rotated_tensor = F.grid_sample(
        tensor.unsqueeze(0).unsqueeze(0),  # Add batch dimension for grid_sample
        sampling_grid.to(device),
        mode=mode,
        padding_mode='zeros',
        align_corners=True
    ).squeeze(0) # Remove batch dimension

    return rotated_tensor.squeeze(0).cpu()  # Remove channel dimension

def _crop_to_square(array, mask):
    """
    Crops a 2D array to a square region around the ROI defined by a binary mask.

    Parameters:
        array (np.ndarray): The input 2D array.
        mask (np.ndarray): The binary mask with the ROI (same shape as the array).

    Returns:
        np.ndarray: The cropped square array.
    """
    # Ensure the mask is binary
    mask = mask > 0

    # Find the bounding box of the ROI
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Calculate width and height of the bounding box
    height = row_max - row_min + 1
    width = col_max - col_min + 1

    # Determine the size of the square
    square_size = max(height, width)
    square_size = int(square_size * 1.1)

    # Center the square around the bounding box
    center_row = (row_min + row_max) // 2
    center_col = (col_min + col_max) // 2

    # Calculate new boundaries for the square
    half_size = square_size // 2
    square_row_min = max(0, center_row - half_size)
    square_row_max = min(array.shape[0], center_row + half_size)
    square_col_min = max(0, center_col - half_size)
    square_col_max = min(array.shape[1], center_col + half_size)

    # Adjust boundaries to ensure the square size
    if square_row_max - square_row_min < square_size:
        if square_row_min == 0:
            square_row_max = min(array.shape[0], square_row_min + square_size)
        else:
            square_row_min = max(0, square_row_max - square_size)
    if square_col_max - square_col_min < square_size:
        if square_col_min == 0:
            square_col_max = min(array.shape[1], square_col_min + square_size)
        else:
            square_col_min = max(0, square_col_max - square_size)

    # Crop the square region
    cropped_array = array[square_row_min:square_row_max, square_col_min:square_col_max]

    return cropped_array

def _pad_to_square(array, padding_value=0):
    """
    Pads a 2D array to make it square by adding padding with the specified value.

    Parameters:
        array (np.ndarray): The input 2D array.
        padding_value: The value used for padding.

    Returns:
        np.ndarray: The padded square array.
    """
    rows, cols = array.shape
    size = max(rows, cols)  # Determine the size for the square matrix

    # Calculate padding for rows and columns
    pad_rows = size - rows
    pad_cols = size - cols

    # Pad the array using np.pad
    padded_array = np.pad(
        array,
        ((0, pad_rows), (0, pad_cols)),  # (top/bottom padding, left/right padding)
        mode='constant',
        constant_values=padding_value
    )
    return padded_array

def _create_sphere(n_views: int, output_dir: str, save_sphere: bool = True):
    
    def normalize(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def random_points_on_sphere(n):
        """Uniform random points on unit sphere."""
        vec = np.random.randn(n, 3)
        return normalize(vec)

    def coulomb_repulsion(points, fixed_mask, lr=0.01, steps=2000):
        """
        Optimize free points on a sphere by minimizing Coulomb energy.
        points: (N,3) array on sphere
        fixed_mask: boolean array, True for fixed points
        """
        n = len(points)
        pts = points.copy()

        for step in range(steps):
            forces = np.zeros_like(pts)

            for i in range(n):
                for j in range(i+1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    f = diff / (dist**3 + 1e-9)  # Coulomb force
                    forces[i] += f
                    forces[j] -= f

            # Update only free points
            pts[~fixed_mask] += lr * forces[~fixed_mask]
            pts[~fixed_mask] = normalize(pts[~fixed_mask])

        return pts
    
    N = n_views

    if not os.path.exists(os.path.join(output_dir, f"sphere_{N}_views.pt")):
        print(f"Creating sphere with {N} views...")

        # Fixed points
        fixed_points = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        fixed_mask = np.array([True, True, True] + [False]*(N-3))

        # Initialize
        free_points = random_points_on_sphere(N-3)
        points_init = np.vstack([fixed_points, free_points])

        # Optimize
        points_opt = coulomb_repulsion(points_init, fixed_mask, lr=0.01, steps=10000)

        point_cloud = pv.PolyData(points_opt)
        mesh = point_cloud.delaunay_3d()
        surf = mesh.extract_surface()
        vertices = surf.points
        faces = surf.faces.reshape(-1, 4)[:, 1:]

        vertices_faces = {"vertices": vertices, "faces": faces}
        torch.save(vertices_faces, os.path.join(output_dir, f"sphere_{N}_views.pt"))
    
    else:
        print(f"Loading sphere with {N} views...")
        vertices_faces = torch.load(os.path.join(output_dir, f"sphere_{N}_views.pt"), weights_only=False)
        vertices = vertices_faces["vertices"]
        faces = vertices_faces["faces"]

    if save_sphere:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces) 
        mesh.export(os.path.join(output_dir, f"sphere_{N}_views.ply"))

    return vertices, faces

def extract_slices(volume_path: str = None, mask_path: str = None, output_dir: str = None, n_views: int = None):

    assert volume_path is not None, "Please provide a valid path to the 3D volume."
    assert os.path.exists(volume_path), f"The specified volume path does not exist: {volume_path}"
    assert mask_path is not None, "Please provide a valid path to the 3D mask."
    assert os.path.exists(mask_path), f"The specified mask path does not exist: {mask_path}"
    assert output_dir is not None, "Please provide a valid output directory."
    assert n_views is not None, "Please provide the number of views for OmniSlicer."
    assert n_views >= 3, "Number of views must be at least 3."
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a CUDA-capable GPU."

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
          
    vertices, _ = _create_sphere(n_views=n_views, output_dir=output_dir)
    rot_matrices = _create_rotation_matrices(vertices=vertices)  

    filename = os.path.basename(volume_path).split(".")[0]  

    img = tio.ScalarImage(volume_path)
    seg = tio.LabelMap(mask_path)
    seg = tio.Resample(target=img)(seg)
    subject = tio.Subject(image=img, mask=seg)
    subject = tio.ToCanonical()(subject)
    subject = tio.Resample((1.0, 1.0, 1.0))(subject)

    # Crop cube around lesion with margin
    subject_temp = tio.CropOrPad(mask_name="mask")(subject)
    max_dim = np.max(subject_temp.image.shape)
    target_dim = int(max_dim * np.sqrt(2))
    subject = tio.CropOrPad(target_shape=(target_dim, target_dim, target_dim), mask_name="mask")(subject)

    img_slices = []
    seg_slices = []

    img_tensor = subject.image.tensor[0]
    seg_tensor = subject.mask.tensor[0]

    idx_slice = _find_largest_lesion_slice(seg_tensor, axis=2)
    img_slices.append(img_tensor[:, :, idx_slice])
    seg_slices.append(seg_tensor[:, :, idx_slice])

    for rot_matrix in rot_matrices:
        img_tensor_rotated = _rotate_3d_tensor_around_center(img_tensor.to(torch.float32).cuda(), torch.tensor(rot_matrix, dtype=torch.float32).cuda(), order=1, device='cuda')
        seg_tensor_rotated = _rotate_3d_tensor_around_center(seg_tensor.to(torch.float32).cuda(), torch.tensor(rot_matrix, dtype=torch.float32).cuda(), order=0, device='cuda')
        idx_slice = _find_largest_lesion_slice(seg_tensor_rotated, axis=2)
        img_slices.append(img_tensor_rotated[:, :, idx_slice])
        seg_slices.append(seg_tensor_rotated[:, :, idx_slice])

    i = 0
    for img_slice, seg_slice in tqdm(zip(img_slices, seg_slices), total=len(img_slices), desc="Saving slices"):
        img_slice = img_slice.numpy()
        img_slice = _crop_to_square(img_slice, mask=seg_slice.numpy())
        # img_slice = _pad_to_square(img_slice, padding_value=0)
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        img_slice = (img_slice * 255).astype(np.uint8)
        img_pil = transforms.ToPILImage()(img_slice)
        img_pil.save(os.path.join(output_dir, f"{filename}_omnislicer_output_image_{i}.png"), format="PNG")
        i += 1        
    
    print("\n")
    print("#####################################################################################")
    print(f"[INFO] Extracted {len(img_slices)} omnidirectional slices and saved to {output_dir}.")
    print("#####################################################################################")
    print("\n")