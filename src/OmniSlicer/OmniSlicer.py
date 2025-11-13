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
    """
    Calculate the rotation matrix that rotates vector v1 to align with vector v2.
    This function computes the 3D rotation matrix using Rodrigues' rotation formula.
    The rotation is performed around the axis perpendicular to both input vectors,
    with the minimum angle required to align v1 with v2.
    Args:
        v1 (np.ndarray): Source vector of shape (3,) to be rotated.
        v2 (np.ndarray): Target vector of shape (3,) to align with.
    Returns:
        np.ndarray: 3x3 rotation matrix that transforms v1 to align with v2.
                   Returns identity matrix if vectors are already aligned.
    Notes:
        - Input vectors are automatically normalized before computation.
        - Uses Rodrigues' rotation formula: R = I + sin(θ)K + (1-cos(θ))K²
        - Where K is the skew-symmetric matrix of the rotation axis.
        - Handles edge case where vectors are identical (returns identity matrix).
    Examples:
        >>> v1 = np.array([1, 0, 0])
        >>> v2 = np.array([0, 1, 0])
        >>> R = _calculate_rotation_matrix(v1, v2)
        >>> # R rotates v1 to align with v2
    """



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
    """
    Create rotation matrices based on the provided vertices configuration.
    This function generates rotation matrices for different viewing configurations:
    - Single vertex: No rotation matrices generated
    - Three vertices: Returns predefined rotation matrices for standard orthogonal views
    - Multiple vertices (≥8): Computes rotation matrices from origin to each subsequent vertex
    Args:
        vertices (np.ndarray): Array of vertex coordinates with shape (n, 3) where n is the
                              number of vertices. Each row represents a 3D point [x, y, z].
                              Must have 1, 3, or ≥8 vertices.
    Returns:
        list: List of rotation matrices as numpy arrays. Each matrix is 3x3 and represents
              a rotation transformation. Returns empty list for single vertex case.
    Raises:
        AssertionError: If the number of vertices is not 1, 3, or ≥8.
    Note:
        For the 3-vertex case, the function returns two predefined rotation matrices
        corresponding to standard orthogonal viewing directions. For multiple vertices
        (≥8), it uses the first vertex as origin and calculates rotation matrices
        to align with each subsequent vertex position.
    """



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
        # assert vertices.shape[0] >= 8, "Please make sure number of views is greater than 8 or equal to 1 or 3!"
    
        origin = np.array(vertices[0])

        
        for i in range(1, len(vertices)):
            rot_matrices.append(_calculate_rotation_matrix(origin, np.array(vertices[i])))
        
    return rot_matrices

def _find_largest_lesion_slice(mask: torch.Tensor, axis: int) -> int:
    """
    Find the slice index with the largest lesion area along a specified axis.
    This function analyzes a 3D binary mask tensor to identify which slice
    contains the maximum lesion area (defined as the number of non-zero voxels)
    when viewed along the specified axis.
    Args:
        mask (torch.Tensor): A 3D binary tensor representing the lesion mask.
                           Non-zero values indicate lesion presence.
        axis (int): The axis along which to analyze slices. Must be 0, 1, or 2
                   corresponding to the three spatial dimensions.
    Returns:
        int: The index of the slice with the largest lesion area along the
             specified axis.
    Raises:
        AssertionError: If mask is not a 3D tensor or if axis is not 0, 1, or 2.
    Example:
        >>> mask = torch.zeros(10, 20, 30)
        >>> mask[5, 10:15, 15:20] = 1  # Create lesion in slice 5 along axis 0
        >>> _find_largest_lesion_slice(mask, axis=0)
        5
    """


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
    Crop a 2D array to a square region based on the bounding box of a binary mask.
    This function finds the bounding box of the region of interest (ROI) defined by the mask,
    then creates a square crop centered on this bounding box. The square size is determined
    by the larger dimension of the bounding box, expanded by 10% for padding.
    Parameters
    ----------
    array : numpy.ndarray
        2D input array to be cropped.
    mask : numpy.ndarray
        2D binary mask array of the same shape as `array`. Values > 0 are considered
        as the region of interest.
    Returns
    -------
    numpy.ndarray
        Cropped square region from the input array. The dimensions of the returned array
        will be square, with size determined by the larger dimension of the ROI bounding box
        multiplied by 1.1 (rounded down to nearest integer).
    Notes
    -----
    - The function ensures the cropped region stays within the bounds of the original array.
    - If the desired square size exceeds the array boundaries, the function adjusts the
      crop region to maintain the largest possible square while staying within bounds.
    - The mask is automatically converted to binary (values > 0 become True).
    Examples
    --------
    >>> import numpy as np
    >>> array = np.random.rand(100, 100)
    >>> mask = np.zeros((100, 100))
    >>> mask[30:50, 40:60] = 1  # Create ROI
    >>> cropped = _crop_to_square(array, mask)
    >>> print(cropped.shape)  # Should be approximately (22, 22) due to 1.1x expansion
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
    Pad a 2D array to make it square by adding padding to the shorter dimension.
    This function takes a 2D numpy array and pads it with a specified value to create
    a square array. Padding is added to the bottom and right edges as needed.
    Args:
        array (numpy.ndarray): A 2D numpy array to be padded.
        padding_value (int or float, optional): The value to use for padding. 
            Defaults to 0.
    Returns:
        numpy.ndarray: A square 2D array with dimensions equal to the maximum 
            of the original array's height and width.
    Example:
        >>> import numpy as np
        >>> arr = np.array([[1, 2], [3, 4], [5, 6]])
        >>> padded = _pad_to_square(arr)
        >>> print(padded)
        [[1 2 0]
         [3 4 0]
         [5 6 0]]
    Note:
        This is a private function intended for internal use within the module.
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
    """
    Create or load a sphere mesh with optimally distributed points using Coulomb repulsion.
    This function generates a spherical mesh with a specified number of views by placing points
    on a unit sphere and optimizing their distribution using Coulomb repulsion forces. Three
    points are fixed along the coordinate axes, while the remaining points are optimized to
    minimize energy and achieve uniform distribution.
    Args:
        n_views (int): Total number of viewpoints/vertices to place on the sphere. Must be >= 3.
        output_dir (str): Directory path where the sphere data will be saved/loaded from.
        save_sphere (bool, optional): Whether to export the generated sphere as a PLY file.
            Defaults to True.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Array of shape (N, 3) containing the 3D coordinates
              of the sphere vertices.
            - faces (np.ndarray): Array of shape (M, 3) containing the triangular faces
              of the sphere mesh, where each row contains indices into the vertices array.
    Notes:
        - The function caches results as PyTorch tensors in the format "sphere_{n_views}_views.pt"
        - Three points are fixed at [1,0,0], [0,1,0], and [0,0,1] for consistency
        - Uses Delaunay triangulation to create the mesh surface from optimized points
        - Coulomb repulsion optimization runs for 10,000 steps with learning rate 0.01
        - If save_sphere is True, exports the mesh as "sphere_{n_views}_views.ply"
    Raises:
        ValueError: If n_views < 3 (implicitly, as fixed points require minimum 3 views)
    """
    
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
    """
    Extract omnidirectional 2D slices from a 3D medical volume and corresponding mask.
    This function processes a 3D medical volume and its associated segmentation mask to generate
    multiple 2D slice views from different orientations. The slices are extracted by rotating
    the volume around uniformly distributed viewpoints on a sphere and finding the slice with
    the largest lesion area for each rotation.
    Args:
        volume_path (str): Path to the 3D medical volume file (e.g., NIfTI format).
        mask_path (str): Path to the 3D segmentation mask file corresponding to the volume.
        output_dir (str): Directory path where the extracted slice images will be saved.
        n_views (int): Number of viewing angles for omnidirectional slice extraction. 
                      Must be at least 4.
    Returns:
        None: The function saves extracted slices as PNG images to the specified output directory.
    Raises:
        AssertionError: If any of the following conditions are not met:
            - volume_path is None or the file doesn't exist
            - mask_path is None or the file doesn't exist  
            - output_dir is None
            - n_views is None or less than 4
            - CUDA is not available on the system
    Notes:
        - Requires CUDA-capable GPU for 3D tensor rotations
        - The function automatically resamples the mask to match the volume resolution
        - Volumes are normalized to canonical orientation and isotropic spacing (1.0mm³)
        - Output images are cropped to square format and normalized to 0-255 range
        - The first slice extracted is from the original volume (z-axis), followed by rotated views
    Example:
        >>> extract_slices(
        ...     volume_path="/path/to/volume.nii.gz",
        ...     mask_path="/path/to/mask.nii.gz", 
        ...     output_dir="/path/to/output",
        ...     n_views=12
        ... )
    """    

    assert volume_path is not None, "Please provide a valid path to the 3D volume."
    assert os.path.exists(volume_path), f"The specified volume path does not exist: {volume_path}"
    assert mask_path is not None, "Please provide a valid path to the 3D mask."
    assert os.path.exists(mask_path), f"The specified mask path does not exist: {mask_path}"
    assert output_dir is not None, "Please provide a valid output directory."
    assert n_views is not None, "Please provide the number of views for OmniSlicer."
    assert n_views >= 4, "Number of views must be at least 4."
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