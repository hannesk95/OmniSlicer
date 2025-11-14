# Omnidirectional Volume Slicer - OmniSlicer 

Implementation of the omnidirectional volume slicer package in Python and PyTorch for 3D medical image analysis, from our publication ["TomoGraphView: 3D Medical Image Classification with Omnidirectional Slice Representations and Graph Neural Networks"](https://www.arxiv.org).

![Comparison of OmniSlicer against traditional methods](https://raw.githubusercontent.com/hannesk95/OmniSlicer/main/assets/volume_slicing.png)

## Example Usage

```python
from OmniSlicer import OmniSlicer

volume_path = "path_to_volume.nii.gz"
mask_path = "path_to_mask.nii.gz"
output_dir = "output_dir"
n_views = N

OmniSlicer.extract_slices(volume_path=volume_path,
                          mask_path=mask_path,
                          output_dir=output_dir,
                          n_views=n_views)
```

## Tested Dependencies

The functionality of **OmniSlicer** has been successfully validated using the following dependency versions. These represent the environment in which the package has been developed and tested:

| Dependency     | Version Tested |
|----------------|----------------|
| `python`       | 3.11.14        |
| `torch`        | 2.6.0+cu124    |
| `torchvision`  | 0.21.0+cu124   |
| `trimesh`      | 4.6.8          |
| `numpy`        | 2.2.6          |
| `pyvista`      | 0.45.0         |
| `torchio`      | 0.20.7         |
| `tqdm`         | 4.67.1         |

These versions are defined in the project’s installation requirements and are automatically resolved when installing OmniSlicer via `pip`. While other combinations may work, the dependency set above is the configuration against which all core features have been verified. **Please make sure that you install torch with CUDA**.


## Citation

Stay tuned, coming soon!

```bibtex
@misc{kiechle2025tomographview3dmedicalimage,
      title={TomoGraphView: 3D Medical Image Classification with Omnidirectional Slice Representations and Graph Neural Networks}, 
      author={Johannes Kiechle and Stefan M. Fischer and Daniel M. Lang and Cosmin I. Bercea and Matthew J. Nyflot and Lina Felsner and Julia A. Schnabel and Jan C. Peeken},
      year={2025},
      eprint={2511.09605},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2511.09605}, 
}
```