# Omnidirectional Volume Slicer - OmniSlicer 

Implementation of the omnidirectional volume slicer package in Python and PyTorch for 3D medical image analysis, from our Medical Image Analysis publication ["TomoGraphView: 3D Medical Image Classification with Omnidirectional Slice Representations and Graph Neural Networks"](https://www.sciencedirect.com/science/article/pii/S1361841526002434).

![Comparison of OmniSlicer against traditional methods](https://raw.githubusercontent.com/hannesk95/OmniSlicer/main/assets/volume_slicing.png)

## Example Usage

```python
from OmniSlicer import OmniSlicer

volume_path = "path/to/volume.nii.gz" 
mask_path = "path/to/mask.nii.gz" 
output_dir = "path/to/output_dir"
n_views = N

OmniSlicer.encode_volumes(
  volume_list=[volume_path],   # you can pass a list of several image paths here
  mask_list=[mask_path],       # you can pass a list of several segmentation mask paths here
  output_dir=output_dir,
  n_views=n_views,
  model_name="facebook/dinov3-vits16-pretrain-lvd1689m"  # currently, choose from any official facebook/dinov2 or facebook/dinov3 model available on huggingface
)
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
| `transformers` | 5.21.0         |


These versions are defined in the project’s installation requirements and are automatically resolved when installing OmniSlicer via `pip`. While other combinations may work, the dependency set above is the configuration against which all core features have been verified. **Please make sure that you install torch with CUDA**.


## Citation

```bibtex
@article{kiechle2026tomographview,
  title={TomoGraphView: 3D medical image classification with omnidirectional slice representations and graph neural networks},
  author={Kiechle, Johannes and Fischer, Stefan M and Lang, Daniel M and Bercea, Cosmin I and Nyflot, Matthew J and Felsner, Lina and Schnabel, Julia A and Peeken, Jan C},
  journal={Medical Image Analysis},
  pages={104174},
  year={2026},
  publisher={Elsevier}
}
```
