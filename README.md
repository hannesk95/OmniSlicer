# OmniSlicer 

Implementation of the omnidirectional volume slicer in PyTorch for 3D medical images. 

## How to use?

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

## Citations

Stay tuned, coming soon!

<!-- ```bibtex
@inproceedings{polat2022class,
  title={Class distance weighted cross-entropy loss for ulcerative colitis severity estimation},
  author={Polat, Gorkem and Ergenc, Ilkay and Kani, Haluk Tarik and Alahdab, Yesim Ozen and Atug, Ozlen and Temizel, Alptekin},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={157--171},
  year={2022},
  organization={Springer}
}
``` -->