# molani
automated molecular animator 


## Requirements
* [vmd](https://www.ks.uiuc.edu/Research/vmd/) to set up scenes
* [tachyon 0.99b6](http://jedi.ks.uiuc.edu/~johns/tachyon/) ray tracer (bundled with vmd)
* [imagemagick](https://imagemagick.org/index.php) for annotation

## Setup

You can install the latest version directly from GitHub.

    pipx install -U git+https://github.com/theia-dev/molani.git#egg=molani]

## Usage

* `molani movie.json` or `mpirun -np 4 molani movie.json`

**`movie.json`**
```json
{
  "name": "cluster",
  "base_dir": "/media/scratch/cluster",
  "movie_name": "cluster_720",
  "height": 720,
  "rotations_per_minute": 0.0,
  "rotate_axis": "z",
  "final_scale": 1.0,
  "render_scale": 1.2,
  "periodic": "xyXY",
  "color_map": [["resname \"P.*\"", ["nipy_spectral", -0.1]], ["resname \"A.*\"", ["nipy_spectral", 0.1]]],
  "type": "cluster_set", 
  "depthcue": "off",
  "trajectory": "reduced",
  "meta_data": "cluster_ratio.json",
  "clean_up": false,
  "cpu_per_task": 3
}
```

***

The full source code can be accessed on [GitHub](https://github.com/theia-dev/molani).
