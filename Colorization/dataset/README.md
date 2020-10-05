# BigEarthNet Dataset
BigEarthNet is a new large-scale Sentinel-2 benchmark archive, consisting of 590,236 Sentinel-2 image patches. Can be downloaded [here](http://bigearth.net/), together with the files 
``Image patches with seasonal snow`` and ``Image patches with cloud & shadow``.
## Create csv file of the dataset
Once downloaded the files, the first step consist in the creation of a csv file containing the paths for all the images. To do that, run the following code:
 ```
 python big_earth_utils.py --big_e_path [path to BigEarthNet dataset] --num_samples [-1 for all the images] --csv_filename [name of the file] --mode [csv_creation]
 ```
Since this operation may take a while, as alternative I loaded on drive [here](https://drive.google.com/drive/folders/19MsGGVveafgS5IG1A61brAoxsjCCBg3k?usp=sharing) my csv file,
to change the paths run the code below.
 ```
 python big_earth_utils.py --big_e_path [path to BigEarthNet dataset] --csv_filename [name of the file] --mode [replace_path_csv] --new_path_csv [your path]
 ```
 
The BigEarthNet dataset contains also images covered by snow or cloud, to remove this latters run:
 ```
 python big_earth_utils.py --csv_filename [name of the file] --mode [delete_patches_v2] 
 ```
## Quantiles
To conclude the pre-processing stage, you need to calculate the min and max quantile for each different band, to put a threshold on eventual too high or too low pixel values. 
I already loaded the file ``quantiles.json``, to change the number of samples used or recalculate the values run the ``big_earth_utils.py`` file with ``--mode quantiles``.
 
## BigEarth dataset vs BigEarth dataset torch version
The .csv file and the quantiles.json created above are exploited in the two files:  ``dataset_big_earth.py`` and ``dataset_big_earth_torch.py``. The torch version was created to speed up
the training process, as it allows you to load only one tensor at a time instead of 12 .tif bands.
The creation of the tensor takes long time (so eventually run it in a tmux session) and can be done with the following command:
 ```
 python dataset_big_earth.py --csv_filename [name of the file] --n_samples [Number of samples to use] --create_torch_dataset [1] 
 ```



## Credits
*G. Sumbul, M. Charfuelan, B. Demir, V. Markl, "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding", IEEE International Geoscience and Remote Sensing Symposium, pp. 5901-5904, Yokohama, Japan, 2019.*
