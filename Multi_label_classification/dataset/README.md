## Change Labels
Some of the original labels introduced in the original version of the BigEarthNet dataset are tough to predict. 
To tackle this problem, the paper "BigEarthNet Deep Learning Models with A New Class-Nomenclature for Remote Sensing Image Understanding" proposed a new nomenclature with 19 labels.
By running the code below you can change the csv file created in the colorization phase with this new set of labels.
 ```
 python change_labels.py --csv_filename [name of the file] --new_path_csv [name of the new csv file]
 ```

## Credits
```bibtex
@article{sumbul2020bigearthnet,
  title={BigEarthNet Deep Learning Models with A New Class-Nomenclature for Remote Sensing Image Understanding},
  author={Sumbul, Gencer and Kang, Jian and Kreuziger, Tristan and Marcelino, Filipe and Costa, Hugo and Benevides, Pedro and Caetano, Mario and Demir, Beg{\"u}m},
  journal={arXiv preprint arXiv:2001.06372},
  year={2020}
}
```
