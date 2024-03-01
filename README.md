# U-Net for 4D seismic Forward Modeling

In this repository, we address the challenges associated with conventional 4D seismic forward modeling, specifically its time-intensive nature and significant computational resource consumption. Using hte scripts available in this repository, it is possible to develop a proxy model for 4D seismic forward modeling using the U-Net encoder-decoder (a class of machine learning architecture). The main objective of this proxy model is to enhance the efficiency of reservoir model calibration through the reservoir data assimilation method.     


## 1. Install the dependencies
The model is implemented using PyTorch. The full list of used libraries can be found in requirements.txt.
```
pip install -r requirements.txt
```

## 2. Data
The dataset you are using for training or testing Y-GAN should have the following directory structure where the names of individual classes are represented with unique numbers or strings:

```



```

## 3. Usage
To train run the following:

```
python main.py --dataroot data --dataset Dataset_name --name experiment_name --isize image_size
```

## License
This code is distributed under MIT License. Please see the file LICENSE for more details..

