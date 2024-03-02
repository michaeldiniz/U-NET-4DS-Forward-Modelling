# U-Net for 4D seismic Forward Modeling

In this repository, we address the challenges associated with conventional 4D seismic forward modeling, specifically its time-intensive nature and significant computational resource consumption. Using the scripts available in this repository, it is possible to develop a proxy model for 4D seismic forward modeling using the U-Net encoder-decoder (a class of machine learning architecture). The main objective of this proxy model is to enhance the efficiency of reservoir model calibration through the reservoir data assimilation method.     


## 1. Install the dependencies
The model is implemented using PyTorch. The full list of used libraries can be found in requirements.txt.
```
pip install -r requirements.txt
```

## 2. Data 

To demonstrate the performance of our proposal, we applied our methodology to the open-access UNISIM IV Benchmark. The raw data of UNISIM IV can be found at ... . The data converted to tensors compatible with PyTorch is located in the folder ./dataset/tensors/Lx_features, where x ∈ {1,2,3}.

For each map, the input data is represented in the file "prepro_X.pt". This file contains a tensor with dimensions (100, 8, 46, 46).

The first dimension represents each simulation model contained in each map.
The second dimension represents each of the physical properties used as input to the model. The order of these properties can be found in the file "ordem_modelos.pkl" in this same folder.

The last two dimensions represent the height and width of each map.

## 3. Usage
Follow the instructions to run the code: 

```
python main.py --gpu gpu_num --workers workers_num --cp ckpt_file --l map_id --r resolution
 ```
Where

• gpu_num: An integer representing the GPU that will process the training.
• workers_num: An integer representing the number of workers to be used.
• ckpt_file: A string with the path to a pre-trained model for testing.
• map_id: UNISIM IV map to be addressed. Choices=['L1', 'L2', 'L3']
• resolution: Resolution of the output map. Choices=['46x46', '94x78']


## License
This code is distributed under MIT License. Please see the file LICENSE for more details..

