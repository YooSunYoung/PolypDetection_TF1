# PolypDetection_TF1
### Overview
This project consists of some tools for polyp detection.

```
VitisAIForPolyp
    ├───EnvironmentSettings
    ├───Models
    ├───Training
    └───Utils (data pre-processing,
               evaluation,
               train results post-processing)
```
After running data pre-processing and training, more directories will be added like below.
```
    ├───data
    │     ├─── PolypImages
    │     ├─── PolypImages_train
    │     ├─── PolypImages_aug
    │     ├─── PolypImages_valid
    │     ├─── train_image.npy
    │     ├─── train_label.npy
    │     ├─── valid_image.npy
    │     ├─── valid_label.npy
    │     ├─── test_image.npy
    │     ├─── test_label.npy
    └─── results
          ├─── checkpoints
          └─── logs
```
### Environment Setting
`EnvironmentSettings/` consists of the `.yml` files for conda environments.
You can create the virtual environment via command below regarding your hardware(CPU/GPU).
There are also pip recipes.
#### Conda (Recommended)

```bash
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate polyp-tf1-gpu
```

<!-- 
```
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-tf2-gpu
```
-->

#### Pip

```bash
pip install -r pip_requirements_cpu.txt
```

### Data
This project is for polyp detection so we are using the images taken by the endoscopy.
The size and the number of the channels(color) of images are stated in the `Models/config.py` as a dictionary.
The dataset should be a set of image(`.bmp` or `.jpg`) and label(`.xml`) pairs.
In the label(`.xml`) file, it should include the `filename`, image `size` (which consists of `width`, `height`, `channel`)
and the `object` which is the information of polyp in this case.

`object` should includes the `name` of the object such as 'Polyp' in this case.
It also has to include the location of the bounding box around the polyp as `xmin`, `ymin`, `xmax` and `ymax`.

The image files and the xml files should have same name before their file execution tag.
It would be easier to put all the data in one directory. For example,
```
├───data
      ├───PolypImages
            ├─── 001.bmp
            ├─── 001.xml
            ├─── 002.bmp
            ├─── 002.xml
                    .
                    .
                    .
```

### Data Pre-processing
#### 1. Split data into Train/Valid/Test set
Here is how to use `Utils/split_dataset.py` to split dataset randomly into `training`, `validation` and `test` sets.
Before you split the dataset all the dataset including images and the label files(usually .xml files) should be in one directory.
It doesn't remove the files in the `data_dir` but copy them in to other directories. 

- --data_dir : path to the directory which has all the dataset
- --output_dir_prefix : output files will be copied into `${output_dir_prefix}_train`, `${output_dir_prefix}_valid` and `${output_dir_prefix}_test`directory.                    
- --fraction : proportional expression for the number of train, valid and test images.
```bash
python Utils/split_dataset.py \
                        --data_dir ./data/PolypImages \
                        --output_dir_prefix ./data/PolypImages \
                        --fraction 5:3:2
```
> For example, if you execute the command line above, 
> and you have 10 images and labels in the `./data/PolypImages` directory,
> you will have 5 files in the `./data/PolypImages_train` directory,
> 3 files in the `./data/PolypImages_valid` directory and
> 2 files in the `./data/PolypImages_test` directory as a result.

#### 2. Image Augmentation
As the lack of polyp images we did image augmentation for the training dataset.
All the images should have the same size.

first, you need to install `imgaug` module with conda.

```bash
conda config --add channels conda-forge
conda install imgaug
```

#### 3. Save Images and Labels as Numpy Binary Files
When you finished splitting dataset into `train`, `valid`, and `test`, 
you should save the images and labels in binary files 
so that we can call them fast for training or evaluation.
`dataset.py` load images and labels from each directory of dataset and save them as `.npy` binary files.
* You have to change the settings in the script in order to use different directory path or name of binary files.
```bash
python dataset.py
```  

### Models
<!-- description for the models needed-->
- `Models/config.py` contains the information as a dictionary variable `configuration`.
  If you change the content in the `configuration` and it would affect all other processes.
  You may also put the information as arguments to the specific methods.
- `Models/dataset.py` contains the `Dataset` class and its static methods.
- `Models/models.py` contains the `PolypDetectionModel` class.
  For now, the only model is modified from squeezenet and loss function is from Yolo V1.

### Training

`Training/train.py` perform training the `PolypDetectionModel` model with .npy binary files we made in data preprocessing.
With `validate` option `True`, it saves the model where validation loss is the lowest.
You can set `save_point` and `validation_point` where you save model every `(save/validatoin)_point` epochs.
Other options are self-describing.
The command below shows all options you can set for training.
```bash
python Training/train.py \
                        --train_image ./data/train_image.npy \
                        --train_label ./data/train_label.npy \
                        --validate True \
                        --valid_image ./data/valid_image.npy \
                        --valid_label ./data/valid_label.npy \
                        --classes ./data/polyp.names \
                        --checkpoint_dir_path ./results/checkpoints \
                        --log_dir_path ./results/log \
                        --epochs 200 \
                        --save_point 50 \
                        --validation_point 10 \
                        --batch_size 32 \
                        --val_batch_size 32 \
                        --learning_rate 1e-3
```

Default settings are in the `Training/training_recipy.py`.
You can change the default setting by editing `Training/training_recipy.py`.
Then you can simply run the training by `python Training/train.py`.

### Evaluation
Evaluation is done with `train_image.npy` and `train_label.npy`.

### Postprocessing (Weight and Model files) 

