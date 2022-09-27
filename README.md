# Help a hematologist challenge

We placed third at the [Help a hematologist out challenge](https://helmholtz-data-challenges.de/web/challenges/challenge-page/93/overview). Here is the solution.

## Training Cycle-GAN for domain adaptation

We used the Cycle-GAN model to train a generator for generating Mat_19/Ace_20 images from WBC1 images and vice versa.

```
python train_mean_std.py --name mat_ace_wbc_mean_std --model cycle_gan \
--pool_size 50 --no_dropout --batch_size 16 --netG resnet_9blocks_noTanh
```

It expects the following in the root directory of the repo:

- `metadata.csv` file
- `Datasets/Acevedo_20`, `Datasets/Matek_19`, `Datasets/WBC1` datasets
- `Datasets/Mean_image.pickle` and `Datasets/Std_image.pickle` which are the channel-wise mean and standard deviation of the Mat_19 and Ace_20 images

It saves the log and model files under `checkpoints/mat_ace_wbc_mean_std` directory.

It also saves example generated images, Mat_19/Ace_20 <----> WBC1 after each epoch under `figures_mean_std` 

## Using trained Cycle-GAN to generate WBC1-like images for the Mat_19/Ace_20 images

```
python test_mean_std.py --name mat_ace_wbc_mean_std --model cycle_gan \
--no_dropout --epoch 25 --results_dir Datasets/MAT_ACE_AS_WBC_MEAN_STD \
--netG resnet_9blocks_noTanh
```

It uses the saved model at epoch 25 (you can change it to other epochs) to generate WBC1-like images
for Mat_19/Ace_20 dataset and saves them at `Datasets/MAT_ACE_AS_WBC_MEAN_STD`

The generated images (`Datasets/MAT_ACE_AS_WBC_MEAN_STD`) are then used to train a `resnet18` 
classifier model. The trained model is used for making predictions on the dev phase (WBC1) and 
test phase (WBC2) datasets, evaluated on the challenge website.

![alt text](https://github.com/Bashirkazimi/Help-a-hematologist-out-Challenge/blob/main/examples/0.png)

![alt text](https://github.com/Bashirkazimi/Help-a-hematologist-out-Challenge/blob/main/examples/10.png)

![alt text](https://github.com/Bashirkazimi/Help-a-hematologist-out-Challenge/blob/main/examples/11.png)

![alt text](https://github.com/Bashirkazimi/Help-a-hematologist-out-Challenge/blob/main/examples/12.png)

![alt text](https://github.com/Bashirkazimi/Help-a-hematologist-out-Challenge/blob/main/examples/13.png)

![alt text](https://github.com/Bashirkazimi/Help-a-hematologist-out-Challenge/blob/main/examples/1.png)

![alt text](https://github.com/Bashirkazimi/Help-a-hematologist-out-Challenge/blob/main/examples/2.png)


## Data Preprocessing Details

All the images are RGB. They were first center-cropped to  25 x 25 micrometers to keep the area of the background of the cell the same in all images.
Then they were resized to 225 x 225 pixels.
A mean and standard deviation image was calculated from the resized images for all the 3 channels.

<img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " /> 
 
## Training and making inference using resnet18

To train a classifier resnet18, run the following:

```
python train_resnet.py &> train.log
```

It will train the resnet18 model on the fake WBC dataset generated from Mat_19/Ace_20 dataset by the Cycle-GAN

model files and results will be saved under `models/resnet_train` and `results/resnet_train` folder.

The csv file to be submitted in the dev phase of the data challenge is at `results/resnet_train/submission.csv`

Then, the trained model can be used to make prediction on WBC2 (test phase) datasets that is
expected to be downloaded and saved at `Datasets/WBC2/DATA-TEST`

```
python test_resnet.py &> test.log
```

The results will be saved under `results/resnet_test/submission.csv` file which can be uploaded to test phase in the
data challenge

## The Team

BLAMAD: our team name is basically the first letter of the first names of all team members.

Team Members: Bashir Kazimi, Lea Gabele, Ankita Negi, Martin Brenzke, Arnab Majumdar, and Dawit Hailu

# Instructions using the code

The code is copied from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and adapted.

For detailed instruction on on its use, please go to their repo linked above.

Thanks to the original authors of Cycle-GAN.



## Citation
Citations for the original Cycle-GAN publications.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Acknowledgments
Code copied from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and adapted.
