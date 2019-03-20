This project is NOT for commercial use.
You are not allowed to share this project.

Step 0:
Intall Matlab CUDA CUDNN in Win10 System.

Step 1:
Put cell images in data/cells.
Crop those images with size of 600*600 to 450*450.

Step 2:
Install vlfeat and matconvnet to your path of matlab toolbox.

Step 3:
Download pre-trained models from 
http://www.vlfeat.org/matconvnet/pretrained/imagenet-resnet-50-dag.mat
http://www.vlfeat.org/matconvnet/pretrained/imagenet-resnet-101-dag.mat
http://www.vlfeat.org/matconvnet/pretrained/imagenet-resnet-152-dag.mat
and put them in images in data/models.

Step 4:
Create a new file with name isbi_test_guess.csv
and paste the follow content in it. It contains 'label' and 2586 binary numbers.
label
1
0
0
1
1
....

Step 5:
Run Phrase_III_of_ISBI_NMC.m

Step 6:
You can also download our trained basic models from
https://drive.google.com/drive/folders/1DPc85NsP4PiQz39x3up55Qx7KoP0iw9g?usp=sharing
https://drive.google.com/drive/folders/177dyrKsWdkP5XoNhsaD6Cp1wyIdx3aDt?usp=sharing


