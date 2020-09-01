# VGG net

## How to set up the libtorch environment based on Visual Studio

https://blog.csdn.net/defi_wang/article/details/107450428

## Code Introduce

https://blog.csdn.net/defi_wang/article/details/107889818
https://blog.csdn.net/defi_wang/article/details/108032208

## Convert Image to Tensor

https://blog.csdn.net/defi_wang/article/details/107936757

## How to run it?
*VGGNet [options] command [arg...]*

### *Commands*

|command|description|
|--------------|-----------------|
|state | Show the net state |
|train | train the network |
|verify | verify the pre-trained network with the test-set|
|classify | classify an input picture with the pre-trained network |
|help | show the help information |

### *options*
|option|description|
|------|--------|
|-v |Verbose mode to output more logs|
|-y | Proceed the operation without any prompt |
|-t nntype | Specify the neutral network mode, please see the "neutral network" table|
|-r image_set_root_path| specify the image set root path |
|-i imgset_type | specify the image-set type, at default, it is folder image set, please see the "image set type" table |


### *neutral network*
|type|description|type|description|
|------|--------|------|--------|
|**LENET**|LeNet5|**RESNET18**|ResNet-18|
|**VGGA**|VGG-A|**RESNET34**|ResNet-34|
|**VGGA_LRN**|VGG-A LRN|**RESNET50**|ResNet-50|
|**VGGB**|VGG-B|**RESNET101**|ResNet-101|
|**VGGC**|VGG-C|**RESNET152**|ResNet-152|
|**VGGD**|VGG-D|
|**VGGE**|VGG-E|

### *image set type*
|type|description|
|------|--------------|
|folder|the train/test image set are organized with folder/files |
|MNIST|the MNIST hand-writing train and test set |
|CIFAR10|The CIFAR10 image set |
|CIFAR100| The CIFAR100 image set |

## *arguments for command*
### **state**
*VGGNet state [--bn/-batchnorm] [-n numclass] [-s/--smallsize] [train_output]*

If no arg is specified, it will print the VGG-D net at default.

examples:
```
VGGNet.exe state --bn --numclass 10 --smallsize
```
print the neutral network state with batchnorm layers, the output number of classes and use the 32x32 small input image instead the 224x224 image.
```
VGGNet.exe I:\catdog.pt
```
print the information of neutral network loading from I:\catdog.pt.

#### *args*
|name|shortname|arg|description|
|----|---------|---|-----------|
|**batchnorm**<br>**bn**|*n/a*|*n/a*|enable batchnorm after CNN |
|**numclass**|**n**|num of classes|The specified final number of classes, the default value is 1000|
|**smallsize**|**s**|*n/a*|Use 32x32 input instead of the original 224\*224|


### **train**
***NNUtil*** \[\-v ] \[\-y] \[\-r image_set_root_path] \[\-i imgset_type] ***train*** [train_output] [-b/--batchsize batchsize] [-e/--epochnum epochnum] [-l/--learningrate fixed_learningrate] [--bn/--batchnorm] [-n numclass] [-s/--smallsize] [--showloss once_num_batch] [--clean]

#### *args*
|name|shortname|arg|description|
|----|---------|---|-----------|
|**batchsize**|**b**|batchsize|the batch size of sending to network|
|**epochnum**|**e**|epochnum|the number of train epochs|
|**learningrate**|**l**|learning rate|the fixed learning rate<br>(\*)if it is not specified, default learning rate is used, dynamic learning rate is used|
|**lrdm**|*n/a*|learning rate decay mode|learning rate manager mode, see learning rate decay mode table|
|**lr_decay_steps**|*n/a*|*decay_steps*|learning rate decay steps|
|**lr_decay_rate**|*n/a*|decay_rate|learning rate decay rate |
|**lr_end**|*n/a*|end_learning_rate|end at the specified learning rate|
|**lr_cycle**|*n/a*||the learning rate decline, and rise again, true or false|
|**lr_staircase**|*n/a*||use the floor value for (global_step/decay_steps), true or false|
|**lr_alpha**|*n/a*||the parameter used in cosine decay |
|**lr_beta**|*n/a*||the parameter used in linear cosine decay|
|**lr_num_periods**|*n/a*||fade cosine periods|
|**lr_power**|*n/a*||the power used in polynomial decay|
|**lr_initial_variance**|*n/a*||the noise initial variance |
|**lr_variance_decay**|*n/a*||the decay noise variance|
|**batchnorm**<br>**bn**|*n/a*|*n/a*|enable batchnorm after CNN |
|**numclass**|**n**|num of classes|The specified final number of classes, the default value is 1000|
|**smallsize**|**s**|*n/a*|Use 32x32 input instead of the original 224\*224|
|**showloss**|*n/a*|once_num_batch|stat. the loss every num batch |
|**clean**|*n/a*|*n/a*|clean the previous pre-trained net state file |
|**optimizer**|**o**|optimizer | specify the optimizer, at default SGD will be used |
|**weight_decay**|**w**|weight decay for optimizer | the L2 Regularization |
|**momentum**|**m**|momentum|momentum|
|**dampening**|*n/a*|dampening|dampening|
|**nesterov**|*n/a*|nesterov|nesterov|


### *learning rate decay mode*
|learning rate decay mode | name|
|---------------------|------------|
|exponent|exponent decay, lr = initial_learning_rate*decay_rate^(global_step/decay_steps)|
|natural_exp|natural exp decay|
|polynomial|polynomial decay|
|inversetime|inverse time decay|
|cosine|cosine decay|
|lcosine|linear cosine decay|

Train a network with the specified train image set, and the image set folder structure is like as

```
{image_set_root_path} 
  |-training_set
  |   |-tag1
  |   |  |-- images......
  |   |-tag2
  |   |  |-- images......
  |   ......
  |_test_set
      |-tag1
      |  |-- images......
```
Examples
```
NNUtil -r I:\CatDog train I:\catdog.pt --bn -b 64 -l 0.0001 --showloss 10
```
Train the image set lies at I:\CatDog, and save the output to I:\catdog.pt, the batchnorm layers will be introduced, and the batch size is 64, the learning rate use the fixed 0.0001, and show the loss rate every 10 batches.
```
NNUtil -t RESNET18 -r I:\CIFAR\cifar-10-batches-bin -d CIFAR10 train I:\cifar_resnet18.pt -b 64 -l 0.0001 --showloss 10 --optim adam
```
Train the CIFAR10 image set, and save the train result to I:\cifar_resnet18.pt, batch size is 64, learning rate is 0.0001, and show the loss every 10 batches, and use adam optimizer.
### **verify**
***NNUtil*** \[\-v] \[\-y] \[\-r image_set_root_pah] \[\-i imgset_type] ***verify*** image_set_root_path pretrain_network*
Verify the test-set and show the pass-rate and other information

```
NNUtil -r I:\CatDog verify I:\catdog.pt
```

### **classify**
***NNUtil classify*** pretrain_network image_file_path*
With the specified pre-trained network, classify a image.

```
NNUtil classify I:\catdog.pt PIC_001.png
```