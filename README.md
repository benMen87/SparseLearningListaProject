# SparseLearningListaProject

**Overview**
This reposotory contains a TF implementation of paper 
__Learned Convolutional Sparse Coding__  
https://arxiv.org/abs/1711.00328.
 
 *master branch* contains VANILA implemintation of the proposed model while in *master_test*  branch contain a more variants of the model.
 ![ACSC model](https://github.com/benMen87/SparseLearningListaProject/blob/master/acsc_model.png)

**Train**
---------
to run training code:  
> cd encoder\_decoder  
> python train.py --grayscale

see encoder_decoder/train.py for more arguments.


**Test**
--------
to run test code:  
> cd encoder\_decoder  
> python test.py --test_type denoise   
*or*  
> python test.py --test_type inpint

where the defult test is run on on the checkpint model in 'code/encoder_decoder/logdir/models'
