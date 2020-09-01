#include <stdio.h>

void PrintHelp()
{
	printf("Usage:\n\tVGGNet [options] command [args...]\n");

	printf("\t\tcommands:\n");
	printf("\t\t\tstate:\t\tPrint the VGG layout\n");
	printf("\t\t\ttrain:\t\tTrain the VGG16\n");
	printf("\t\t\tverify:\t\tVerify the train network with the test set\n");
	printf("\t\t\tclassify:\tClassify the input image\n");

	printf("\t\targs:\n");
	printf("\t\t\t--batchsize, -b\tThe batch size of training the network\n");
	printf("\t\t\t--epochnum\tSpecify how many train epochs the network will be trained for\n");
	printf("\t\t\t--lr, -l\tSpecify the learning rate\n");
	printf("\t\t\t--batchnorm,\n\t\t\t--bn\t\tEnable batchnorm or not\n");
	printf("\t\t\t--numclass\tSpecify the num of classes of output\n");
	printf("\t\t\t--smallsize, -s\tUse 32x32 input image or not\n");
	printf("\t\t\t--showloss, -s\tSpecify how many batches the loss rate is print once\n");
	printf("\t\t\t--clean\t\tclean the previous train result\n");

	printf("\t\texamples:\n");
	printf("\t\t\tVGGNet state\n");
	printf("\t\t\tVGGNet train I:\\CatDog I:\\catdog.pt --bn -b 64 --showloss 10 --lr 0.001\n");
	printf("\t\t\tVGGNet verify I:\\CatDog I:\\catdog.pt\n");
	printf("\t\t\tVGGNet classify I:\\catdog.pt I:\\test.png\n");
}

void PrintHelpMenu()
{
	printf("NNUtil -- Neutral network utility\n");
	printf("\nUsage:\n\tVGGNet [options] command [args...]\n\n");
	printf("Try: \n");
	printf("\tNNUtil help simple\t\tlist the general usage\n");
	printf("\tNNUtil help commands\t\tlist all commands]n");
	printf("\tNNUtil help command\t\thelp on a specific command\n");

	printf("\n");

	printf("\tNNUtil help networks\t\tlist supported neutral networks\n");
	printf("\tNNUtil help options\t\tgeneric options\n");
}