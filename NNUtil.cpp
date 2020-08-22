#include <stdio.h>
#include <tchar.h>
#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <tuple>
#include <chrono>
#include <io.h>

#include "VGGNet.h"
#include "ResNet.h"
#include "ImageProcess.h"
#include "CmdLineParser.h"

extern std::map<VGG_CONFIG, std::string> _VGG_CONFIG_NAMES;

void FreeBlob(void* p)
{
	//printf("Free the blob which is loaded by a Tensor.\n");
	free(p);
}

#define NUM_OF_CLASSES		1000

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

void freeargv(int argc, char** argv)
{
	if (argv == NULL)
		return;

	for (int i = 0; i < argc; i++)
	{
		if (argv[i] == NULL)
			continue;

		delete[] argv[i];
	}
	delete argv;
}

void Test()
{
	torch::Tensor a = torch::randn({ 1, 1, 4, 4 });
	std::cout << a << '\n';

	auto conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 2, 3).padding(1));

	auto b = conv1(a);

	std::cout << b << '\n';

	std::cout << a << '\n';

#if 0
	ImageProcess imageprocessor;
	if (SUCCEEDED(imageprocessor.Init(10, 10)))
	{
		torch::Tensor tensor;
		{
			float means[3] = { 0.f, 0.f, 0.f };
			float stds[3] = { 1.f, 1.f, 1.f };
			imageprocessor.SetRGBMeansAndStds(means, stds);
		}
		if (SUCCEEDED(imageprocessor.ToTensor(_T("I:\\RGB.png"), tensor)))
		{
			printf("before transforming....\n");
			std::cout << tensor << '\n';
		}

		{
			float means[3] = { 0.5f, 0.5f, 0.5f };
			float stds[3] = { 0.5f, 0.5f, 0.5f };
			imageprocessor.SetRGBMeansAndStds(means, stds);
		}
		if (SUCCEEDED(imageprocessor.ToTensor(_T("I:\\RGB.png"), tensor)))
		{
			printf("after transforming....\n");
			std::cout << tensor << '\n';
		}
	}

	imageprocessor.Uninit();
#endif
}

int _tmain(int argc, const TCHAR* argv[])
{
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;

	if (argc <= 1)
	{
		PrintHelp();
		return 0;
	}

	const char** u8argv = NULL;
#ifdef _UNICODE
	u8argv = new const char*[argc];
	for (int i = 0; i < argc; i++)
	{
		if (argv[i] == NULL)
			u8argv[i] = NULL;
		else
		{
			size_t ccLen = _tcslen(argv[i]);
			u8argv[i] = new const char[ccLen * 4 + 1];
			WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, (LPSTR)u8argv[i], (int)ccLen * 4 + 1, NULL, NULL);
		}
	}
#else
	u8argv = (const char**)argv;
#endif

	if (CmdLineParser::ProcessCommandLineArgs(argc, u8argv) == false)
	{
		freeargv(argc, (char**)u8argv);
		PrintHelp();
		return -1;
	}

	//CmdLineParser::GetCmdLineParser().Print();

	if (FAILED(CoInitializeEx(NULL, COINIT_MULTITHREADED)))
	{
		freeargv(argc, (char**)u8argv);
		return -1;
	}

	BaseNNet* ptrNet = nullptr;
	CmdLineParser& ctxCmd = CmdLineParser::GetCmdLineParser();

	if (ctxCmd.nn_type >= NN_TYPE_VGGA && ctxCmd.nn_type <= NN_TYPE_VGGE)
	{
		ptrNet = new VGGNet();
	}
	else if (ctxCmd.nn_type >= NN_TYPE_RESNET18 && ctxCmd.nn_type <= NN_TYPE_RESNET152)
	{
		ptrNet = new ResNet();

	}
	else
	{
		freeargv(argc, (char**)u8argv);
		printf("Unsupported neutral network model is specified!\n");
		return -1;
	}

	switch (ctxCmd.cmd)
	{
	case NN_CMD_STATE:
	{
		std::cout << torch::show_config() << '\n';
		// load the pre-trained net and show its information
		if (_access(ctxCmd.train_net_state_path.c_str(), 0) == 0 && ptrNet->loadnet(ctxCmd.train_net_state_path.c_str()) == 0)
			ptrNet->Print();
		else
		{
			ptrNet->SetOptions({
					{"NN::nn_type", std::to_string((int)ctxCmd.nn_type)},
					{"NN::enable_batch_norm", std::to_string(ctxCmd.enable_batch_norm?1:0)},
					{"NN::final_out_classes", std::to_string(ctxCmd.num_classes) },
					{"NN::use_32x32_input", std::to_string(ctxCmd.use_32x32_input ? 1 : 0) },
				});

			if (ptrNet->loadnet(nullptr) == 0)
				ptrNet->Print();
		}
	}
	break;
	case NN_CMD_TRAIN:
	{
		bool bLoadSucc = false;
		if (ctxCmd.image_set_root_path.size() == 0 || ctxCmd.train_net_state_path.size() == 0)
		{
			PrintHelp();
			goto done;
		}

		// delete the previous pre-train net state
		if (ctxCmd.clean_pretrain_net)
		{
			if (DeleteFileA(ctxCmd.train_net_state_path.c_str()) == FALSE)
			{
				printf("Failed to delete the file '%s' {err: %lu}.\n", ctxCmd.train_net_state_path.c_str(), GetLastError());
			}
		}

		// Try to load the net from the pre-trained file if it exist
		if (_access(ctxCmd.train_net_state_path.c_str(), 0) == 0)
		{
			if (ptrNet->loadnet(ctxCmd.train_net_state_path.c_str()) != 0)
				printf("Failed to load the VGG network from %s, retraining the VGG net.\n", ctxCmd.train_net_state_path.c_str());
			else
				bLoadSucc = true;
		}

		// Failed to load net from the previous trained net, retrain the net
		if (bLoadSucc == false)
		{
			ptrNet->SetOptions({
					{"NN::nn_type", std::to_string((int)ctxCmd.nn_type)},
					{"NN::enable_batch_norm", std::to_string(ctxCmd.enable_batch_norm ? 1 : 0)},
					{"NN::final_out_classes", std::to_string(ctxCmd.num_classes) },
					{"NN::use_32x32_input", std::to_string(ctxCmd.use_32x32_input ? 1 : 0) },
				});

			if (ptrNet->loadnet(NULL) != 0)
			{
				printf("Failed to load the neutral network.\n");
				goto done;
			}
		}

		tm_end = std::chrono::system_clock::now();

		{
			long long load_duration =
				std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
			printf("It took %lldh:%02dm:%02d.%03ds msec to construct the '%s' network.\n",
				load_duration / 1000 / 3600,
				(int)(load_duration / 1000 / 60 % 60),
				(int)(load_duration / 1000 % 60),
				(int)(load_duration % 1000),
				ptrNet->nnname().c_str());
			tm_start = std::chrono::system_clock::now();
		}

		ptrNet->train(ctxCmd.image_set_root_path.c_str(),
			ctxCmd.train_net_state_path.c_str(),
			ctxCmd.batchsize,
			ctxCmd.epochnum,
			ctxCmd.learningrate,
			ctxCmd.showloss_per_num_of_batches);
	}
	break;
	case NN_CMD_VERIFY:
	{
		if (ctxCmd.image_set_root_path.size() == 0 || ctxCmd.train_net_state_path.size() == 0)
		{
			PrintHelp();
			goto done;
		}

		ptrNet->verify(ctxCmd.image_set_root_path.c_str(), ctxCmd.train_net_state_path.c_str());
	}
	break;
	case NN_CMD_CLASSIFY:
	{
		if (ctxCmd.train_net_state_path.size() == 0 || ctxCmd.image_path.size() == 0)
		{
			PrintHelp();
			goto done;
		}

		if (ptrNet->loadnet(ctxCmd.train_net_state_path.c_str()) != 0)
		{
			printf("Failed to load the VGG network from %s.\n", ctxCmd.train_net_state_path.c_str());
			goto done;
		}

		ptrNet->classify(ctxCmd.image_path.c_str());
	}
	break;
	case NN_CMD_TEST:
		Test();
		break;
	default:
	{
		PrintHelp();
		goto done;
	}
	}

done:
	if (ptrNet != nullptr)
		delete ptrNet;

	CoUninitialize();

	freeargv(argc, (char**)u8argv);

	return 0;
}

