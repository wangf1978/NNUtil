#include "ResNet.h"

#include <torch/nn/module.h>
#include <iostream>
#include <tuple>
#include <chrono>
#include <io.h>
#include <tchar.h>
#include <random>
#include <algorithm>
#include "util.h"

extern void FreeBlob(void* p);

#define RESNET_INPUT_IMG_WIDTH					224
#define RESNET_INPUT_IMG_HEIGHT					224
#define RESNET_TRAIN_BATCH_SIZE					64

#define RESNET_DEFAULT_WEIGHT_DECAY				0.0001
#define RESNET_DEFAULT_MOMENTUM					0.9

std::map<RESNET_CONFIG, std::string> _RESNET_CONFIG_NAMES =
{
	{RESNET_18,				"RESNET18"},
	{RESNET_34,				"RESNET34"},
	{RESNET_50,				"RESNET50"},
	{RESNET_101,			"RESNET101"},
	{RESNET_152,			"RESNET152"}
};

ResNet::ResNet()
	: m_RESNET_config(RESNET_UNKNOWN)
	, m_num_classes(-1)
	, m_use_32x32_input(false) {
}

ResNet::~ResNet()
{
	m_imageprocessor.Uninit();
	Uninit();
}

int ResNet::_Init()
{
	int iRet = -1;

	if (m_bInit)
	{
		printf("The current neutral network is already initialized.\n");
		return 0;
	}

	if (m_RESNET_config == RESNET_UNKNOWN)
	{
		printf("Don't know the current net configuration.\n");
		return -1;
	}

	auto iter = _RESNET_CONFIG_NAMES.find(m_RESNET_config);
	if (iter != _RESNET_CONFIG_NAMES.end())
	{
		SetOptions({
			{"NN::final_out_classes", std::to_string(m_num_classes) },
			{"NN::use_32x32_input", std::to_string(m_use_32x32_input ? 1 : 0) },
			});
		iRet = Init(iter->second.c_str());
	}

	if (iRet >= 0)
		m_bInit = true;

	return iRet;
}

int ResNet::unloadnet()
{
	m_RESNET_config = RESNET_UNKNOWN;
	m_num_classes = -1;
	m_use_32x32_input = false;

	m_imageprocessor.Uninit();

	return Uninit();
}

int ResNet::train(const char* szImageSetRootPath,
	IMGSET_TYPE img_type,
	const char* szTrainSetStateFilePath,
	LearningRateMgr* pLRMgr,
	int batch_size,
	int num_epoch,
	unsigned int showloss_per_num_of_batches,
	double weight_decay,
	double momentum,
	OPTIM_TYPE optim_type)
{
	TCHAR szImageFile[MAX_PATH] = { 0 };
	// store the file name classname/picture_file_name
	std::vector<tstring> train_image_files;
	std::vector<tstring> train_image_labels;
	std::vector<tstring> train_image_coarse_labels;
	int num_of_images = 0;

	std::vector<size_t> train_image_shuffle_set;
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;

	TCHAR* tszImageSetRootPath = NULL;
	// Convert UTF-8 to Unicode
#ifdef _UNICODE
	wchar_t wszImageSetRootPath[MAX_PATH + 1] = { 0 };
	MultiByteToWideChar(CP_UTF8, 0, szImageSetRootPath, -1, wszImageSetRootPath, MAX_PATH + 1);
	tszImageSetRootPath = wszImageSetRootPath;
#else
	tszImageSetRootPath = szImageSetRootPath;
#endif

	batch_size = batch_size < 0 ? 1 : batch_size;

	HRESULT hr = m_imageprocessor.loadImageSet(img_type, tszImageSetRootPath, train_image_labels, train_image_coarse_labels, num_of_images, batch_size);
	if (FAILED(hr))
	{
		printf("Failed to load the image set.\n");
		return -1;
	}

	m_imageprocessor.SetImageTransformMode((ImageTransformMode)(/*IMG_TM_PADDING_RESIZE | */IMG_TM_RANDOM_CROP | IMG_TM_CENTER_CROP | IMG_TM_RANDOM_HORIZONTAL_FLIPPING));

	double lr = pLRMgr->GetLearningRate();
	double wd = isnan(weight_decay) ? RESNET_DEFAULT_WEIGHT_DECAY : (double)weight_decay;
	double m = isnan(momentum) ? RESNET_DEFAULT_MOMENTUM : momentum;
	ImageTransformMode image_tm = m_imageprocessor.GetImageTransformMode();

	printf("=======================================================================\n");
	printf("Image train-set enhancement:\n");
	printf("\trandom padding_resize: %s\n", (image_tm&IMG_TM_PADDING_RESIZE) ? "enable" : "disable");
	printf("\trandom throughout resize: %s\n", (image_tm&IMG_TM_RESIZE) ? "enable" : "disable");
	printf("\trandom crop: %s\n", (image_tm&IMG_TM_RANDOM_CROP) ? "enable" : "disable");
	printf("\trandom center crop: %s\n", (image_tm&IMG_TM_CENTER_CROP) ? "enable" : "disable");
	printf("\trandom horizontal flipping: %s\n", (image_tm&IMG_TM_RANDOM_HORIZONTAL_FLIPPING) ? "enable" : "disable");
	printf("optimizer: %s\n", OPTIM_NAME(optim_type));
	printf("weight decay: %s\n", doublecompactstring(wd).c_str());
	printf("momentum: %s\n", doublecompactstring(m).c_str());
	printf("learning rate: %s\n", doublecompactstring(lr).c_str());
	printf("batch size: %d\n", batch_size);
	printf("=======================================================================\n");

	auto criterion = torch::nn::CrossEntropyLoss();
	torch::optim::Optimizer* optimizer = NULL;
	if (optim_type == OPTIM_Adam)
		optimizer = new torch::optim::Adam(parameters(), torch::optim::AdamOptions(lr).weight_decay(wd));
	else
		optimizer = new torch::optim::SGD(parameters(), torch::optim::SGDOptions(lr).momentum(m).weight_decay(wd));

	tm_start = std::chrono::system_clock::now();

	torch::Tensor tensor_input;
	for (int64_t epoch = 1; epoch <= (int64_t)num_epoch; ++epoch)
	{
		auto running_loss = 0.;
		auto epoch_loss = 0.;
		auto correct = 0.;
		auto total = 0.;
		size_t nBatchProcessed = 0;
		std::vector<long long> label_batches;
		std::vector<long long> coarse_label_batches;

		m_imageprocessor.initImageSetBatchIter();

		// Take the image shuffle
		while(SUCCEEDED(m_imageprocessor.nextImageSetBatchIter(tensor_input, label_batches, coarse_label_batches)))
		{
			// dynamic learning rate if no learning rate is specified
			if (lr != pLRMgr->GetLearningRate())
			{
				printf("Change the learning rate from %s to %s\n", 
					doublecompactstring(lr).c_str(),
					doublecompactstring(pLRMgr->GetLearningRate()).c_str());
				lr = pLRMgr->GetLearningRate();
				for (auto& pg : optimizer->param_groups())
				{
					if (pg.has_options())
					{
						auto& options = pg.options();
						switch (optim_type)
						{
						case OPTIM_SGD:
							static_cast<torch::optim::SGDOptions&>(pg.options()).lr() = lr;
							break;
						case OPTIM_Adam:
							static_cast<torch::optim::AdamOptions&>(pg.options()).lr() = lr;
							break;
						case OPTIM_AdamW:
							static_cast<torch::optim::AdamWOptions&>(pg.options()).lr() = lr;
							break;
						case OPTIM_LBFGS:
							static_cast<torch::optim::LBFGSOptions&>(pg.options()).lr() = lr;
							break;
						case OPTIM_RMSprop:
							static_cast<torch::optim::RMSpropOptions&>(pg.options()).lr() = lr;
							break;
						case OPTIM_Adagrad:
							static_cast<torch::optim::AdagradOptions&>(pg.options()).lr() = lr;
							break;
						default:
							break;
						}
					}
				}
			}

			if (label_batches.size() == 0)
				continue;

			// Label在这里必须是一阶向量，里面元素必须是整数类型
			torch::Tensor tensor_label = torch::tensor(label_batches);

			optimizer->zero_grad();
			// 喂数据给网络
			auto outputs = forward(tensor_input);

			// 通过交叉熵计算损失
			auto loss = criterion(outputs, tensor_label);
			auto predicted = std::get<1>(torch::max(outputs, 1));
			// 反馈给网络，调整权重参数进一步优化
			loss.backward();
			optimizer->step();

			running_loss += loss.item().toFloat();
			epoch_loss += loss.item().toFloat();

			total += tensor_label.size(0);
			correct += predicted.eq(tensor_label).sum().item().toFloat();

			pLRMgr->OnTrainStepFinish(loss.item().toFloat());

			nBatchProcessed++;

			if (showloss_per_num_of_batches > 0)
			{
				if ((nBatchProcessed % showloss_per_num_of_batches) == 0)
				{
					printf("[%lld, %5zu] Loss: %.3f epoch Loss: %.3f | Acc: %.3f%%\n", epoch, nBatchProcessed,
						running_loss / showloss_per_num_of_batches, epoch_loss / nBatchProcessed, 100.*correct / total);
					running_loss = 0.;
				}
				else if (nBatchProcessed == (train_image_shuffle_set.size() + batch_size - 1) / batch_size)
				{
					printf("[%lld, %5zu] Loss: %.3f epoch Loss: %.3f | Acc: %.3f%%\n", epoch, nBatchProcessed,
						running_loss / (nBatchProcessed % showloss_per_num_of_batches), epoch_loss / nBatchProcessed, 100.*correct / total);
					running_loss = 0.;
				}
			}
		}

		pLRMgr->OnTrainEpochFinish();
	}

	m_imageprocessor.SetImageTransformMode(IMG_TM_PADDING_RESIZE);

	printf("Finish training!\n");

	tm_end = std::chrono::system_clock::now();
	long long train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
	printf("It took %lldh:%02dm:%02d.%03ds to finish training RESNET network!\n",
		train_duration / 1000 / 3600, (int)(train_duration / 1000 / 60 % 60), 
		(int)(train_duration / 1000 % 60), (int)(train_duration % 1000));

	m_image_labels = train_image_labels;

	savenet(szTrainSetStateFilePath);

	delete optimizer;

	return 0;
}

void ResNet::verify(const char* szImageSetRootPath, IMGSET_TYPE img_type)
{
	TCHAR szImageFile[MAX_PATH] = { 0 };
	// store the file name with the format 'classname/picture_file_name'
	std::vector<tstring> train_image_files;
	std::vector<tstring> train_image_labels;
	std::vector<tstring> train_image_coarse_labels;
	int num_of_images = 0;

	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;
	TCHAR szDirPath[MAX_PATH] = { 0 };

	TCHAR* tszImageSetRootPath = NULL;
	// Convert UTF-8 to Unicode
#ifdef _UNICODE
	wchar_t wszImageSetRootPath[MAX_PATH + 1] = { 0 };
	MultiByteToWideChar(CP_UTF8, 0, szImageSetRootPath, -1, wszImageSetRootPath, MAX_PATH + 1);
	tszImageSetRootPath = wszImageSetRootPath;
#else
	tszImageSetRootPath = szImageSetRootPath;
#endif

	HRESULT hr = m_imageprocessor.loadImageSet(img_type, tszImageSetRootPath, train_image_labels, train_image_coarse_labels, num_of_images, 1, false, false);
	if (FAILED(hr))
	{
		printf("Failed to load the image set.\n");
		return;
	}

	std::vector<long long> label_batches;
	std::vector<long long> coarse_label_batches;

	m_imageprocessor.initImageSetBatchIter();

	torch::NoGradGuard no_grade;

	eval();

	torch::Tensor tensor_input;
	int total_test_items = 0, passed_test_items = 0;
	// Take the image shuffle
	while (SUCCEEDED(m_imageprocessor.nextImageSetBatchIter(tensor_input, label_batches, coarse_label_batches)))
	{
		// Label在这里必须是一阶向量，里面元素必须是整数类型
		torch::Tensor tensor_label = torch::tensor(label_batches);

		auto outputs = forward(tensor_input);
		auto predicted = torch::max(outputs, 1);
		auto predicted_labels = std::get<1>(predicted);

		for (int64_t i = 0; i < predicted_labels.size(0); i++)
		{
			int predict_label = predicted_labels[i].item<int>();
			int fact_label = tensor_label[i].item<int>();

			_tprintf(_T("predicted: %s, fact: %s.\n"),
				predict_label >= 0 && predict_label < m_image_labels.size() ? m_image_labels[predict_label].c_str() : _T("Unknown"),
				m_image_labels[tensor_label[i].item<int>()].c_str());

			if (fact_label == predict_label)
				passed_test_items++;

			total_test_items++;
		}
	}

	tm_end = std::chrono::system_clock::now();

	long long verify_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
	printf("Total test items: %d, passed test items: %d, pass rate: %.3f%%, cost %lldh:%02dm:%02d.%03ds.\n",
		total_test_items, passed_test_items, passed_test_items*100.f / total_test_items,
		verify_duration / 1000 / 3600,
		(int)(verify_duration / 1000 / 60 % 60),
		(int)(verify_duration / 1000 % 60),
		(int)(verify_duration % 1000));
}

void ResNet::classify(const char* cszImageFile)
{
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;
	torch::Tensor tensor_input;
	TCHAR *tcsImageFile;

#ifdef _UNICODE
	wchar_t wszImageFilePath[MAX_PATH + 1] = { 0 };
	MultiByteToWideChar(CP_UTF8, 0, cszImageFile, -1, wszImageFilePath, MAX_PATH + 1);
	tcsImageFile = wszImageFilePath;
#else
	tcsImageFile = cszImageFile;
#endif

	if (m_imageprocessor.ToTensor(tcsImageFile, tensor_input) != 0)
	{
		printf("Failed to convert the image to tensor.\n");
		return;
	}

	//std::cout << tensor_input << '\n';

	auto outputs = forward(tensor_input);
	auto predicted = torch::max(outputs, 1);

	std::cout << std::get<0>(predicted) << '\n';
	std::cout << std::get<1>(predicted) << '\n';
	std::cout << outputs << '\n';

	tm_end = std::chrono::system_clock::now();

	_tprintf(_T("This image seems to %s, cost %lld msec.\n"),
		m_image_labels.size() > std::get<1>(predicted).item<int>() ? m_image_labels[std::get<1>(predicted).item<int>()].c_str() : _T("Unknown"),
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
}

int ResNet::savenet(const char* szTrainSetStateFilePath)
{
	// Save the net state to xxxx.pt and save the labels to xxxx.pt.label
	char szLabel[MAX_LABEL_NAME] = { 0 };

	try
	{
		torch::serialize::OutputArchive archive;

		// Add nested archive here
		c10::List<std::string> label_list;
		for (size_t i = 0; i < m_image_labels.size(); i++)
		{
			memset(szLabel, 0, sizeof(szLabel));
			WideCharToMultiByte(CP_UTF8, 0, m_image_labels[i].c_str(), -1, szLabel, MAX_LABEL_NAME, NULL, NULL);
			label_list.emplace_back((const char*)szLabel);
		}
		torch::IValue value(label_list);
		archive.write("RESNET_labels", label_list);

		// also save the current network configuration
		torch::IValue valNumClass(m_num_classes);
		archive.write("RESNET_num_of_class", valNumClass);

		torch::IValue valNetConfig((int64_t)m_RESNET_config);
		archive.write("RESNET_config", valNetConfig);

		torch::IValue valUseSmallSize(m_use_32x32_input);
		archive.write("RESNET_use_32x32_input", valUseSmallSize);

		// save the learning rate configuration

		save(archive);

		archive.save_to(szTrainSetStateFilePath);
	}
	catch (...)
	{
		printf("Failed to save the trained %s net state.\n", nnname().c_str());
		return -1;
	}
	printf("Save the training result to %s.\n", szTrainSetStateFilePath);

	return 0;
}

int ResNet::loadnet(const char* szPreTrainSetStateFilePath)
{
	wchar_t szLabel[MAX_LABEL_NAME] = { 0 };

	NN_TYPE nn_type = nntype_from_options();
	int64_t NN_enable_batch_norm = int64_from_options("NN::enable_batch_norm");
	int64_t NN_final_out_classes = int64_from_options("NN::final_out_classes");
	int64_t NN_use_32x32_input = int64_from_options("NN::use_32x32_input", 0);

	// if the pre-trained neutral network state file is not specified
	RESNET_CONFIG config = RESNET_UNKNOWN;
	switch (nn_type)
	{
	case NN_TYPE_RESNET18:  config = RESNET_18; break;
	case NN_TYPE_RESNET34:  config = RESNET_34; break;
	case NN_TYPE_RESNET50:  config = RESNET_50; break;
	case NN_TYPE_RESNET101: config = RESNET_101; break;
	case NN_TYPE_RESNET152: config = RESNET_152; break;
	}

	if (szPreTrainSetStateFilePath == NULL)
	{
		if (nn_type == NN_TYPE_UNKNOWN || NN_final_out_classes <= 0)
			return -1;

		if (config == RESNET_UNKNOWN)
			m_RESNET_config = RESNET_18;
		else
			m_RESNET_config = config;
		m_num_classes = NN_final_out_classes;
		m_use_32x32_input = NN_use_32x32_input ? true : false;

		m_imageprocessor.Init(m_use_32x32_input ? 32 : RESNET_INPUT_IMG_WIDTH, m_use_32x32_input ? 32 : RESNET_INPUT_IMG_HEIGHT);
		
		int iRet = -1;
		if ((iRet = _Init()) == 0)
		{
#if 0
			for (auto& m : modules(false))
			{
				if (m->name() == "torch::nn::Conv2dImpl")
				{
					printf("init the conv2d parameters.\n");
					auto spConv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m);
					//torch::nn::init::xavier_normal_(spConv2d->weight);
					torch::nn::init::kaiming_normal_(spConv2d->weight, 0.0, torch::kFanOut, torch::kReLU);
					//torch::nn::init::constant_(spConv2d->weight, 1);
				}
				else if (m->name() == "torch::nn::BatchNorm2dImpl")
				{
					printf("init the batchnorm2d parameters.\n");
					auto spBatchNorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m);
					torch::nn::init::constant_(spBatchNorm2d->weight, 1);
					torch::nn::init::constant_(spBatchNorm2d->bias, 0);
				}
				//else if (m->name() == "torch::nn::LinearImpl")
				//{
				//	auto spLinear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m);
				//	torch::nn::init::constant_(spLinear->weight, 1);
				//	torch::nn::init::constant_(spLinear->bias, 0);
				//}
			}

			//torch::Tensor a = torch::eye(64);
			//torch::Tensor c = torch::stack({ a, a, a });
			//c = c.unsqueeze(0);
			////std::cout << c << '\n';
			//c = forward(c);
			//std::cout << c << '\n';
			//abort();
#endif
		}
		return iRet;
	}

	try
	{
		torch::serialize::InputArchive archive;

		archive.load_from(szPreTrainSetStateFilePath);

		torch::IValue value;
		if (archive.try_read("RESNET_labels", value) && value.isList())
		{
			auto& label_list = value.toListRef();
			for (size_t i = 0; i < label_list.size(); i++)
			{
#ifdef _UNICODE
				if (MultiByteToWideChar(CP_UTF8, 0, label_list[i].toStringRef().c_str(), -1, szLabel, MAX_LABEL_NAME) <= 0)
					m_image_labels.push_back(_T("Unknown"));
				else
					m_image_labels.push_back(szLabel);
#else
				m_image_labels.push_back(label_list.get(i).toStringRef());
#endif
			}
		}

		archive.read("RESNET_num_of_class", value);
		m_num_classes = (int)value.toInt();

		archive.read("RESNET_config", value);
		m_RESNET_config = (RESNET_CONFIG)value.toInt();

		archive.read("RESNET_use_32x32_input", value);
		m_use_32x32_input = value.toBool();

		// Check the previous neutral network config is the same with current specified parameters
		if (config != RESNET_UNKNOWN && config != m_RESNET_config ||
			NN_final_out_classes > 0 && NN_final_out_classes != m_num_classes ||
			NN_use_32x32_input != -1 && (bool)NN_use_32x32_input != m_use_32x32_input)
		{
			printf("==========================================================================\n");
			printf("The pre-trained network config is different with the specified parameters:\n");
			if (config != RESNET_UNKNOWN && config != m_RESNET_config)
			{
				auto iter1 = _RESNET_CONFIG_NAMES.find(m_RESNET_config);
				auto iter2 = _RESNET_CONFIG_NAMES.find(config);
				printf("\tcurrent config: %s, the specified config: %s\n",
					iter1 != _RESNET_CONFIG_NAMES.end() ? iter1->second.c_str() : "Unknown",
					iter2 != _RESNET_CONFIG_NAMES.end() ? iter2->second.c_str() : "Unknown");
			}

			if (NN_final_out_classes > 0 && NN_final_out_classes != m_num_classes)
				printf("\tcurrent numclass: %d, the specified numclass: %lld\n", m_num_classes, NN_final_out_classes);

			if (NN_use_32x32_input != -1 && (bool)NN_use_32x32_input != m_use_32x32_input)
				printf("\tcurrent use_32x32_input: %s, the specified use_32x32_input: %s\n",
					m_use_32x32_input ? "yes" : "no", NN_use_32x32_input ? "yes" : "no");
			printf("==========================================================================\n");
			printf("Continue using the network config in the pre-train net state...\n");
		}

		m_imageprocessor.Init(m_use_32x32_input ? 32 : RESNET_INPUT_IMG_WIDTH, m_use_32x32_input ? 32 : RESNET_INPUT_IMG_HEIGHT);

		// Construct network layout,weight layers and so on
		if (_Init() < 0)
		{
			printf("Failed to initialize the current network {num_of_classes: %d, RESNET config: %d, use_32x32_input: %s}.\n",
				m_num_classes, m_RESNET_config, m_use_32x32_input ? "yes" : "no");
			return -1;
		}

		// Load the network state into the constructed neutral network
		load(archive);
		//for (auto const& p : named_parameters())
		//	std::cout << p.key() << ":\n" << p.value() << '\n';

	}
	catch (...)
	{
		printf("Failed to load the pre-trained %s net state.\n", nnname().c_str());
		return -1;
	}

	return 0;
}

void ResNet::Print()
{
	auto iter = _RESNET_CONFIG_NAMES.find(m_RESNET_config);
	if (iter != _RESNET_CONFIG_NAMES.end())
		printf("Neutral Network: %s\n", iter->second.c_str());

	printf("Enable Batch Normal: yes\n");
	printf("Use small input size 32x32: %s\n", m_use_32x32_input?"yes":"no");
	printf("number of output classes: %d\n", m_num_classes);

	BaseNNet::Print();
}

