#include "VGGNet.h"
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

#define VGG_INPUT_IMG_WIDTH						224
#define VGG_INPUT_IMG_HEIGHT					224
#define VGG_TRAIN_BATCH_SIZE					64

#define VGG_DEFAULT_WEIGHT_DECAY				0.0005
#define VGG_DEFAULT_MOMENTUM					0.9

std::map<VGG_CONFIG, std::string> _VGG_CONFIG_NAMES =
{
	{VGG_A,					"VGGA_NoBatchNorm"},
	{VGG_A_BATCHNORM,		"VGGA_BatchNorm"},
	{VGG_A_LRN,				"VGGA_LRN_NoBatchNorm"},
	{VGG_A_LRN_BATCHNORM,	"VGGA_LRN_BatchNorm"},
	{VGG_B,					"VGGB_NoBatchNorm"},
	{VGG_B_BATCHNORM,		"VGGB_BatchNorm"},
	{VGG_C,					"VGGC_NoBatchNorm"},
	{VGG_C_BATCHNORM,		"VGGC_BatchNorm"},
	{VGG_D,					"VGGD_NoBatchNorm"},
	{VGG_D_BATCHNORM,		"VGGD_BatchNorm"},
	{VGG_E,					"VGGD_NoBatchNorm"},
	{VGG_E_BATCHNORM,		"VGGD_BatchNorm"},
};

VGGNet::VGGNet()
	: m_VGG_config(VGG_UNKNOWN)
	, m_num_classes(-1)
	, m_use_32x32_input(false) {
}

VGGNet::~VGGNet()
{
	m_imageprocessor.Uninit();
	Uninit();
}

int VGGNet::_Init()
{
	int iRet = -1;

	if (m_bInit)
	{
		printf("The current neutral network is already initialized.\n");
		return 0;
	}

	if (m_VGG_config == VGG_UNKNOWN)
	{
		printf("Don't know the current net configuration.\n");
		return -1;
	}

	auto iter = _VGG_CONFIG_NAMES.find(m_VGG_config);
	if (iter != _VGG_CONFIG_NAMES.end())
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

int VGGNet::unloadnet()
{
	m_VGG_config = VGG_UNKNOWN;
	m_num_classes = -1;
	m_use_32x32_input = false;

	m_imageprocessor.Uninit();

	return Uninit();
}

int VGGNet::train(const char* szImageSetRootPath, 
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
	TCHAR szImageFile[MAX_PATH] = {0};
	// store the file name classname/picture_file_name
	std::vector<tstring> train_image_files;
	std::vector<tstring> train_image_labels;
	std::vector<size_t> train_image_shuffle_set;
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

	_tcscpy_s(szDirPath, MAX_PATH, tszImageSetRootPath);
	size_t ccDirPath = _tcslen(tszImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	if (FAILED(m_imageprocessor.loadImageSet(tszImageSetRootPath,
		train_image_files, train_image_labels, true)))
	{
		printf("Failed to load the train image/label set.\n");
		return -1;
	}

	m_imageprocessor.SetImageTransformMode((ImageTransformMode)(IMG_TM_PADDING_RESIZE | IMG_TM_RANDOM_CROP | IMG_TM_CENTER_CROP | IMG_TM_RANDOM_HORIZONTAL_FLIPPING));

	batch_size = batch_size < 0 ? 1 : batch_size;

	double lr = pLRMgr->GetLearningRate();
	double wd = isnan(weight_decay) ? VGG_DEFAULT_WEIGHT_DECAY : weight_decay;
	double m = isnan(momentum) ? VGG_DEFAULT_MOMENTUM : momentum;
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
	tm_end = std::chrono::system_clock::now();
	printf("It takes %lld msec to prepare training classifying cats and dogs.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

	tm_start = std::chrono::system_clock::now();
	
	torch::Tensor tensor_input;
	for (int64_t epoch = 1; epoch <= (int64_t)num_epoch; ++epoch)
	{
		auto running_loss = 0.;
		auto epoch_loss = 0.;
		size_t totals = 0;

		// Shuffle the list
		if (train_image_files.size() > 0)
		{
			// generate the shuffle list to train
			train_image_shuffle_set.resize(train_image_files.size());
			for (size_t i = 0; i < train_image_files.size(); i++)
				train_image_shuffle_set[i] = i;
			std::random_device rd;
			std::mt19937_64 g(rd());
			std::shuffle(train_image_shuffle_set.begin(), train_image_shuffle_set.end(), g);
		}

		// dynamic learning rate if no learning rate is specified
		if (lr != pLRMgr->GetLearningRate())
		{
			lr = pLRMgr->GetLearningRate();
			for (auto& pg : optimizer->param_groups())
			{
				if (pg.has_options())
				{
					auto& options = static_cast<torch::optim::SGDOptions&>(pg.options());
					options.lr() = lr;
				}
			}
		}

		// Take the image shuffle
		for(size_t i = 0;i<(train_image_shuffle_set.size() + batch_size -1)/ batch_size;i++)
		{
			std::vector<VGGNet::tstring> image_batches;
			std::vector<long long> label_batches;

			for (int b = 0; b < batch_size; b++)
			{
				size_t idx = i * batch_size + b;
				if (idx >= train_image_shuffle_set.size())
					break;

				tstring& strImgFilePath = train_image_files[train_image_shuffle_set[idx]];
				const TCHAR* cszImgFilePath = strImgFilePath.c_str();
				const TCHAR* pszTmp = _tcschr(cszImgFilePath, _T('\\'));

				if (pszTmp == NULL)
					continue;

				size_t label = 0;
				for (label = 0; label < train_image_labels.size(); label++)
					if (_tcsnicmp(train_image_labels[label].c_str(), cszImgFilePath,
						(pszTmp - cszImgFilePath) / sizeof(TCHAR)) == 0)
						break;

				if (label >= train_image_labels.size())
					continue;

				_stprintf_s(szImageFile, _T("%s\\training_set\\%s"), szDirPath, cszImgFilePath);

				image_batches.push_back(szImageFile);
				label_batches.push_back((long long)label);
			}

			if (image_batches.size() == 0)
				continue;
			
			if (m_imageprocessor.ToTensor(image_batches, tensor_input) != 0)
				continue;

			//_tprintf(_T("now training %s for the file: %s.\n"), 
			//	train_image_labels[label].c_str(), cszImgFilePath);
			// Label在这里必须是一阶向量，里面元素必须是整数类型
			torch::Tensor tensor_label = torch::tensor(label_batches);
			//tensor_label = tensor_label.view({ 1, -1 });

			totals += label_batches.size();

			optimizer->zero_grad();
			// 喂数据给网络
			auto outputs = forward(tensor_input);

			//std::cout << "tensor_label:" << tensor_label << "\noutputs.sizes(): " << outputs << '\n';

			//std::cout << outputs << '\n';
			//std::cout << tensor_label << '\n';

			// 通过交叉熵计算损失
			auto loss = criterion(outputs, tensor_label);
			// 反馈给网络，调整权重参数进一步优化
			loss.backward();
			optimizer->step();

			running_loss += loss.item().toFloat();
			epoch_loss += loss.item().toFloat();

			pLRMgr->OnTrainStepFinish(loss.item().toFloat());

			if (showloss_per_num_of_batches > 0 && ((i + 1) % showloss_per_num_of_batches == 0))
			{
				printf("[%lld, %5zu] loss: %.3f\n", epoch, i + 1, running_loss / showloss_per_num_of_batches);
				running_loss = 0.;
			}
		}

		pLRMgr->OnTrainEpochFinish();
	}

	m_imageprocessor.SetImageTransformMode(IMG_TM_PADDING_RESIZE);

	printf("Finish training!\n");

	tm_end = std::chrono::system_clock::now();
	long long train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
	printf("It took %lldh:%02dm:%02d.%03ds to finish training VGG network!\n",
		train_duration / 1000 / 3600,
		(int)(train_duration / 1000 / 60 % 60),
		(int)(train_duration / 1000 % 60),
		(int)(train_duration % 1000));

	m_image_labels = train_image_labels;
	savenet(szTrainSetStateFilePath);

	delete optimizer;

	return 0;
}

void VGGNet::verify(const char* szImageSetRootPath)
{
	TCHAR szImageFile[MAX_PATH] = { 0 };
	// store the file name with the format 'classname/picture_file_name'
	std::vector<tstring> test_image_files;
	std::vector<tstring> test_image_labels;
	std::vector<size_t> test_image_shuffle_set;
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

	_tcscpy_s(szDirPath, MAX_PATH, tszImageSetRootPath);
	size_t ccDirPath = _tcslen(tszImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	if (FAILED(m_imageprocessor.loadImageSet(tszImageSetRootPath,
		test_image_files, test_image_labels, false)))
	{
		printf("Failed to load the test image/label sets.\n");
		return;
	}

	// Shuffle the list
	if (test_image_files.size() > 0)
	{
		// generate the shuffle list to train
		test_image_shuffle_set.resize(test_image_files.size());
		for (size_t i = 0; i < test_image_files.size(); i++)
			test_image_shuffle_set[i] = i;
		std::random_device rd;
		std::mt19937_64 g(rd());
		std::shuffle(test_image_shuffle_set.begin(), test_image_shuffle_set.end(), g);
	}

	tm_end = std::chrono::system_clock::now();
	printf("It took %lld msec to load the test images/labels set.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
	tm_start = std::chrono::system_clock::now();

	tm_end = std::chrono::system_clock::now();
	printf("It took %lld msec to load the pre-trained network state.\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
	tm_start = std::chrono::system_clock::now();

	torch::Tensor tensor_input;
	int total_test_items = 0, passed_test_items = 0;
	for (size_t i = 0; i < test_image_shuffle_set.size(); i++)
	{
		tstring& strImgFilePath = test_image_files[test_image_shuffle_set[i]];
		const TCHAR* cszImgFilePath = strImgFilePath.c_str();
		const TCHAR* pszTmp = _tcschr(cszImgFilePath, _T('\\'));

		if (pszTmp == NULL)
			continue;

		size_t label = 0;
		for (label = 0; label < m_image_labels.size(); label++)
			if (_tcsnicmp(m_image_labels[label].c_str(), 
				cszImgFilePath, (pszTmp - cszImgFilePath) / sizeof(TCHAR)) == 0)
				break;

		if (label >= m_image_labels.size())
		{
			tstring strUnmatchedLabel(cszImgFilePath, (pszTmp - cszImgFilePath) / sizeof(TCHAR));
			_tprintf(_T("Can't find the test label: %s\n"), strUnmatchedLabel.c_str());
			continue;
		}

		_stprintf_s(szImageFile, _T("%s\\test_set\\%s"), szDirPath, cszImgFilePath);
		if (m_imageprocessor.ToTensor(szImageFile, tensor_input) != 0)
			continue;

		// Label在这里必须是一阶向量，里面元素必须是整数类型
		torch::Tensor tensor_label = torch::tensor({ (int64_t)label });

		//std::cout << "tensor_input.sizes:" << tensor_input.sizes() << '\n';
		//std::cout << "tensor_label.sizes:" << tensor_label.sizes() << '\n';

		auto outputs = forward(tensor_input);
		auto predicted = torch::max(outputs, 1);

		//_tprintf(_T("predicted: %s, fact: %s --> file: %s.\n"), 
		//	m_image_labels[std::get<1>(predicted).item<int>()].c_str(), 
		//	m_image_labels[tensor_label[0].item<int>()].c_str(), szImageFile);

		if (tensor_label[0].item<int>() == std::get<1>(predicted).item<int>())
			passed_test_items++;

		total_test_items++;

		//printf("label: %d.\n", labels[0].item<int>());
		//printf("predicted label: %d.\n", std::get<1>(predicted).item<int>());
		//std::cout << std::get<1>(predicted) << '\n';

		//break;
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

void VGGNet::classify(const char* cszImageFile)
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

	auto outputs = forward(tensor_input);
	auto predicted = torch::max(outputs, 1);

	//std::cout << std::get<0>(predicted) << '\n';
	//std::cout << std::get<1>(predicted) << '\n';

	tm_end = std::chrono::system_clock::now();

	_tprintf(_T("This image seems to %s, cost %lld msec.\n"),
		m_image_labels.size() > std::get<1>(predicted).item<int>()?m_image_labels[std::get<1>(predicted).item<int>()].c_str():_T("Unknown"),
		std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());
}

int VGGNet::savenet(const char* szTrainSetStateFilePath)
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
		archive.write("VGG_labels", label_list);

		// also save the current network configuration
		torch::IValue valNumClass(m_num_classes);
		archive.write("VGG_num_of_class", valNumClass);

		torch::IValue valNetConfig((int64_t)m_VGG_config);
		archive.write("VGG_config", valNetConfig);

		torch::IValue valUseSmallSize(m_use_32x32_input);
		archive.write("VGG_use_32x32_input", valUseSmallSize);

		save(archive);

		archive.save_to(szTrainSetStateFilePath);
	}
	catch (...)
	{
		printf("Failed to save the trained VGG net state.\n");
		return -1;
	}
	printf("Save the training result to %s.\n", szTrainSetStateFilePath);

	return 0;
}

int VGGNet::loadnet(const char* szPreTrainSetStateFilePath)
{
	wchar_t szLabel[MAX_LABEL_NAME] = { 0 };

	NN_TYPE nn_type = nntype_from_options();
	int64_t NN_enable_batch_norm = int64_from_options("NN::enable_batch_norm");
	int64_t NN_final_out_classes = int64_from_options("NN::final_out_classes");
	int64_t NN_use_32x32_input = int64_from_options("NN::use_32x32_input");

	// if the pre-trained neutral network state file is not specified
	VGG_CONFIG config = VGG_UNKNOWN;
	switch (nn_type)
	{
	case NN_TYPE_VGGA: config = NN_enable_batch_norm ? VGG_A_BATCHNORM : VGG_A; break;
	case NN_TYPE_VGGA_LRN: config = NN_enable_batch_norm ? VGG_A_LRN_BATCHNORM : VGG_A_LRN; break;
	case NN_TYPE_VGGB: config = NN_enable_batch_norm ? VGG_B_BATCHNORM : VGG_B; break;
	case NN_TYPE_VGGC: config = NN_enable_batch_norm ? VGG_C_BATCHNORM : VGG_C; break;
	case NN_TYPE_VGGD: config = NN_enable_batch_norm ? VGG_D_BATCHNORM : VGG_D; break;
	case NN_TYPE_VGGE: config = NN_enable_batch_norm ? VGG_E_BATCHNORM : VGG_E; break;
	}

	if (szPreTrainSetStateFilePath == NULL)
	{
		if (nn_type == NN_TYPE_UNKNOWN || NN_final_out_classes <= 0)
			return -1;

		if (config == VGG_UNKNOWN)
			m_VGG_config = NN_enable_batch_norm ? VGG_D_BATCHNORM : VGG_D;
		else
			m_VGG_config = config;
		m_num_classes = NN_final_out_classes;
		m_use_32x32_input = NN_use_32x32_input ? true : false;

		m_imageprocessor.Init(m_use_32x32_input ? 32 : VGG_INPUT_IMG_WIDTH, m_use_32x32_input ? 32 : VGG_INPUT_IMG_HEIGHT);

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
					spConv2d->reset_parameters();
					torch::nn::init::xavier_normal_(spConv2d->weight);

					//torch::nn::init::kaiming_normal_(spConv2d->weight, 0.0, torch::kFanOut, torch::kReLU);
					//torch::nn::init::constant_(spConv2d->weight, 1);
					if (spConv2d->options.bias())
						torch::nn::init::zeros_(spConv2d->bias);
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
#endif
		}
		return iRet;
	}

	try
	{
		torch::serialize::InputArchive archive;

		archive.load_from(szPreTrainSetStateFilePath);

		torch::IValue value;
		if (archive.try_read("VGG_labels", value) && value.isList())
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

		archive.read("VGG_num_of_class", value);
		m_num_classes = (int)value.toInt();

		archive.read("VGG_config", value);
		m_VGG_config = (VGG_CONFIG)value.toInt();
		m_bEnableBatchNorm = IS_BATCHNORM_ENABLED(m_VGG_config);

		archive.read("VGG_use_32x32_input", value);
		m_use_32x32_input = value.toBool();

		// Check the previous neutral network config is the same with current specified parameters
		if (config != VGG_UNKNOWN && config != m_VGG_config ||
			NN_final_out_classes > 0 && NN_final_out_classes != m_num_classes ||
			NN_use_32x32_input != -1 && (bool)NN_use_32x32_input != m_use_32x32_input)
		{
			printf("==========================================================================\n");
			printf("The pre-trained network config is different with the specified parameters:\n");
			if (config != VGG_UNKNOWN && config != m_VGG_config)
			{
				auto iter1 = _VGG_CONFIG_NAMES.find(m_VGG_config);
				auto iter2 = _VGG_CONFIG_NAMES.find(config);
				printf("\tcurrent config: %s, the specified config: %s\n",
					iter1 != _VGG_CONFIG_NAMES.end() ? iter1->second.c_str() : "Unknown",
					iter2 != _VGG_CONFIG_NAMES.end() ? iter2->second.c_str() : "Unknown");
			}

			if (NN_final_out_classes > 0 && NN_final_out_classes != m_num_classes)
				printf("\tcurrent numclass: %d, the specified numclass: %lld\n", m_num_classes, NN_final_out_classes);

			if (NN_use_32x32_input != -1 && (bool)NN_use_32x32_input != m_use_32x32_input)
				printf("\tcurrent use_32x32_input: %s, the specified use_32x32_input: %s\n",
					m_use_32x32_input ? "yes" : "no", NN_use_32x32_input ? "yes" : "no");
			printf("==========================================================================\n");
			printf("Continue using the network config in the pre-train net state...\n");
		}

		m_imageprocessor.Init(m_use_32x32_input ? 32 : VGG_INPUT_IMG_WIDTH, m_use_32x32_input ? 32 : VGG_INPUT_IMG_HEIGHT);

		// Construct network layout,weight layers and so on
		if (_Init() < 0)
		{
			printf("Failed to initialize the current network {num_of_classes: %d, VGG config: %d, use_32x32_input: %s}.\n",
				m_num_classes, m_VGG_config, m_use_32x32_input?"yes":"no");
			return -1;
		}

		// Load the network state into the constructed neutral network
		load(archive);
	}
	catch (...)
	{
		printf("Failed to load the pre-trained VGG net state.\n");
		return -1;
	}

	return 0;
}

void VGGNet::Print()
{
	auto iter = _VGG_CONFIG_NAMES.find(m_VGG_config);
	if (iter != _VGG_CONFIG_NAMES.end())
		printf("Neutral Network: %s\n", iter->second.c_str());

	printf("Enable Batch Normal: %s\n", m_bEnableBatchNorm?"yes":"no");

	BaseNNet::Print();
}

