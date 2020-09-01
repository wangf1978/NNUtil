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


#include <torch/torch.h>
#include <iostream>


struct BasicBlock : torch::nn::Module {

	static const int expansion;

	int64_t stride;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d bn2;
	torch::nn::Sequential downsample;

	BasicBlock(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
		torch::nn::Sequential downsample_ = torch::nn::Sequential())
		: conv1(torch::nn::Conv2dOptions(inplanes, planes, 3).stride(stride_).padding(1)),
		bn1(planes),
		conv2(torch::nn::Conv2dOptions( planes, planes, 3).stride(1).padding(1)),
		bn2(planes),
		downsample(downsample_)
	{
		register_module("conv1", conv1);
		register_module("bn1", bn1);
		register_module("conv2", conv2);
		register_module("bn2", bn2);
		stride = stride_;
		if (!downsample->is_empty()) {
			register_module("downsample", downsample);
		}
	}

	torch::Tensor forward(torch::Tensor x) {
		at::Tensor residual(x.clone());

		x = conv1->forward(x);
		x = bn1->forward(x);
		x = torch::relu(x);

		x = conv2->forward(x);
		x = bn2->forward(x);

		if (!downsample->is_empty()) {
			residual = downsample->forward(residual);
		}

		x += residual;
		x = torch::relu(x);

		return x;
	}
};

const int BasicBlock::expansion = 1;


struct BottleNeck : torch::nn::Module {

	static const int expansion;

	int64_t stride;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d bn2;
	torch::nn::Conv2d conv3;
	torch::nn::BatchNorm2d bn3;
	torch::nn::Sequential downsample;

	BottleNeck(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
		torch::nn::Sequential downsample_ = torch::nn::Sequential())
		: conv1(torch::nn::Conv2dOptions(inplanes, planes, 1)),
		bn1(planes),
		conv2(torch::nn::Conv2dOptions(planes, planes, 3).stride(stride_).padding(1)),
		bn2(planes),
		conv3(torch::nn::Conv2dOptions(planes, planes * expansion, 1)),
		bn3(planes * expansion),
		downsample(downsample_)
	{
		register_module("conv1", conv1);
		register_module("bn1", bn1);
		register_module("conv2", conv2);
		register_module("bn2", bn2);
		register_module("conv3", conv3);
		register_module("bn3", bn3);
		stride = stride_;
		if (!downsample->is_empty()) {
			register_module("downsample", downsample);
		}
	}

	torch::Tensor forward(torch::Tensor x) {
		at::Tensor residual(x.clone());

		x = conv1->forward(x);
		x = bn1->forward(x);
		x = torch::relu(x);

		x = conv2->forward(x);
		x = bn2->forward(x);
		x = torch::relu(x);

		x = conv3->forward(x);
		x = bn3->forward(x);

		if (!downsample->is_empty()) {
			residual = downsample->forward(residual);
		}

		x += residual;
		x = torch::relu(x);

		return x;
	}
};

const int BottleNeck::expansion = 4;


template <class Block> struct CResNet : torch::nn::Module {

	int64_t inplanes = 64;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Sequential layer1;
	torch::nn::Sequential layer2;
	torch::nn::Sequential layer3;
	torch::nn::Sequential layer4;
	torch::nn::Linear fc;

	CResNet(torch::IntArrayRef layers, int64_t num_classes = 1000)
		: conv1(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)),
		bn1(64),
		layer1(_make_layer(64, layers[0])),
		layer2(_make_layer(128, layers[1], 2)),
		layer3(_make_layer(256, layers[2], 2)),
		layer4(_make_layer(512, layers[3], 2)),
		fc(512 * Block::expansion, num_classes)
	{
		register_module("conv1", conv1);
		register_module("bn1", bn1);
		register_module("layer1", layer1);
		register_module("layer2", layer2);
		register_module("layer3", layer3);
		register_module("layer4", layer4);
		register_module("fc", fc);

		// Initializing weights
		for (auto& m : this->modules(false)) {
			if (m->name() == "torch::nn::Conv2dImpl")
			{
				printf("init the conv2d parameters.\n");
				auto spConv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m);
				torch::nn::init::xavier_normal_(spConv2d->weight);
				//torch::nn::init::kaiming_normal_(spConv2d->weight, 0.0, torch::kFanOut, torch::kReLU);
				//torch::nn::init::constant_(spConv2d->weight, 1);
			}
			else if (m->name() == "torch::nn::BatchNorm2dImpl")
			{
				printf("init the batchnorm2d parameters.\n");
				auto spBatchNorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m);
				torch::nn::init::constant_(spBatchNorm2d->weight, 1);
				torch::nn::init::constant_(spBatchNorm2d->bias, 0);
			}
		}
	}

	torch::Tensor forward(torch::Tensor x) {

		x = conv1->forward(x);
		x = bn1->forward(x);
		x = torch::relu(x);
		x = torch::max_pool2d(x, 3, 2, 1);

		x = layer1->forward(x);
		x = layer2->forward(x);
		x = layer3->forward(x);
		x = layer4->forward(x);

		x = torch::avg_pool2d(x, 7, 1);
		x = x.view({ x.sizes()[0], -1 });
		x = fc->forward(x);

		return x;
	}


private:
	torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1) {
		torch::nn::Sequential downsample;
		if (stride != 1 || inplanes != planes * Block::expansion) {
			downsample = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * Block::expansion, 1).stride(stride)),
				torch::nn::BatchNorm2d(planes * Block::expansion)
			);
		}
		torch::nn::Sequential layers;
		layers->push_back(Block(inplanes, planes, stride, downsample));
		inplanes = planes * Block::expansion;
		for (int64_t i = 0; i < blocks; i++) {
			layers->push_back(Block(inplanes, planes));
		}

		return layers;
	}
};


CResNet<BasicBlock> resnet18() {
	CResNet<BasicBlock> model({ 2, 2, 2, 2 });
	return model;
}

CResNet<BasicBlock> resnet34() {
	CResNet<BasicBlock> model({ 3, 4, 6, 3 });
	return model;
}

CResNet<BottleNeck> resnet50() {
	CResNet<BottleNeck> model({ 3, 4, 6, 3 });
	return model;
}

CResNet<BottleNeck> resnet101() {
	CResNet<BottleNeck> model({ 3, 4, 23, 3 });
	return model;
}

CResNet<BottleNeck> resnet152() {
	CResNet<BottleNeck> model({ 3, 8, 36, 3 });
	return model;
}

extern void PrintHelp();
extern void PrintHelpMenu();

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
	
	TCHAR szImageFilePath[MAX_PATH] = { 0 };
	ImageProcess imgTensor;
	torch::Tensor tensor;
	std::vector<tstring> image_labels;
	std::vector<tstring> image_coarse_labels;
	std::vector<int64_t> label_idxes;
	std::vector<int64_t> coarse_label_idxes;
	int number_of_imgs;
	int batch_idx = 0;
#if 0
	if (SUCCEEDED(imgTensor.loadImageSet(IMGSET_CIFAR_100, _T("I:\\CIFAR\\cifar-100-binary"), image_labels, image_coarse_labels, number_of_imgs, 64)))
	{
		imgTensor.initImageSetBatchIter();
		while (SUCCEEDED(imgTensor.nextImageSetBatchIter(tensor, label_idxes, coarse_label_idxes)))
		{
			_stprintf_s(szImageFilePath, MAX_PATH, _T("I:\\temp\\CIFA100_%d"), batch_idx);
			imgTensor.ToImage(tensor, szImageFilePath);
			batch_idx++;
		}
	}
	else
		printf("Failed to load image set CIFAR-100.\n");
#endif

	if (SUCCEEDED(imgTensor.loadImageSet(IMGSET_CIFAR_10, _T("I:\\CIFAR\\cifar-10-batches-bin"), image_labels, image_coarse_labels, number_of_imgs, 64)))
	{
		imgTensor.initImageSetBatchIter();
		while (SUCCEEDED(imgTensor.nextImageSetBatchIter(tensor, label_idxes, coarse_label_idxes)))
		{
			_stprintf_s(szImageFilePath, MAX_PATH, _T("I:\\temp\\CIFA10_%d"), batch_idx);
			imgTensor.ToImage(tensor, szImageFilePath);
			batch_idx++;
		}
	}
	else
		printf("Failed to load image set CIFAR-100.\n");

}

int _tmain(int argc, const TCHAR* argv[])
{
	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;

#if 0
	{
		if (FAILED(CoInitializeEx(NULL, COINIT_MULTITHREADED)))
		{
			return -1;
		}

		using tstring = std::basic_string<TCHAR, std::char_traits<TCHAR>, std::allocator<TCHAR>>;
		TCHAR szImageFile[MAX_PATH] = { 0 };
		// store the file name classname/picture_file_name
		std::vector<tstring> train_image_files;
		std::vector<tstring> train_image_labels;
		std::vector<size_t> train_image_shuffle_set;
		auto tm_start = std::chrono::system_clock::now();
		auto tm_end = tm_start;
		TCHAR szDirPath[MAX_PATH] = { 0 };
		TCHAR* tszImageSetRootPath = NULL;
		ImageProcess m_imageprocessor;
		int showloss_per_num_of_batches = 10;
		int batch_size = 64;

		m_imageprocessor.Init(224, 224);

		// Convert UTF-8 to Unicode
#ifdef _UNICODE
		wchar_t wszImageSetRootPath[MAX_PATH + 1] = { 0 };
		MultiByteToWideChar(CP_UTF8, 0, "I:\\CatDog", -1, wszImageSetRootPath, MAX_PATH + 1);
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


		auto rn18 = resnet18();
	
		

		m_imageprocessor.SetImageTransformMode((ImageTransformMode)(IMG_TM_PADDING_RESIZE | IMG_TM_RANDOM_CROP | IMG_TM_CENTER_CROP | IMG_TM_RANDOM_HORIZONTAL_FLIPPING));

		double lr = 0.001;
		double learning_rate = lr;
		auto criterion = torch::nn::CrossEntropyLoss();
		auto optimizer = torch::optim::SGD(rn18.parameters(), torch::optim::SGDOptions(lr).momentum(0.9).weight_decay(0.0005));
		tm_end = std::chrono::system_clock::now();
		printf("It takes %lld msec to prepare training classifying cats and dogs.\n",
			std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

		tm_start = std::chrono::system_clock::now();

		torch::Tensor tensor_input;
		for (int64_t epoch = 1; epoch <= (int64_t)4; ++epoch)
		{
			auto running_loss = 0.;
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
			if (learning_rate <= 0.f)
			{
				for (auto& pg : optimizer.param_groups())
				{
					if (pg.has_options())
					{
						auto& options = static_cast<torch::optim::SGDOptions&>(pg.options());
						options.lr() = lr;
					}
				}
			}

			// Take the image shuffle
			for (size_t i = 0; i < (train_image_shuffle_set.size() + batch_size - 1) / batch_size; i++)
			{
				std::vector<ResNet::tstring> image_batches;
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

					//_tprintf(_T("now training %s for the file: %s.\n"), 
					//	train_image_labels[label].c_str(), szImageFile);
				}

				if (image_batches.size() == 0)
					continue;

				if (m_imageprocessor.ToTensor(image_batches, tensor_input) != 0)
					continue;

				// Label在这里必须是一阶向量，里面元素必须是整数类型
				torch::Tensor tensor_label = torch::tensor(label_batches);
				//tensor_label = tensor_label.view({ 1, -1 });

				totals += label_batches.size();

				optimizer.zero_grad();
				// 喂数据给网络
				auto outputs = rn18.forward(tensor_input);

				//std::cout << "tensor_label:" << tensor_label << "\noutputs.sizes(): " << outputs << '\n';

				//std::cout << outputs << '\n';
				//std::cout << tensor_label << '\n';

				// 通过交叉熵计算损失
				auto loss = criterion(outputs, tensor_label);
				// 反馈给网络，调整权重参数进一步优化
				loss.backward();
				optimizer.step();

				running_loss += loss.item().toFloat();
				if (showloss_per_num_of_batches > 0 && ((i + 1) % showloss_per_num_of_batches == 0))
				{
					printf("[%lld, %5zu] loss: %.3f\n", epoch, i + 1, running_loss / showloss_per_num_of_batches);
					running_loss = 0.;
				}
			}

			// Adjust the learning rate dynamically
			if (learning_rate <= 0.f)
			{
				if (epoch % 2 == 0)
				{
					lr = lr * 0.1;
					if (lr < 0.00001)
						lr = 0.00001;
				}
			}
		}

		m_imageprocessor.SetImageTransformMode(IMG_TM_PADDING_RESIZE);

		printf("Finish training!\n");

		{
			// begin verifying the images

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
			MultiByteToWideChar(CP_UTF8, 0, "I:\\CatDog", -1, wszImageSetRootPath, MAX_PATH + 1);
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
				return -1;
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
				for (label = 0; label < train_image_labels.size(); label++)
					if (_tcsnicmp(train_image_labels[label].c_str(),
						cszImgFilePath, (pszTmp - cszImgFilePath) / sizeof(TCHAR)) == 0)
						break;

				if (label >= train_image_labels.size())
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

				auto outputs = rn18.forward(tensor_input);
				auto predicted = torch::max(outputs, 1);

				auto predicted_label = std::get<1>(predicted).item<int>();
				_tprintf(_T("predicted: %s, fact: %s --> file: %s.\n"),
					predicted_label >= 0 && predicted_label < train_image_labels.size() ? train_image_labels[predicted_label].c_str() : _T("Unknown"),
					train_image_labels[tensor_label[0].item<int>()].c_str(), szImageFile);

				if (tensor_label[0].item<int>() == std::get<1>(predicted).item<int>())
					passed_test_items++;

				total_test_items++;

				//printf("label: %d.\n", tensor_label[0].item<int>());
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

		return 0;
	}
#endif


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

	if (ctxCmd.verbose > 0)
		ctxCmd.Print();

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
		LearningRateMgr lrmgr(ctxCmd.learningrate);
		if (ctxCmd.image_set_root_path.size() == 0 || ctxCmd.train_net_state_path.size() == 0)
		{
			PrintHelp();
			goto done;
		}

		// construct the learning rate manager
		switch (ctxCmd.learningrate_decay_mode)
		{
		case LRD_MODE_PIECEWISE_CONSTANT:
			printf("Unsupported learning mode: piecewise_constant.\n");
			break;
		case LRD_MODE_EXPONENTIAL_DECAY:
			lrmgr.UseExponentDecay(ctxCmd.lr_decay_steps, ctxCmd.lr_decay_rate, ctxCmd.lr_staircase);
			break;
		case LRD_MODE_NATURAL_EXP_DECAY:
			lrmgr.UseNaturalExpDecay(ctxCmd.lr_decay_steps, ctxCmd.lr_decay_rate, ctxCmd.lr_staircase);
			break;
		case LRD_MODE_POLYNOMIAL_DECAY:
			lrmgr.UsePolyNomialDecay(ctxCmd.lr_decay_steps, ctxCmd.lr_end, ctxCmd.lr_power, ctxCmd.lr_cycle);
			break;
		case LRD_MODE_COSINE_DECAY:
			lrmgr.UseCosineDecay(ctxCmd.lr_decay_steps, ctxCmd.lr_alpha);
			break;
		case LRD_MODE_LINEAR_COSINE_DECAY:
			lrmgr.UseLinearCosineDecay(ctxCmd.lr_decay_steps, ctxCmd.num_classes, ctxCmd.lr_alpha, ctxCmd.lr_beta);
			break;
		case LRD_MODE_NOISY_LINEAR_COSINE_DECAY:
			printf("Unsupported learning mode: noisy linear cosine decay.\n");
			break;
		case LRD_MODE_COSINE_DECAY_RESTARTS:
			printf("Unsupported learning mode: cosine decay restarts.\n");
			break;
		case LRD_MODE_INVERSE_TIME_DECAY:
			lrmgr.UseInverseTimeDecay(ctxCmd.lr_decay_steps, ctxCmd.lr_decay_rate, ctxCmd.lr_staircase);
			break;
		}

		// delete the previous pre-train net state
		if (ctxCmd.clean_pretrain_net)
		{
			if (DeleteFileA(ctxCmd.train_net_state_path.c_str()) == FALSE)
				printf("Failed to delete the file '%s' {err: %lu}.\n", ctxCmd.train_net_state_path.c_str(), GetLastError());
			else
				printf("Successfully delete the pre-trailed file '%s'.\n", ctxCmd.train_net_state_path.c_str());
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
			ctxCmd.imgset_type,
			ctxCmd.train_net_state_path.c_str(),
			&lrmgr,
			ctxCmd.batchsize,
			ctxCmd.epochnum,
			ctxCmd.showloss_per_num_of_batches,
			ctxCmd.weight_decay,
			ctxCmd.momentum,
			ctxCmd.optim_type);
	}
	break;
	case NN_CMD_VERIFY:
	{
		bool bLoadSucc = false;
		if (ctxCmd.image_set_root_path.size() == 0)
		{
			PrintHelp();
			goto done;
		}

		if (ctxCmd.train_net_state_path.size() > 0)
		{
			// Try to load the net from the pre-trained file if it exist
			if (_access(ctxCmd.train_net_state_path.c_str(), 0) == 0)
			{
				if (ptrNet->loadnet(ctxCmd.train_net_state_path.c_str()) != 0)
					printf("Failed to load the network from %s, use untrained network to verify.\n", ctxCmd.train_net_state_path.c_str());
				else
					bLoadSucc = true;
			}
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

		ptrNet->verify(ctxCmd.image_set_root_path.c_str());
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

