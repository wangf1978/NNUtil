#include <torch/torch.h>

#define NOMINMAX

#include <wincodec.h>
#include <wincodecsdk.h>
#include <wrl/client.h>
#include <d3d.h>
#include <d2d1.h>
#include <d2d1_2.h>
#include <shlwapi.h>
#include "ImageProcess.h"
#include "BaseNNet.h"

using namespace Microsoft::WRL;
using namespace torch::nn;
using namespace std;


enum RESNET_CONFIG
{
	RESNET_UNKNOWN = -1,
	RESNET_18  = 0,
	RESNET_34,
	RESNET_50,
	RESNET_101,
	RESNET_152,
};

#pragma once
class ResNet : public BaseNNet
{
public:
	using tstring = std::basic_string<TCHAR, std::char_traits<TCHAR>, std::allocator<TCHAR>>;

	ResNet();
	~ResNet();

	int				train(
						const char* szTrainSetRootPath, 
						const char* szTrainSetStateFilePath,
						int batch_size = 1, 
						int num_epoch = 1,
						float learning_rate = -1.0f,
						unsigned int showloss_per_num_of_batches = 10);
	void			verify(const char* szTrainSetRootPath, const char* szTrainSetStateFilePath);
	void			classify(const char* szImageFile);

	// Load and save net
	int				savenet(const char* szTrainSetStateFilePath);
	int				loadnet(const char* szTrainSetStateFilePath);
	int				loadnet(RESNET_CONFIG config, int num_classes, bool use_32x32_input = false);
	int				unloadnet();

	void			Print();

	// get the property
	RESNET_CONFIG	getcurrconfig() { return m_RESNET_config; }
	int				getnumclasses() { return m_num_classes; }
	bool			isuse32x32input() { return m_use_32x32_input; }

protected:
	int				_Init();

protected:
	std::vector<tstring>
					m_image_labels;				// the image labels for this network
	ImageProcess	m_imageprocessor;
	int				m_num_classes = 1000;
	RESNET_CONFIG	m_RESNET_config;
	int				m_batch_size = 1;
	bool			m_use_32x32_input = false;
	bool			m_bInit = false;
};


