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

#pragma once

enum VGG_CONFIG
{
	VGG_UNKNOWN = -1,
	VGG_A = 0,
	VGG_A_BATCHNORM,
	VGG_A_LRN,
	VGG_A_LRN_BATCHNORM,
	VGG_B,
	VGG_B_BATCHNORM,
	VGG_C,
	VGG_C_BATCHNORM,
	VGG_D,
	VGG_D_BATCHNORM,
	VGG_E,
	VGG_E_BATCHNORM
};

#define IS_BATCHNORM_ENABLED(c)	(\
	((c) == VGG_A_BATCHNORM ||\
	 (c) == VGG_A_LRN_BATCHNORM ||\
	 (c) == VGG_B_BATCHNORM ||\
	 (c) == VGG_C_BATCHNORM ||\
	 (c) == VGG_D_BATCHNORM ||\
	 (c) == VGG_E_BATCHNORM)?true:false)

class VGGNet : public BaseNNet
{
public:
	using tstring = std::basic_string<TCHAR, std::char_traits<TCHAR>, std::allocator<TCHAR>>;

	VGGNet();
	~VGGNet();

	int				train(
						const char* szTrainSetRootPath,
						IMGSET_TYPE img_type,
						const char* szTrainSetStateFilePath,
						LearningRateMgr* pLRMgr,
						int batch_size = 1, 
						int num_epoch = 1,
						unsigned int showloss_per_num_of_batches = 10,
						double weight_decay = NAN,
						double momentum = NAN,
						OPTIM_TYPE optim_type = OPTIM_SGD);
	void			verify(const char* szTrainSetRootPath, IMGSET_TYPE img_type);
	void			classify(const char* szImageFile);

	//
	// Load and save net
	//
	int				loadnet(const char* szTrainSetStateFilePath);
	int				savenet(const char* szTrainSetStateFilePath);
	int				unloadnet();

	void			Print();

	// get the property
	VGG_CONFIG		getcurrconfig() { return m_VGG_config; }
	int				getnumclasses() { return m_num_classes; }
	bool			isuse32x32input() { return m_use_32x32_input; }

protected:
	int				_Init();

protected:
	std::vector<tstring>
					m_image_labels;				// the image labels for this network
	ImageProcess	m_imageprocessor;
	bool			m_bEnableBatchNorm = true;
	int				m_num_classes = 1000;
	VGG_CONFIG		m_VGG_config;
	int				m_batch_size = 1;
	bool			m_use_32x32_input = false;
	bool			m_bInit = false;
};

