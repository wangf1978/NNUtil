#include <torch/torch.h>

#define NOMINMAX

#include <wincodec.h>
#include <wincodecsdk.h>
#include <wrl/client.h>
#include <d3d.h>
#include <d2d1.h>
#include <d2d1_2.h>
#include <shlwapi.h>
#include <string>
#include <random>
#include "util.h"

#pragma once

using namespace Microsoft::WRL;

enum ImageTransformMode
{
	IMG_TM_PADDING_RESIZE = (1 << 0),
	IMG_TM_RESIZE = (1 << 1),
	IMG_TM_RANDOM_CROP = (1 << 2),
	IMG_TM_CENTER_CROP = (1 << 3),
	IMG_TM_RANDOM_HORIZONTAL_FLIPPING = (1<<6),
};

enum IMGSET_TYPE
{
	IMGSET_UNKNOWN = -1,
	IMGSET_FOLDER = 0,
	IMGSET_MNIST = 1,
	IMGSET_CIFAR_10 = 2,
	IMGSET_CIFAR_100 = 3,
	IMGSET_MAX
};

struct MNIST_INFO
{
	uint32_t		num_of_image_labels;
	uint32_t		num_of_images;
	uint32_t		image_width;
	uint32_t		image_height;
};

struct CIFAR10_FILE
{
	size_t			start_img_idx = 0;
	size_t			img_count = 0;
	FILE*			fp = NULL;
};

class ImageProcess
{
public:
	ImageProcess();
	~ImageProcess();

	HRESULT			Init(UINT outWidth, UINT outHeight);
	void			SetRGBMeansAndStds(float means[3], float stds[3]);
	void			SetGreyScaleMeanAndStd(float mean, float std);
	void			SetImageTransformMode(ImageTransformMode tm_modes);
	ImageTransformMode
					GetImageTransformMode();
	HRESULT			ToTensor(const TCHAR* cszImageFile, torch::Tensor& tensor);
	HRESULT			ToTensor(std::vector<tstring>& strImageFiles, torch::Tensor& tensor);
	HRESULT			ToCIFAR10Tensor(std::vector<size_t>& batchset, torch::Tensor& tensor, std::vector<int64_t>& labels);
	HRESULT			ToCIFAR100Tensor(std::vector<size_t>& batchset, torch::Tensor& tensor, std::vector<int64_t>& labels, std::vector<int64_t>& coarse_labels);
	HRESULT			ToMNISTTensor(std::vector<size_t>& batchset, torch::Tensor& tensor, std::vector<int64_t>& labels);
	HRESULT			ToImage(torch::Tensor& tensor, const TCHAR* szImageFilePrefix);
	void			Uninit();

	// 
	// Data set management
	// 
	// load image sets and labels
	HRESULT			loadImageSet(const TCHAR* szImageSetRootPath,
						std::vector<tstring>& image_files,
						std::vector<tstring>& image_labels,
						bool bTrainSet = true, bool bShuffle = true);
	HRESULT			loadLabels(const TCHAR* szImageSetRootPath, std::vector<tstring>& image_labels);
	HRESULT			loadCIFAR10Labels(const TCHAR* szImageSetRootPath, std::vector<tstring>& image_labels);
	HRESULT			loadCIFAR100Labels(const TCHAR* szImageSetRootPath, std::vector<tstring>& image_labels, std::vector<tstring>& image_coarse_labels);
	HRESULT			loadMNISTImageSet(const TCHAR* szImageSetRootPath, MNIST_INFO& MNIST_info, bool bTrainSet=true);
	HRESULT			loadCIFAR10ImageSet(const TCHAR* szImageSetRootPath, int& num_of_imgs, bool bTrainSet = true);
	HRESULT			loadCIFAR100ImageSet(const TCHAR* szImageSetRootPath, int& num_of_imgs, bool bTrainSet = true);
	HRESULT			loadImageSet(
						IMGSET_TYPE imgset_type,
						const TCHAR* szImageSetRootPath,
						std::vector<tstring>& image_labels,
						std::vector<tstring>& image_coarse_labels,
						int& number_of_imgs,
						int batch_size = 64,
						bool bTrainSet = true,
						bool bShuffle = true);
	HRESULT			initImageSetBatchIter();
	HRESULT			nextImageSetBatchIter(torch::Tensor& tensor, std::vector<int64_t>& labels, std::vector<int64_t>& coarse_labels);
	HRESULT			unloadImageSet();

	//
	// Utility functions
	//
	static void		SaveAs(ComPtr<IWICBitmap>& bitmap, PCWSTR filename);

protected:
	static bool		GetImageDrawRect(UINT target_width, UINT target_height, 
									 UINT image_width, UINT image_height, 
									 D2D1_RECT_F& dst_rect, D2D1_RECT_F & src_rect,
									 ImageTransformMode tm_modes);
	static ImageTransformMode
					GetImageLayoutTransform(ImageTransformMode tm_modes);
	HRESULT			loadLabelsFromFile(const TCHAR* szLabelFilePath, std::vector<tstring>& labels);

protected:
	ComPtr<ID2D1Factory>	
					m_spD2D1Factory;			// D2D1 factory
	ComPtr<IWICImagingFactory>	
					m_spWICImageFactory;		// Image codec factory
	ComPtr<IWICBitmap>		
					m_spNetInputBitmap;			// The final bitmap 1x224x224
	ComPtr<ID2D1RenderTarget>
					m_spRenderTarget;			// Render target to scale image
	ComPtr<ID2D1SolidColorBrush>
					m_spBGBrush;				// the background brush
	unsigned char*	m_pBGRABuf = NULL;

	UINT			m_outWidth = 0;
	UINT			m_outHeight = 0;

	float			m_RGB_means[3] = { 0.485f, 0.456f, 0.406f };
	float			m_RGB_stds[3] = { 0.229f, 0.224f, 0.225f };
	float			m_GreyScale_mean = 0.5f;
	float			m_GreyScale_std = 0.5f;

	ImageTransformMode
					m_image_transform_mode = IMG_TM_PADDING_RESIZE;
	static std::minstd_rand
					m_rand_eng;
	static std::uniform_real_distribution<float>
					m_real_dis;

	IMGSET_TYPE		m_imgset_type = IMGSET_UNKNOWN;
	TCHAR			m_szImageSetRootPath[MAX_PATH] = { 0 };
	int				m_batch_size = 1;
	bool			m_bTrainSet = true;
	bool			m_bShuffle = true;
	size_t			m_number_of_items = 0;
	std::vector<tstring> 
					m_image_files;
	std::vector<tstring> 
					m_image_labels;
	std::vector<tstring>
					m_image_coarse_labels;
	std::vector<size_t> 
					m_image_shuffle_set;
	std::vector<CIFAR10_FILE>
					m_CIFAR10_files;
	union
	{
		FILE*		m_fpCIFAR100Bin = NULL;
		FILE*		m_fpMNIST;
	};

	std::vector<uint8_t>
					m_MNISTLabels;

	int				m_imageset_iter_pos = -1;
};

