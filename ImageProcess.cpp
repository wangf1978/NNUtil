#include "ImageProcess.h"
#include <stdio.h>
#include <io.h>
#include <tchar.h>
#include "util.h"

extern void FreeBlob(void* p);

std::minstd_rand ImageProcess::m_rand_eng = std::minstd_rand(std::random_device{}());
std::uniform_real_distribution<float> ImageProcess::m_real_dis = std::uniform_real_distribution<float>(0.0f, 1.0f);

ImageProcess::ImageProcess()
{
	HRESULT hr = S_OK;

	// Create D2D1 factory to create the related render target and D2D1 objects
	D2D1_FACTORY_OPTIONS options;
	ZeroMemory(&options, sizeof(D2D1_FACTORY_OPTIONS));
#if defined(_DEBUG)
	// If the project is in a debug build, enable Direct2D debugging via SDK Layers.
	options.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif
	if (FAILED(hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_MULTI_THREADED,
		__uuidof(ID2D1Factory2), &options, &m_spD2D1Factory)))
		printf("Failed to create D2D1 factory {hr: 0X%X}.\n", hr);

	// Create the image factory
	if (FAILED(hr = CoCreateInstance(CLSID_WICImagingFactory,
		nullptr, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory, (LPVOID*)&m_spWICImageFactory)))
		printf("Failed to create WICImaging Factory {hr: 0X%X}.\n", hr);
}

ImageProcess::~ImageProcess()
{

}

HRESULT ImageProcess::Init(UINT outWidth, UINT outHeight)
{
	HRESULT hr = S_OK;
	if (outWidth == 0 || outHeight == 0)
	{
		// Use the original image width and height as the output width and height
		m_outWidth = outWidth;
		m_outHeight = outHeight;
		return hr;
	}

	// 创建一个Pre-multiplexed BGRA的224x224的WICBitmap
	if (SUCCEEDED(hr = m_spWICImageFactory->CreateBitmap(outWidth, outHeight, GUID_WICPixelFormat32bppPBGRA,
		WICBitmapCacheOnDemand, &m_spNetInputBitmap)))
	{
		// 在此WICBitmap上创建D2D1 Render Target
		D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT,
			D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED), 96, 96);
		if (SUCCEEDED(hr = m_spD2D1Factory->CreateWicBitmapRenderTarget(m_spNetInputBitmap.Get(), props, &m_spRenderTarget)))
		{
			hr = m_spRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Black, 1.0f), &m_spBGBrush);
		}
	}

	// Create a buffer to be used for converting ARGB to tensor
	if (SUCCEEDED(hr))
	{
		if (m_pBGRABuf != NULL)
			delete[] m_pBGRABuf;
		m_pBGRABuf = new unsigned char[outWidth*outHeight * 4];
		m_outWidth = outWidth;
		m_outHeight = outHeight;
	}

	return hr;
}

void ImageProcess::Uninit()
{
	if (m_pBGRABuf != NULL)
	{
		delete[] m_pBGRABuf;
		m_pBGRABuf = NULL;
	}
}

void ImageProcess::SetRGBMeansAndStds(float means[3], float stds[3])
{
	for (int i = 0; i < 3; i++)
	{
		m_RGB_means[i] = means[i];
		m_RGB_stds[i] = stds[i];
	}
}

void ImageProcess::SetGreyScaleMeanAndStd(float mean, float std)
{
	m_GreyScale_mean = mean;
	m_GreyScale_std = std;
}

void ImageProcess::SetImageTransformMode(ImageTransformMode tm_mode)
{
	m_image_transform_mode = tm_mode;
}

ImageTransformMode ImageProcess::GetImageTransformMode()
{
	return m_image_transform_mode;
}

bool ImageProcess::GetImageDrawRect(UINT target_width, UINT target_height, UINT image_width, UINT image_height, 
	D2D1_RECT_F& dst_rect, D2D1_RECT_F & src_rect, ImageTransformMode tm_modes)
{
	if (target_width == 0 || target_height == 0 || image_width == 0 || image_height == 0)
		return false;

	src_rect.left = src_rect.top = 0.f;
	src_rect.bottom = image_height;
	src_rect.right = image_width;

	if (tm_modes&IMG_TM_PADDING_RESIZE)
	{
		if (target_width*image_height >= image_width * target_height)
		{
			// align with height
			FLOAT ratio = (FLOAT)target_height / image_height;
			dst_rect.top = 0.f;
			dst_rect.bottom = (FLOAT)target_height;
			dst_rect.left = (target_width - image_width * ratio) / 2.0f;
			dst_rect.right = (target_width + image_width * ratio) / 2.0f;
		}
		else
		{
			// align with width
			FLOAT ratio = (FLOAT)target_width / image_width;
			dst_rect.left = 0.f;
			dst_rect.right = (FLOAT)target_width;
			dst_rect.top = (target_height - image_height * ratio) / 2.0f;
			dst_rect.bottom = (target_height + image_height * ratio) / 2.0f;
		}
	}
	else if (tm_modes&IMG_TM_RESIZE)
	{
		dst_rect.left = 0.f;
		dst_rect.top = 0.f;
		dst_rect.right = (FLOAT)target_width;
		dst_rect.left = (FLOAT)target_height;
	}
	else if (tm_modes&IMG_TM_RANDOM_CROP)
	{
		/*
		|----------------|         |-----------|
		|                |         |           |
		|                |  ==>    |           |
		|________________|         |___________|
		calculate the source clip area

		*/
		dst_rect.left = dst_rect.top = 0.f;
		dst_rect.right = target_width;
		dst_rect.bottom = target_height;
		if (target_width*image_height <= image_width * target_height)
		{
			// align with height
			src_rect.top = 0.f;
			src_rect.bottom = (FLOAT)image_height;

			FLOAT expected_width = (FLOAT)image_height * target_width / target_height;
			FLOAT random_range = image_width - expected_width;
			FLOAT x = random_range * rand() / RAND_MAX;

			if (x + expected_width > image_width)
				x = image_width - expected_width;

			if (x < 0.001f)
				x = 0.f;

			src_rect.left = x;
			src_rect.right = x + expected_width;
		}
		else
		{
			// align with width
			src_rect.left = 0.f;
			src_rect.right = (FLOAT)image_width;

			FLOAT expected_height = (FLOAT)image_width * target_height / target_width;
			FLOAT random_range = image_height - expected_height;
			FLOAT y = random_range * rand() / RAND_MAX;

			if (y + expected_height > image_height)
				y = image_height - expected_height;

			if (y < 0.001f)
				y = 0.f;

			src_rect.top = y;
			src_rect.bottom = y + expected_height;
		}
	}
	else if (tm_modes&IMG_TM_CENTER_CROP)
	{
		/*
		|----------------|         |-----------|
		|                |         |           |
		|                |  ==>    |           |
		|________________|         |___________|
		calculate the source clip area

		*/
		dst_rect.left = dst_rect.top = 0.f;
		dst_rect.right = target_width;
		dst_rect.bottom = target_height;
		if (target_width*image_height <= image_width * target_height)
		{
			// align with height
			src_rect.top = 0.f;
			src_rect.bottom = (FLOAT)image_height;

			FLOAT expected_width = (FLOAT)image_height * target_width / target_height;
			FLOAT random_range = image_width - expected_width;
			FLOAT x = random_range / 2.f;

			if (x + expected_width > image_width)
				x = image_width - expected_width;

			if (x < 0.001f)
				x = 0.f;

			src_rect.left = x;
			src_rect.right = x + expected_width;
		}
		else
		{
			// align with width
			src_rect.left = 0.f;
			src_rect.right = (FLOAT)image_width;

			FLOAT expected_height = (FLOAT)image_width * target_height / target_width;
			FLOAT random_range = image_height - expected_height;
			FLOAT y = random_range / 2.f;

			if (y + expected_height > image_height)
				y = image_height - expected_height;

			if (y < 0.001f)
				y = 0.f;

			src_rect.top = y;
			src_rect.bottom = y + expected_height;
		}
	}

	return true;
}

ImageTransformMode ImageProcess::GetImageLayoutTransform(ImageTransformMode tm_modes)
{
	// check how many image layout transform is available
	ImageTransformMode modes[32];
	int num_modes = 0;

	if (tm_modes&IMG_TM_PADDING_RESIZE)
		modes[num_modes++] = IMG_TM_PADDING_RESIZE;

	if (tm_modes&IMG_TM_RESIZE)
		modes[num_modes++] = IMG_TM_RESIZE;

	if (tm_modes&IMG_TM_RANDOM_CROP)
		modes[num_modes++] = IMG_TM_RANDOM_CROP;

	if (tm_modes&IMG_TM_CENTER_CROP)
		modes[num_modes++] = IMG_TM_CENTER_CROP;

	if (num_modes <= 0)
		return IMG_TM_RANDOM_CROP;

	float dis = m_real_dis(m_rand_eng);

	int n = (int)(num_modes * dis);
	if (n >= num_modes)
		n = num_modes - 1;
	if (n < 0)
		n = 0;

	return modes[n];
}

HRESULT ImageProcess::ToTensor(const TCHAR* cszImageFile, torch::Tensor& tensor)
{
	HRESULT hr = S_OK;
	ComPtr<IWICBitmapDecoder> spDecoder;				// Image decoder
	ComPtr<IWICBitmapFrameDecode> spBitmapFrameDecode;	// Decoded image
	ComPtr<IWICBitmapSource> spConverter;				// Converted image
	ComPtr<IWICBitmap> spHandWrittenBitmap;				// The original bitmap
	ComPtr<ID2D1Bitmap> spD2D1Bitmap;					// D2D1 bitmap

	ComPtr<IWICBitmap> spNetInputBitmap = m_spNetInputBitmap;
	ComPtr<ID2D1RenderTarget> spRenderTarget = m_spRenderTarget;
	ComPtr<ID2D1SolidColorBrush> spBGBrush = m_spBGBrush;

	BOOL bDynamic = FALSE;
	UINT uiFrameCount = 0;
	UINT uiWidth = 0, uiHeight = 0;
	UINT outWidth = m_outWidth;
	UINT outHeight = m_outHeight;
	WICPixelFormatGUID pixelFormat;
	unsigned char* pBGRABuf = m_pBGRABuf;
	D2D1_RECT_F dst_rect = { 0.f, 0.f, (FLOAT)outWidth, (FLOAT)outHeight };
	D2D1_RECT_F src_rect = { 0.f, 0.f, 0.f,0.f };
	WICRect rect = { 0, 0, (INT)outWidth, (INT)outHeight };
	
	if (cszImageFile == NULL || _taccess(cszImageFile, 0) != 0)
		return E_INVALIDARG;

	wchar_t* wszInputFile = NULL;
	size_t cbFileName = _tcslen(cszImageFile);
#ifndef _UNICODE
	wszInputFile = new wchar_t[cbFileName + 1];
	if (MultiByteToWideChar(CP_UTF8, 0, cszCatImageFile, -1, wszInputFile, cbFileName + 1) == 0)
	{
		delete[] wszInputFile;
		return -1;
	}
#else
	wszInputFile = (wchar_t*)cszImageFile;
#endif

	// 加载图片, 并为其创建图像解码器
	if (FAILED(m_spWICImageFactory->CreateDecoderFromFilename(wszInputFile, NULL,
		GENERIC_READ, WICDecodeMetadataCacheOnDemand, &spDecoder)))
		goto done;

	// 得到多少帧图像在图片文件中，如果无可解帧，结束程序
	if (FAILED(hr = spDecoder->GetFrameCount(&uiFrameCount)) || uiFrameCount == 0)
		goto done;

	// 得到第一帧图片
	if (FAILED(hr = hr = spDecoder->GetFrame(0, &spBitmapFrameDecode)))
		goto done;

	// 得到图片大小
	if (FAILED(hr = spBitmapFrameDecode->GetSize(&uiWidth, &uiHeight)))
		goto done;

	// 调整转换和输出
	if (outWidth == 0)
	{
		outWidth = uiWidth;
		dst_rect.right = uiWidth;
		rect.Width = uiWidth;
		bDynamic = TRUE;
	}

	if (outHeight == 0)
	{
		outHeight = uiHeight;
		dst_rect.bottom = uiHeight;
		rect.Height = uiHeight;
		bDynamic = TRUE;
	}

	// Create a buffer to be used for converting ARGB to tensor
	if (bDynamic)
		pBGRABuf = new unsigned char[outWidth*outHeight * 4];

	// 得到图片像素格式
	if (FAILED(hr = spBitmapFrameDecode->GetPixelFormat(&pixelFormat)))
		goto done;

	// 如果图片不是Pre-multiplexed BGRA格式，转化成这个格式，以便用D2D硬件处理图形转换
	if (!IsEqualGUID(pixelFormat, GUID_WICPixelFormat32bppPBGRA))
	{
		if (FAILED(hr = WICConvertBitmapSource(GUID_WICPixelFormat32bppPBGRA,
			spBitmapFrameDecode.Get(), &spConverter)))
			goto done;
	}
	else
		spConverter = spBitmapFrameDecode;

	// If the width and height are not matched with the image width and height, scale the image
	if (!bDynamic && (outWidth != uiWidth || outHeight != uiHeight))
	{
		// 转化为Pre-multiplexed BGRA格式的WICBitmap
		if (FAILED(hr = m_spWICImageFactory->CreateBitmapFromSource(
			spConverter.Get(), WICBitmapCacheOnDemand, &spHandWrittenBitmap)))
			goto done;

		// 将转化为Pre-multiplexed BGRA格式的WICBitmap的原始图片转换到D2D1Bitmap对象中来，以便后面的缩放处理
		if (FAILED(hr = spRenderTarget->CreateBitmapFromWicBitmap(spHandWrittenBitmap.Get(), &spD2D1Bitmap)))
			goto done;

		// 将图片进行缩放处理，转化为m_outWidthxm_outHeight的图片
		spRenderTarget->BeginDraw();

		spRenderTarget->FillRectangle(dst_rect, spBGBrush.Get());

		if (GetImageDrawRect(outWidth, outHeight, uiWidth, uiHeight, dst_rect, src_rect, GetImageLayoutTransform(m_image_transform_mode)))
		{
			bool bRandHoriFlip = false;
			if (m_image_transform_mode&IMG_TM_RANDOM_HORIZONTAL_FLIPPING)
			{
				if (m_real_dis(m_rand_eng) > 0.5f)
					bRandHoriFlip = true;
			}

			if (bRandHoriFlip)
				spRenderTarget->SetTransform(D2D1::Matrix3x2F(-1.f, 0.f, 0.f, 1.f, outWidth, 0.f));

			spRenderTarget->DrawBitmap(spD2D1Bitmap.Get(), &dst_rect, 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, &src_rect);

			if (bRandHoriFlip)
				spRenderTarget->SetTransform(D2D1::IdentityMatrix());
		}

		spRenderTarget->EndDraw();

		//{
		//	TCHAR szFileName[MAX_PATH] = { 0 };
		//	const TCHAR* pszTmp = _tcsrchr(cszImageFile, _T('\\'));
		//	if (pszTmp != NULL)
		//		_stprintf_s(szFileName, MAX_PATH, _T("I:\\temp\\transform_%s"), pszTmp + 1);
		//	else
		//		_tcscpy_s(szFileName, MAX_PATH, L"I:\\temp\\test.png");

		//	ImageProcess::SaveAs(spNetInputBitmap, szFileName);
		//}

		// 并将图像每个channel中数据转化为[-1.0, 1.0]的raw data
		hr = spNetInputBitmap->CopyPixels(&rect, outWidth * 4, 4 * outWidth * outHeight, pBGRABuf);
	}
	else
		hr = spConverter->CopyPixels(&rect, outWidth * 4, 4 * outWidth * outHeight, pBGRABuf);

	float* res_data = (float*)malloc(3 * outWidth * outHeight * sizeof(float));
	for (int c = 0; c < 3; c++)
	{
		for (UINT i = 0; i < outHeight; i++)
		{
			for (UINT j = 0; j < outWidth; j++)
			{
				int pos = c * outWidth*outHeight + i * outWidth + j;
				res_data[pos] = ((pBGRABuf[i * outWidth * 4 + j * 4 + 2 - c]) / 255.0f - m_RGB_means[c]) / m_RGB_stds[c];
			}
		}
	}

	tensor = torch::from_blob(res_data, { 1, 3, outWidth, outHeight }, FreeBlob);

	hr = S_OK;

done:
	if (wszInputFile != NULL && wszInputFile != cszImageFile)
		delete[] wszInputFile;

	if (pBGRABuf != m_pBGRABuf)
		delete[] pBGRABuf;

	return hr;
}

HRESULT ImageProcess::ToTensor(std::vector<tstring>& strImageFiles, torch::Tensor& tensor)
{
	HRESULT hr = S_OK;
	ComPtr<IWICBitmapDecoder> spDecoder;				// Image decoder
	ComPtr<IWICBitmapFrameDecode> spBitmapFrameDecode;	// Decoded image
	ComPtr<IWICBitmapSource> spConverter;				// Converted image
	ComPtr<IWICBitmap> spHandWrittenBitmap;				// The original bitmap
	ComPtr<ID2D1Bitmap> spD2D1Bitmap;					// D2D1 bitmap

	ComPtr<IWICBitmap> spNetInputBitmap = m_spNetInputBitmap;
	ComPtr<ID2D1RenderTarget> spRenderTarget = m_spRenderTarget;
	ComPtr<ID2D1SolidColorBrush> spBGBrush = m_spBGBrush;

	UINT uiFrameCount = 0;
	UINT uiWidth = 0, uiHeight = 0;
	UINT outWidth = m_outWidth;
	UINT outHeight = m_outHeight;
	WICPixelFormatGUID pixelFormat;
	unsigned char* pBGRABuf = m_pBGRABuf;
	D2D1_RECT_F dst_rect = { 0.f, 0.f, (FLOAT)outWidth, (FLOAT)outHeight };
	D2D1_RECT_F src_rect = { 0.f, 0.f, 0.f, 0.f };
	WICRect rect = { 0, 0, (INT)outWidth, (INT)outHeight };
	wchar_t wszImageFile[MAX_PATH + 1] = { 0 };
	const wchar_t* wszInputFile = NULL;
	float* res_data = NULL;

	static long long toTensorDuration = 0LL;

	auto tm_start = std::chrono::system_clock::now();
	auto tm_end = tm_start;

	if (strImageFiles.size() == 0)
		return E_INVALIDARG;

	if (outWidth != 0 && outHeight != 0)
		res_data = new float[strImageFiles.size() * 3 * outWidth*outHeight];

	for (size_t b = 0; b < strImageFiles.size(); b++)
	{
		size_t pos = 0;
		BOOL bDynamic = FALSE;
#ifndef _UNICODE
		if (MultiByteToWideChar(CP_UTF8, 0, strImageFiles[b].c_str(), -1, wszImageFile, MAX_PATH + 1) == 0)
		{
			hr = E_FAIL;
			goto done;
		}
		wszInputFile = (const wchar_t*)wszImageFile;
#else
		wszInputFile = strImageFiles[b].c_str();
#endif

		// 加载图片, 并为其创建图像解码器
		if (FAILED(m_spWICImageFactory->CreateDecoderFromFilename(wszInputFile, NULL,
			GENERIC_READ, WICDecodeMetadataCacheOnDemand, &spDecoder)))
			goto done;

		// 得到多少帧图像在图片文件中，如果无可解帧，结束此函数
		if (FAILED(hr = spDecoder->GetFrameCount(&uiFrameCount)) || uiFrameCount == 0)
			goto done;

		// 得到第一帧图片
		if (FAILED(hr = hr = spDecoder->GetFrame(0, &spBitmapFrameDecode)))
			goto done;

		// 得到图片大小
		if (FAILED(hr = spBitmapFrameDecode->GetSize(&uiWidth, &uiHeight)))
			goto done;

		// 调整转换和输出
		if (outWidth == 0)
		{
			outWidth = uiWidth;
			dst_rect.right = uiWidth;
			rect.Width = uiWidth;
			bDynamic = TRUE;
		}

		if (outHeight == 0)
		{
			outHeight = uiHeight;
			dst_rect.bottom = uiHeight;
			rect.Height = uiHeight;
			bDynamic = TRUE;
		}

		// Create a buffer to be used for converting ARGB to tensor
		if (bDynamic)
		{
			if (pBGRABuf == NULL)
				pBGRABuf = new unsigned char[outWidth*outHeight * 4];

			if (res_data == NULL)
				res_data = new float[strImageFiles.size() * 3 * outWidth*outHeight];
		}

		// 得到图片像素格式
		if (FAILED(hr = spBitmapFrameDecode->GetPixelFormat(&pixelFormat)))
			goto done;

		// 如果图片不是Pre-multiplexed BGRA格式，转化成这个格式，以便用D2D硬件处理图形转换
		if (!IsEqualGUID(pixelFormat, GUID_WICPixelFormat32bppPBGRA))
		{
			if (FAILED(hr = WICConvertBitmapSource(GUID_WICPixelFormat32bppPBGRA,
				spBitmapFrameDecode.Get(), &spConverter)))
				goto done;
		}
		else
			spConverter = spBitmapFrameDecode;

		// If the width and height are not matched with the image width and height, scale the image
		if (!bDynamic && (outWidth != uiWidth || outHeight != uiHeight))
		{
			// 转化为Pre-multiplexed BGRA格式的WICBitmap
			if (FAILED(hr = m_spWICImageFactory->CreateBitmapFromSource(
				spConverter.Get(), WICBitmapCacheOnDemand, &spHandWrittenBitmap)))
				goto done;

			// 将转化为Pre-multiplexed BGRA格式的WICBitmap的原始图片转换到D2D1Bitmap对象中来，以便后面的缩放处理
			if (FAILED(hr = spRenderTarget->CreateBitmapFromWicBitmap(spHandWrittenBitmap.Get(), &spD2D1Bitmap)))
				goto done;

			// 将图片进行缩放处理，转化为m_outWidthxm_outHeight的图片
			spRenderTarget->BeginDraw();

			spRenderTarget->FillRectangle(dst_rect, spBGBrush.Get());

			if (GetImageDrawRect(outWidth, outHeight, uiWidth, uiHeight, dst_rect, src_rect, GetImageLayoutTransform(m_image_transform_mode)))
			{
				bool bRandHoriFlip = false;
				if (m_image_transform_mode&IMG_TM_RANDOM_HORIZONTAL_FLIPPING)
				{
					if (rand() * 2 > RAND_MAX)
						bRandHoriFlip = true;
				}

				if (bRandHoriFlip)
					spRenderTarget->SetTransform(D2D1::Matrix3x2F(-1.f, 0.f, 0.f, 1.f, outWidth, 0.f));

				spRenderTarget->DrawBitmap(spD2D1Bitmap.Get(), &dst_rect, 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, &src_rect);

				if (bRandHoriFlip)
					spRenderTarget->SetTransform(D2D1::IdentityMatrix());
			}

			spRenderTarget->EndDraw();

			//{
			//	TCHAR szFileName[MAX_PATH] = { 0 };
			//	const TCHAR* pszTmp = _tcsrchr(strImageFiles[b].c_str(), _T('\\'));
			//	if (pszTmp != NULL)
			//		_stprintf_s(szFileName, MAX_PATH, _T("I:\\temp\\transform_%s"), pszTmp + 1);
			//	else
			//		_tcscpy_s(szFileName, MAX_PATH, L"I:\\temp\\test.png");

			//	ImageProcess::SaveAs(spNetInputBitmap, szFileName);
			//}

			// 并将图像每个channel中数据转化为[-1.0, 1.0]的raw data
			hr = spNetInputBitmap->CopyPixels(&rect, outWidth * 4, 4 * outWidth * outHeight, pBGRABuf);
		}
		else
			hr = spConverter->CopyPixels(&rect, outWidth * 4, 4 * outWidth * outHeight, pBGRABuf);

		pos = b * 3 * outWidth*outHeight;
		for (int c = 0; c < 3; c++)
		{
			for (UINT i = 0; i < outHeight; i++)
			{
				for (UINT j = 0; j < outWidth; j++)
				{
					size_t cpos = pos + c * outWidth*outHeight + i * outWidth + j;
					res_data[cpos] = ((pBGRABuf[i * outWidth * 4 + j * 4 + 2 - c]) / 255.0f - m_RGB_means[c]) / m_RGB_stds[c];
				}
			}
		}
	}

	tensor = torch::from_blob(res_data, { (long long)strImageFiles.size(), 3, outWidth, outHeight }, FreeBlob);

	hr = S_OK;

done:
	if (pBGRABuf != m_pBGRABuf)
		delete[] pBGRABuf;

	tm_end = std::chrono::system_clock::now();
	toTensorDuration += std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();

	//printf("Load batch tensors, cost %lldh:%02dm:%02d.%03ds\n", 
	//	toTensorDuration/1000/3600, 
	//	(int)(toTensorDuration/1000/60%60), 
	//	(int)(toTensorDuration/1000%60),
	//	(int)(toTensorDuration%1000));

	return hr;
}

/* Pytorch use NCWH layout for tensor */
HRESULT ImageProcess::ToCIFAR10Tensor(std::vector<size_t>& batchset, torch::Tensor& tensor, std::vector<int64_t>& labels)
{
	if (batchset.size() == 0)
		return E_INVALIDARG;

	uint8_t buf[3073];
	size_t pos = 0;
	float* res_data = new float[batchset.size() * 3 * 32 * 32];
	for (size_t i = 0; i < batchset.size(); i++)
	{
		size_t idx = batchset[i];
		size_t file_index = 0;
		for (; file_index < m_CIFAR10_files.size(); file_index++)
		{
			if (idx >= m_CIFAR10_files[file_index].start_img_idx &&
				idx < m_CIFAR10_files[file_index].start_img_idx + m_CIFAR10_files[file_index].img_count)
			{
				if (_fseeki64(m_CIFAR10_files[file_index].fp, (idx - m_CIFAR10_files[file_index].start_img_idx) * 3073, SEEK_SET) != 0)
					break;

				if (fread(buf, 1, 3073, m_CIFAR10_files[file_index].fp) != 3073)
					break;

				if (buf[0] > 9)
					printf("Unexpected CIFAR-10 label: %d\n", buf[0]);

				// get the label of the current image
				labels.push_back(buf[0]);

				for (int c = 0; c < 3; c++)
				{
					for (UINT h = 0; h < 32; h++)	// row
					{
						for (UINT w = 0; w < 32; w++)	// column
						{
							size_t cpos = pos + c * 1024 + h * 32 + w;
							res_data[cpos] = ((buf[c*1024 + h*32 + w + 1]) / 255.0f - m_RGB_means[c]) / m_RGB_stds[c];
						}
					}
				}

				pos += 3072;
				break;
			}
		}
	}

	tensor = torch::from_blob(res_data, { (long long)labels.size(), 3, 32, 32 }, FreeBlob);

	return S_OK;
}

HRESULT ImageProcess::ToCIFAR100Tensor(std::vector<size_t>& batchset, torch::Tensor& tensor, std::vector<int64_t>& labels, std::vector<int64_t>& coarse_labels)
{
	if (batchset.size() == 0)
		return E_FAIL;

	if (m_fpCIFAR100Bin == NULL)
	{
		printf("No file pointer to read the CIFAR100 images.\n");
		return E_FAIL;
	}

	uint8_t buf[3074];
	size_t pos = 0;
	float* res_data = new float[batchset.size() * 3 * 32 * 32];

	for (size_t i = 0; i < batchset.size(); i++)
	{
		size_t idx = batchset[i];
		if (_fseeki64(m_fpCIFAR100Bin, idx * 3074, SEEK_SET) != 0)
			break;

		if (fread(buf, 1, 3074, m_fpCIFAR100Bin) != 3074)
			break;

		if (buf[0] >= m_image_coarse_labels.size())
			printf("Unexpected CIFAR-100 coarse label: %d\n", buf[0]);

		if (buf[1] >= m_image_labels.size())
			printf("Unexpected CIFAR-100 label: %d\n", buf[1]);

		// get the coarse and fine label of the current image
		coarse_labels.push_back(buf[0]);
		labels.push_back(buf[1]);

		for (int c = 0; c < 3; c++)
		{
			for (UINT h = 0; h < 32; h++)	// row
			{
				for (UINT w = 0; w < 32; w++)	// column
				{
					size_t cpos = pos + c * 1024 + h * 32 + w;
					res_data[cpos] = ((buf[c * 1024 + h * 32 + w + 2]) / 255.0f - m_RGB_means[c]) / m_RGB_stds[c];
				}
			}
		}

		pos += 3072;
	}

	tensor = torch::from_blob(res_data, { (long long)labels.size(), 3, 32, 32 }, FreeBlob);

	return S_OK;
}

HRESULT ImageProcess::ToMNISTTensor(std::vector<size_t>& batchset, torch::Tensor& tensor, std::vector<int64_t>& labels)
{
	if (batchset.size() == 0)
		return E_FAIL;

	if (m_fpMNIST == NULL)
	{
		printf("No file pointer to read the MNIST images.\n");
		return E_FAIL;
	}

	uint8_t buf[784];
	size_t pos = 0;
	float* res_data = new float[batchset.size() * 28 * 28];

	for (size_t i = 0; i < batchset.size(); i++)
	{
		size_t idx = batchset[i];
		if (_fseeki64(m_fpMNIST, 16 + idx * 784, SEEK_SET) != 0)
			break;

		if (fread(buf, 1, 784, m_fpMNIST) != 784)
			break;

		labels.push_back(m_MNISTLabels[idx]);

		for (UINT i = 0; i < 28; i++)	// row
		{
			for (UINT j = 0; j < 28; j++)	// column
			{
				size_t cpos = pos + i * 28 + j;
				res_data[cpos] = ((255 - buf[784 + j * 28 + i]) / 255.0f - m_GreyScale_mean) / m_GreyScale_std;
			}
		}

		pos += 784;

	}

	tensor = torch::from_blob(res_data, { (long long)labels.size(), 1, 28, 28 }, FreeBlob);

	return S_OK;
}

HRESULT ImageProcess::ToImage(torch::Tensor& tensor, const TCHAR* szImageFilePrefix)
{
	HRESULT hr = E_FAIL;
	ComPtr<ID2D1Bitmap> spBitmap;
	D2D1_PIXEL_FORMAT pixelFormat = D2D1::PixelFormat(
		DXGI_FORMAT_B8G8R8A8_UNORM,
		D2D1_ALPHA_MODE_IGNORE
	);
	ComPtr<IWICBitmap> spWICBitmap;
	ComPtr<ID2D1RenderTarget> spRenderTarget;
	ComPtr<ID2D1SolidColorBrush> spBGBrush;

	TCHAR szImagePath[MAX_PATH] = { 0 };

	D2D1_BITMAP_PROPERTIES property;
	property.pixelFormat = pixelFormat;
	property.dpiX = 0;
	property.dpiY = 0;

	if (tensor.sizes().size() != 4)
		return E_FAIL;

	// Make sure there are RGB channels or gray channels
	if (tensor.size(1) != 3 && tensor.size(1) != 1)
	{
		printf("The input tensor need have RGB channels or gray channel.\n");
		return E_FAIL;
	}

	UINT32 width = (UINT32)tensor.size(2);
	UINT32 height = (UINT32)tensor.size(3);

	if (FAILED(hr = m_spWICImageFactory->CreateBitmap(width, height, GUID_WICPixelFormat32bppPBGRA,
		WICBitmapCacheOnDemand, &spWICBitmap)))
	{
		printf("Failed to create WICBitmap.\n");
		return E_FAIL;
	}

	// 在此WICBitmap上创建D2D1 Render Target
	D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT,
		D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED), 96, 96);
	if (FAILED(hr = m_spD2D1Factory->CreateWicBitmapRenderTarget(spWICBitmap.Get(), props, &spRenderTarget)))
	{
		printf("Failed to create WicBitmap Render Target {hr: 0X%X}.\n", hr);
		return E_FAIL;
	}

	if (FAILED(hr = spRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Red, 1.0f), &spBGBrush)))
	{
		printf("Failed to create a solid color brush {hr: 0X%X}.\n", hr);
		return E_FAIL;
	}

	D2D1_SIZE_U size_u;
	size_u.width = width;
	size_u.height = height;
	D2D1_RECT_F dst_rect = { 0.f, 0.f, (FLOAT)width, (FLOAT)height };

	float* ptrData = (float*)tensor.data_ptr();
	uint8_t* pBitmapData = new uint8_t[(size_t)width*height * 4];
	for (int64_t i = 0; i < tensor.size(0); i++)	// batch
	{
		if (tensor.size(1) == 3)
		{
			float* pFR = ptrData + i * 3 * width*height;
			float* pFG = ptrData + (i * 3 + 1) * width*height;
			float* pFB = ptrData + (i * 3 + 2) * width*height;
			for (UINT32 h = 0; h < height; h++)
			{
				for (UINT32 w = 0; w < width; w++)
				{
					size_t pos = (h * width + w) << 2;
					pBitmapData[pos + 0] = (*(pFB++)*m_RGB_stds[2] + m_RGB_means[2]) * 255;	// B
					pBitmapData[pos + 1] = (*(pFG++)*m_RGB_stds[1] + m_RGB_means[1]) * 255; // G
					pBitmapData[pos + 2] = (*(pFR++)*m_RGB_stds[0] + m_RGB_means[0]) * 255; // R
					pBitmapData[pos + 3] = 0xFF; // A
				}
			}

			if (FAILED(spRenderTarget->CreateBitmap(size_u, pBitmapData, width*4, property, &spBitmap)))
			{
				printf("Failed to create the D2D1 bitmap.\n");
				continue;
			}

			spRenderTarget->BeginDraw();
			spRenderTarget->FillRectangle(dst_rect, spBGBrush.Get());
			spRenderTarget->DrawBitmap(spBitmap.Get(), dst_rect);
			spRenderTarget->EndDraw();

			_stprintf_s(szImagePath, MAX_PATH, _T("%s_%lld.png"), szImageFilePrefix, i);
			SaveAs(spWICBitmap, szImagePath);
		}
		else if (tensor.size(1) == 1)
		{

		}
	}

	delete[] pBitmapData;
	return S_OK;
}

void ImageProcess::SaveAs(ComPtr<IWICBitmap>& bitmap, PCWSTR filename)
{
	HRESULT hr = S_OK;
	GUID guid = GUID_ContainerFormatPng;
	ComPtr<IWICImagingFactory> spWICImageFactory;

	PCWSTR cwszExt = wcsrchr(filename, L'.');
	if (cwszExt != NULL)
	{
		if (_wcsicmp(cwszExt, L".png") == 0)
			guid = GUID_ContainerFormatPng;
		else if (_wcsicmp(cwszExt, L".jpg") == 0 || _wcsicmp(cwszExt, L".jpeg") == 0 || _wcsicmp(cwszExt, L".jpg+") == 0)
			guid = GUID_ContainerFormatJpeg;
	}

	ComPtr<IStream> file;
	GUID pixelFormat;
	ComPtr<IWICBitmapFrameEncode> frame;
	ComPtr<IPropertyBag2> properties;
	ComPtr<IWICBitmapEncoder> encoder;
	UINT width, height;

	hr = SHCreateStreamOnFileEx(filename,
		STGM_CREATE | STGM_WRITE | STGM_SHARE_EXCLUSIVE,
		FILE_ATTRIBUTE_NORMAL,
		TRUE, // create
		nullptr, // template
		file.GetAddressOf());
	if (FAILED(hr))
		goto done;

	if (FAILED(CoCreateInstance(CLSID_WICImagingFactory,
		nullptr,
		CLSCTX_INPROC_SERVER,
		IID_IWICImagingFactory,
		(LPVOID*)&spWICImageFactory)))
		goto done;

	hr = spWICImageFactory->CreateEncoder(guid,
		nullptr, // vendor
		encoder.GetAddressOf());
	if (FAILED(hr))
		goto done;

	hr = encoder->Initialize(file.Get(), WICBitmapEncoderNoCache);
	if (FAILED(hr))
		goto done;

	hr = encoder->CreateNewFrame(frame.GetAddressOf(), properties.GetAddressOf());
	if (FAILED(hr))
		goto done;

	if (FAILED(hr = frame->Initialize(properties.Get())))
		goto done;

	if (FAILED(hr = bitmap->GetSize(&width, &height)))
		goto done;

	if (FAILED(hr = frame->SetSize(width, height)))
		goto done;

	if (FAILED(hr = bitmap->GetPixelFormat(&pixelFormat)))
		goto done;

	{
		auto negotiated = pixelFormat;
		if (FAILED(hr = frame->SetPixelFormat(&negotiated)))
			goto done;
	}

	if (FAILED(hr = frame->WriteSource(bitmap.Get(), nullptr)))
		goto done;

	if (FAILED(hr = frame->Commit()))
		goto done;

	if (FAILED(hr = encoder->Commit()))
		goto done;

done:
	return;
}

HRESULT ImageProcess::loadImageSet(
	const TCHAR* szRootPath,				// the root path to place training_set or test_set folder
	std::vector<tstring>& image_files,		// the image files to be trained or tested
	std::vector<tstring>& image_labels,		// the image label
	bool bTrainSet, bool bShuffle)
{
	HRESULT hr = S_OK;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szImageFile[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szRootPath);
	size_t ccDirPath = _tcslen(szRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szImageFile, MAX_PATH, _T("%s\\%s\\*.*"),
		szDirPath, bTrainSet ? _T("training_set") : _T("test_set"));

	// Find all image file names under the train set, 2 level
	WIN32_FIND_DATA find_data;
	HANDLE hFind = FindFirstFile(szImageFile, &find_data);
	if (hFind == INVALID_HANDLE_VALUE)
		return E_FAIL;

	do {
		if (!(find_data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) ||
			_tcsicmp(find_data.cFileName, _T(".")) == 0 ||
			_tcsicmp(find_data.cFileName, _T("..")) == 0)
			continue;

		WIN32_FIND_DATA image_find_data;
		_stprintf_s(szImageFile, MAX_PATH, _T("%s\\%s\\%s\\*.*"),
			szDirPath, bTrainSet ? _T("training_set") : _T("test_set"), find_data.cFileName);

		BOOL bHaveTrainImages = FALSE;
		HANDLE hImgFind = FindFirstFile(szImageFile, &image_find_data);
		if (hImgFind == INVALID_HANDLE_VALUE)
			continue;

		do
		{
			if (image_find_data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)
				continue;

			// check whether it is a supported image file
			const TCHAR* szTmp = _tcsrchr(image_find_data.cFileName, _T('.'));
			if (szTmp && (_tcsicmp(szTmp, _T(".jpg")) == 0 ||
				_tcsicmp(szTmp, _T(".png")) == 0 ||
				_tcsicmp(szTmp, _T(".jpeg")) == 0))
			{
				// reuse szImageFile
				_stprintf_s(szImageFile, _T("%s\\%s"), find_data.cFileName, image_find_data.cFileName);
				image_files.emplace_back(szImageFile);
				if (bHaveTrainImages == FALSE)
				{
					bHaveTrainImages = TRUE;
					image_labels.emplace_back(find_data.cFileName);
				}
			}

		} while (FindNextFile(hImgFind, &image_find_data));

		FindClose(hImgFind);

	} while (FindNextFile(hFind, &find_data));

	FindClose(hFind);

	return hr;
}

HRESULT ImageProcess::loadLabels(const TCHAR* szImageSetRootPath, std::vector<tstring>& image_labels)
{
	HRESULT hr = S_OK;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szImageFile[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szImageFile, MAX_PATH, _T("%s\\training_set\\*.*"), szDirPath);

	// Find all image file names under the train set, 2 level
	WIN32_FIND_DATA find_data;
	HANDLE hFind = FindFirstFile(szImageFile, &find_data);
	if (hFind == INVALID_HANDLE_VALUE)
		return E_FAIL;

	do {
		if (!(find_data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY))
			continue;

		WIN32_FIND_DATA image_find_data;
		_stprintf_s(szImageFile, MAX_PATH, _T("%s\\training_set\\%s\\*.*"), szDirPath, find_data.cFileName);

		BOOL bHaveTrainImages = FALSE;
		HANDLE hImgFind = FindFirstFile(szImageFile, &image_find_data);
		if (hImgFind == INVALID_HANDLE_VALUE)
			continue;

		do
		{
			if (image_find_data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)
				continue;

			// check whether it is a supported image file
			const TCHAR* szTmp = _tcsrchr(image_find_data.cFileName, _T('.'));
			if (szTmp && (_tcsicmp(szTmp, _T(".jpg")) == 0 ||
				_tcsicmp(szTmp, _T(".png")) == 0 ||
				_tcsicmp(szTmp, _T(".jpeg")) == 0))
			{
				bHaveTrainImages = TRUE;
				break;
			}

		} while (FindNextFile(hImgFind, &image_find_data));

		if (bHaveTrainImages)
			image_labels.emplace_back(find_data.cFileName);

		FindClose(hImgFind);

	} while (FindNextFile(hFind, &find_data));

	FindClose(hFind);

	return S_OK;
}

HRESULT ImageProcess::loadLabelsFromFile(const TCHAR* szLabelFilePath, std::vector<tstring>& labels)
{
	FILE* fp = NULL;
	if (_tfopen_s(&fp, szLabelFilePath, _T("rb")) != 0)
	{
		_tprintf(_T("Failed to open the file '%s'.\n"), szLabelFilePath);
		return E_FAIL;
	}

	long file_size = -1;
	char szBuf[2048] = { 0 };
	int cbRead = (int)fread(szBuf, 1, sizeof(szBuf) - 1, fp);
	if (cbRead >= sizeof(szBuf))
	{
		fclose(fp);
		return E_FAIL;
	}

	// convert the read content to TCHAR
	TCHAR* tcsBuf = NULL;
	int ccRead = cbRead;
#ifdef _UNICODE
	wchar_t wcsBuf[2048] = { 0 };
	if ((ccRead = MultiByteToWideChar(CP_UTF8, 0, szBuf, -1, wcsBuf, 2048)) == 0)
	{
		_tprintf(_T("Hit the abnormal in the file '%s'.\n"), szLabelFilePath);
		return E_FAIL;
	}
	tcsBuf = wcsBuf;
#else
	tcsBuf = szBuf;
#endif

	TCHAR tszLabel[1024] = { 0 };
	int read_pos = 0, write_pos = 0;
	while (read_pos < ccRead)
	{
		if (write_pos == 0)
		{
			// skip the white-space at the beginning of label name
			if (tcsBuf[read_pos] == _T('\t') || tcsBuf[read_pos] == _T(' ') ||
				tcsBuf[read_pos] == _T('\r') || tcsBuf[read_pos] == _T('\n') || tcsBuf[read_pos] == _T('\0'))
			{
				read_pos++;
				continue;
			}
		}
		if (tcsBuf[read_pos] == _T('\r') || tcsBuf[read_pos] == _T('\n') || tcsBuf[read_pos] == _T('\0'))
		{
			if (write_pos < sizeof(tszLabel) / sizeof(TCHAR))
				tszLabel[write_pos] = _T('\0');
			else
				tszLabel[sizeof(tszLabel) / sizeof(TCHAR) - 1] = _T('\0');
			labels.push_back(tszLabel);
			write_pos = 0;
			memset(tszLabel, 0, sizeof(tszLabel));
		}
		else if (write_pos < sizeof(tszLabel) / sizeof(TCHAR))
			tszLabel[write_pos++] = tcsBuf[read_pos];

		read_pos++;
	}

	if (write_pos > 0)
		labels.push_back(tszLabel);

	fclose(fp);
	return S_OK;
}

HRESULT ImageProcess::loadCIFAR10Labels(const TCHAR* szImageSetRootPath, std::vector<tstring>& image_labels)
{
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szMetaFile[MAX_PATH] = { 0 };
	TCHAR szBatchFilePath[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szMetaFile, MAX_PATH, _T("%s\\batches.meta.txt"), szDirPath);
	return loadLabelsFromFile(szMetaFile, image_labels);
}

HRESULT ImageProcess::loadCIFAR100Labels(const TCHAR* szImageSetRootPath, std::vector<tstring>& image_labels, std::vector<tstring>& image_coarse_labels)
{
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szMetaFile[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szMetaFile, MAX_PATH, _T("%s\\fine_label_names.txt"), szDirPath);
	if (FAILED(loadLabelsFromFile(szMetaFile, image_labels)))
		return E_FAIL;

	_stprintf_s(szMetaFile, MAX_PATH, _T("%s\\coarse_label_names.txt"), szDirPath);
	loadLabelsFromFile(szMetaFile, image_coarse_labels);

	return S_OK;
}

HRESULT ImageProcess::loadMNISTImageSet(const TCHAR* szImageSetRootPath, MNIST_INFO& MNIST_info, bool bTrainSet)
{
	HRESULT hr = E_FAIL;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szMNISTFile[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szMNISTFile, MAX_PATH, _T("%s\\%s"), szDirPath, bTrainSet?_T("train-labels-idx1-ubyte"):_T("t10k-labels-idx1-ubyte"));
	FILE* fp = NULL;
	if (_tfopen_s(&fp, szMNISTFile, _T("rb")) != 0)
	{
		_tprintf(_T("Failed to open the MNIST file '%s'\n"), szMNISTFile);
		return E_INVALIDARG;
	}

	uint32_t magic_number = 0;
	if (fread(&magic_number, 1, 4, fp) != 4)
		goto done;

	ULONG_FIELD_ENDIAN(magic_number);
	if (magic_number != 0x00000801)
	{
		_tprintf(_T("An invalid label idx file '%s'\n"), szMNISTFile);
		goto done;
	}

	if (fread(&MNIST_info.num_of_image_labels, 1, 4, fp) != 4)
		goto done;

	ULONG_FIELD_ENDIAN(MNIST_info.num_of_image_labels);

	m_MNISTLabels.reserve(MNIST_info.num_of_image_labels);

	uint8_t* pLabelData = new uint8_t[MNIST_info.num_of_image_labels];
	size_t clRead = fread(pLabelData, 1, MNIST_info.num_of_image_labels, fp);
	for (size_t i = 0; i < clRead; i++)
		m_MNISTLabels.push_back(pLabelData[i]);
	delete[] pLabelData;

	fclose(fp);
	fp = NULL;

	_stprintf_s(szMNISTFile, MAX_PATH, _T("%s\\%s"), szDirPath, bTrainSet ? _T("train-images-idx3-ubyte") : _T("t10k-images-idx3-ubyte"));
	if (_tfopen_s(&fp, szMNISTFile, _T("rb")) != 0)
	{
		_tprintf(_T("Failed to open the MNIST file '%s'\n"), szMNISTFile);
		hr = E_INVALIDARG;
		goto done;
	}

	if (fread(&magic_number, 1, 4, fp) != 4)
		goto done;

	ULONG_FIELD_ENDIAN(magic_number);
	if (magic_number != 0x00000803)
	{
		_tprintf(_T("An invalid image-set file '%s'\n"), szMNISTFile);
		goto done;
	}

	if (fread(&MNIST_info.num_of_images, 1, 4, fp) != 4)
		goto done;
	ULONG_FIELD_ENDIAN(MNIST_info.num_of_images);

	if (fread(&MNIST_info.image_height, 1, 4, fp) != 4)
		goto done;
	ULONG_FIELD_ENDIAN(MNIST_info.image_height);

	if (fread(&MNIST_info.image_width, 1, 4, fp) != 4)
		goto done;
	ULONG_FIELD_ENDIAN(MNIST_info.image_width);

	if (MNIST_info.num_of_images != MNIST_info.num_of_image_labels)
	{
		printf("The size of images set and labels set is different {%lu != %lu}.\n", 
			MNIST_info.num_of_images, MNIST_info.num_of_image_labels);
		goto done;
	}

	if (m_fpMNIST != NULL)
	{
		fclose(m_fpMNIST);
		m_fpMNIST = NULL;
	}

	m_fpMNIST = fp;
	fp = NULL;

	hr = S_OK;

done:
	if (fp != NULL)
		fclose(fp);
	return hr;
}

struct CIFAR10FileCompare
{
	bool operator()(const WIN32_FIND_DATA& a, const WIN32_FIND_DATA& b)
	{
		int a_file_index = -1, b_file_index = -1;
		int64_t file_index[2] = { -1, -1 };

		for (int i = 0; i < 2; i++)
		{
			const TCHAR* pTmp = _tcsrchr(i==0?a.cFileName:b.cFileName, _T('_'));
			const TCHAR* pTmp2 = NULL;

			if (pTmp != NULL)
			{
				pTmp2 = _tcschr(pTmp, _T('.'));
				if (pTmp2 == NULL)
					pTmp2 = a.cFileName + _tcslen(a.cFileName);


#ifdef _UNICODE
				if (ConvertToIntW((wchar_t*)pTmp, (wchar_t*)pTmp2, file_index[i]))
#else
				if (ConvertToInt((wchar_t*)pTmp, (wchar_t*)pTmp2, file_index[i]))
#endif
				{
					if (file_index[i] >= 0 && file_index[i] <= INT32_MAX)
						if (i == 0)
							a_file_index = file_index[i];
						else
							b_file_index = file_index[i];
				}
			}
		}

		return a_file_index < b_file_index;
	}
};

HRESULT ImageProcess::loadCIFAR10ImageSet(const TCHAR* szImageSetRootPath, int& num_of_imgs, bool bTrainSet)
{
	HRESULT hr = E_FAIL;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szBatchFilePath[MAX_PATH] = { 0 };
	unsigned long long ullTotalFileSize = 0;

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szBatchFilePath, MAX_PATH, _T("%s\\%s*"), szDirPath, bTrainSet ? _T("data_batch") : _T("test_batch"));

	for (size_t i = 0; i < m_CIFAR10_files.size(); i++)
	{
		if (m_CIFAR10_files[i].fp != NULL)
		{
			fclose(m_CIFAR10_files[i].fp);
			m_CIFAR10_files[i].fp = NULL;
		}
		m_CIFAR10_files[i].start_img_idx = 0;
		m_CIFAR10_files[i].img_count = 0;
	}
	m_CIFAR10_files.clear();

	WIN32_FIND_DATA find_data;
	std::vector<WIN32_FIND_DATA> win32_find_datas;
	HANDLE hFind = FindFirstFile(szBatchFilePath, &find_data);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do {
			ullTotalFileSize += ((unsigned long long)find_data.nFileSizeHigh << 32) | find_data.nFileSizeLow;
			win32_find_datas.push_back(find_data);
		} while (FindNextFile(hFind, &find_data));
	}

	num_of_imgs = ullTotalFileSize / 3073;
	std::sort(win32_find_datas.begin(), win32_find_datas.end(), CIFAR10FileCompare());

	size_t start_img_index = 0;
	for (size_t i = 0; i < win32_find_datas.size(); i++)
	{
		CIFAR10_FILE cifar10_file;
		cifar10_file.start_img_idx = start_img_index;
		cifar10_file.img_count = (((unsigned long long)win32_find_datas[i].nFileSizeHigh << 32) | find_data.nFileSizeLow) / 3073;

		_stprintf_s(szBatchFilePath, MAX_PATH, _T("%s\\%s"), szDirPath, win32_find_datas[i].cFileName);
		_tfopen_s(&cifar10_file.fp, szBatchFilePath, _T("rb"));

		m_CIFAR10_files.push_back(cifar10_file);

		start_img_index += cifar10_file.img_count;
	}

	return S_OK;
}

HRESULT ImageProcess::loadCIFAR100ImageSet(const TCHAR* szImageSetRootPath, int& num_of_imgs, bool bTrainSet)
{
	HRESULT hr = E_FAIL;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szBatchFilePath[MAX_PATH] = { 0 };
	unsigned long long ullTotalFileSize = 0;

	_tcscpy_s(szDirPath, MAX_PATH, szImageSetRootPath);
	size_t ccDirPath = _tcslen(szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	_stprintf_s(szBatchFilePath, MAX_PATH, _T("%s\\%s"), szDirPath, bTrainSet ? _T("train.bin") : _T("test.bin"));

	if (m_fpCIFAR100Bin != NULL)
	{
		fclose(m_fpCIFAR100Bin);
		m_fpCIFAR100Bin = NULL;
	}

	if (_tfopen_s(&m_fpCIFAR100Bin, szBatchFilePath, _T("rb")) != 0)
	{
		_tprintf(_T("Failed to open the CIFAR100 file '%s'.\n"), szBatchFilePath);
		return E_FAIL;
	}

	if (_fseeki64(m_fpCIFAR100Bin, 0, SEEK_END) != 0)
		goto done;

	long long file_size = _ftelli64(m_fpCIFAR100Bin);
	num_of_imgs = (int)(file_size / 3074);

	_fseeki64(m_fpCIFAR100Bin, 0, SEEK_SET);

	hr = S_OK;

done:
	return hr;
}

HRESULT ImageProcess::loadImageSet(
	IMGSET_TYPE imgset_type,
	const TCHAR* szImageSetRootPath,
	std::vector<tstring>& image_labels,
	std::vector<tstring>& image_coarse_labels,
	int& number_of_imgs,
	int batch_size,
	bool bTrainSet,
	bool bShuffle)
{
	// Check the parameters
	if (imgset_type < IMGSET_FOLDER || imgset_type >= IMGSET_MAX)
	{
		printf("Need specify a valid image set type {%d).\n", imgset_type);
		return E_INVALIDARG;
	}

	if (szImageSetRootPath == NULL || _taccess(szImageSetRootPath, 0) != 0)
	{
		_tprintf(_T("the file path '%s' does not exist.\n"), szImageSetRootPath);
		return E_INVALIDARG;
	}

	if (imgset_type == IMGSET_FOLDER)
	{
		image_labels.clear();
		m_image_files.clear();
		if (FAILED(loadImageSet(szImageSetRootPath, m_image_files, image_labels, bTrainSet, bShuffle)))
		{
			_tprintf(_T("Failed to load the image labels from the path '%s'.\n"), szImageSetRootPath);
			return E_FAIL;
		}

		number_of_imgs = (int)m_image_files.size();
		m_image_labels = image_labels;
	}
	else if (imgset_type == IMGSET_MNIST)
	{
		MNIST_INFO MNIST_info;
		for (int i = 0; i <= 9; i++)
#ifdef _UNICODE
			image_labels.push_back(std::to_wstring(i));
#else
			image_labels.push_back(std::to_string(i));
#endif

		if (FAILED(loadMNISTImageSet(szImageSetRootPath, MNIST_info, bTrainSet)))
		{
			_tprintf(_T("Unsupported MNIST dataset in the path '%s'.\n"), szImageSetRootPath);
			return E_FAIL;
		}

		number_of_imgs = MNIST_info.num_of_images;
		m_image_labels = image_labels;
	}
	else if (imgset_type == IMGSET_CIFAR_10)
	{
		image_labels.clear();
		if (FAILED(loadCIFAR10Labels(szImageSetRootPath, image_labels)))
		{
			_tprintf(_T("Failed to load CIFAR-10 labels.\n"));
			return E_FAIL;
		}

		if (FAILED(loadCIFAR10ImageSet(szImageSetRootPath, number_of_imgs, bTrainSet)))
		{
			_tprintf(_T("Failed to load the CIFAR10 %s image set.\n"), bTrainSet?_T("train"):_T("test"));
			return E_FAIL;
		}

		m_image_labels = image_labels;
	}
	else if (imgset_type == IMGSET_CIFAR_100)
	{
		image_labels.clear();
		image_coarse_labels.clear();
		if (FAILED(loadCIFAR100Labels(szImageSetRootPath, image_labels, image_coarse_labels)))
		{
			_tprintf(_T("Failed to load CIFAR-100 labels.\n"));
			return E_FAIL;
		}

		if (FAILED(loadCIFAR100ImageSet(szImageSetRootPath, number_of_imgs,  bTrainSet)))
		{
			_tprintf(_T("Failed to load the CIFAR100 %s image set.\n"), bTrainSet ? _T("train") : _T("test"));
			return E_FAIL;
		}

		m_image_coarse_labels = image_coarse_labels;
		m_image_labels = image_labels;
	}
	else
		return E_NOTIMPL;

	m_imgset_type = imgset_type;
	_tcscpy_s(m_szImageSetRootPath, MAX_PATH, szImageSetRootPath);
	m_batch_size = batch_size;
	m_bTrainSet = bTrainSet;
	m_bShuffle = true;
	m_number_of_items = number_of_imgs;

	return S_OK;
}

HRESULT ImageProcess::initImageSetBatchIter()
{
	// generate the shuffle list to train
	m_image_shuffle_set.resize(m_number_of_items);
	for (size_t i = 0; i < m_number_of_items; i++)
		m_image_shuffle_set[i] = i;
	std::random_device rd;
	std::mt19937_64 g(rd());
	std::shuffle(m_image_shuffle_set.begin(), m_image_shuffle_set.end(), g);

	m_imageset_iter_pos = -1;

	return S_OK;
}

HRESULT ImageProcess::nextImageSetBatchIter(torch::Tensor& tensor, std::vector<int64_t>& labels, std::vector<int64_t>& coarse_labels)
{
	HRESULT hr = E_FAIL;
	TCHAR szDirPath[MAX_PATH] = { 0 };
	TCHAR szImageFile[MAX_PATH] = { 0 };

	_tcscpy_s(szDirPath, MAX_PATH, m_szImageSetRootPath);
	size_t ccDirPath = _tcslen(m_szImageSetRootPath);
	if (szDirPath[ccDirPath - 1] == _T('\\'))
		szDirPath[ccDirPath - 1] = _T('\0');

	if (m_imageset_iter_pos + 1 >= m_number_of_items)
		return E_FAIL;

	int batch_size = m_batch_size;
	if (batch_size > m_number_of_items - m_imageset_iter_pos - 1)
		batch_size = (int)(m_number_of_items - m_imageset_iter_pos - 1);

	if (m_imgset_type == IMGSET_FOLDER)
	{
		std::vector<tstring> image_batches;

		m_imageset_iter_pos++;
		labels.clear();
		coarse_labels.clear();

		while (image_batches.size() < batch_size)
		{
			size_t idx = ++m_imageset_iter_pos;
			if (idx >= m_image_shuffle_set.size())
				break;

			tstring& strImgFilePath = m_image_files[m_image_shuffle_set[idx]];
			const TCHAR* cszImgFilePath = strImgFilePath.c_str();
			const TCHAR* pszTmp = _tcschr(cszImgFilePath, _T('\\'));

			if (pszTmp == NULL)
			{
				_tprintf(_T("unexpected file path: %s {%s(): %d}\n"), cszImgFilePath, _T(__FUNCTION__), __LINE__);
				continue;
			}

			size_t label = 0;
			for (label = 0; label < m_image_labels.size(); label++)
				if (_tcsnicmp(m_image_labels[label].c_str(), cszImgFilePath,
					(pszTmp - cszImgFilePath) / sizeof(TCHAR)) == 0)
					break;

			if (label >= m_image_labels.size())
			{
				_tprintf(_T("can't find the label in unexpected file path: %s {%s(): %d}\n"), cszImgFilePath, _T(__FUNCTION__), __LINE__);
				continue;
			}

			_stprintf_s(szImageFile, _T("%s\\training_set\\%s"), szDirPath, cszImgFilePath);

			image_batches.push_back(szImageFile);
			labels.push_back((int64_t)label);
		}

		if (image_batches.size() == 0)
			goto done;

		hr = ToTensor(image_batches, tensor);
	}
	else if (m_imgset_type == IMGSET_MNIST)
	{
		std::vector<size_t> MNIST_imgset;
		for (int b = 0; b < m_batch_size; b++)
		{
			if (m_imageset_iter_pos + 1 >= m_image_shuffle_set.size())
				break;

			MNIST_imgset.push_back(m_image_shuffle_set[++m_imageset_iter_pos]);
		}

		labels.clear();
		hr = ToMNISTTensor(MNIST_imgset, tensor, labels);
	}
	else if (m_imgset_type == IMGSET_CIFAR_10)
	{
		std::vector<size_t> CIFAR10_imgset;
		for (int b = 0; b < m_batch_size; b++)
		{
			if (m_imageset_iter_pos + 1 >= m_image_shuffle_set.size())
				break;

			CIFAR10_imgset.push_back(m_image_shuffle_set[++m_imageset_iter_pos]);
		}

		labels.clear();
		hr = ToCIFAR10Tensor(CIFAR10_imgset, tensor, labels);
	}
	else if (m_imgset_type == IMGSET_CIFAR_100)
	{
		std::vector<size_t> CIFAR100_imgset;
		for (int b = 0; b < m_batch_size; b++)
		{
			if (m_imageset_iter_pos + 1 >= m_image_shuffle_set.size())
				break;

			CIFAR100_imgset.push_back(m_image_shuffle_set[++m_imageset_iter_pos]);
		}

		labels.clear();
		coarse_labels.clear();
		hr = ToCIFAR100Tensor(CIFAR100_imgset, tensor, labels, coarse_labels);
	}
	else
		hr = E_NOTIMPL;

done:
	return hr;
}

HRESULT ImageProcess::unloadImageSet()
{
	m_imgset_type = IMGSET_UNKNOWN;
	m_image_files.clear();
	m_image_labels.clear();
	m_image_shuffle_set.clear();
	for (size_t i = 0; i < m_CIFAR10_files.size(); i++)
	{
		if (m_CIFAR10_files[i].fp != NULL)
		{
			fclose(m_CIFAR10_files[i].fp);
			m_CIFAR10_files[i].fp = NULL;
		}
		m_CIFAR10_files[i].start_img_idx = 0;
		m_CIFAR10_files[i].img_count = 0;
	}
	m_CIFAR10_files.clear();

	if (m_fpCIFAR100Bin)
	{
		fclose(m_fpCIFAR100Bin);
		m_fpCIFAR100Bin = NULL;
	}

	return S_OK;
}
