#include "BaseNNet.h"
#include "util.h"
#include <assert.h>
#include <memory>
#include <ostream>
#include <Windows.h>
#include "resource.h"

// Load from Win32 resource file
bool GetNNConfig(std::string& nnconfig_xml)
{
	FILE* fp = NULL;
	bool bRet = false;
	errno_t err = fopen_s(&fp, "nnconfig.xml", "rb");

	if (err == 0 && fp != NULL)
	{
		fseek(fp, 0, SEEK_END);
		long file_size = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		if (file_size > 0)
		{
			char* pBuf = new char[file_size + 1];
			memset(pBuf, 0, file_size + 1);
			if (pBuf != NULL && fread(pBuf, 1, file_size, fp) == file_size)
			{
				nnconfig_xml.assign(pBuf, file_size);
				fclose(fp);
				return true;
			}
		}
	}

	if (fp != NULL)
		fclose(fp);

	HRSRC hRes = FindResource(NULL, MAKEINTRESOURCE(IDR_NNCONFIG), TEXT("XML"));
	HGLOBAL hGlobalRes = LoadResource(NULL, hRes);
	if (hGlobalRes == NULL)
		return false;

	LPVOID lp = LockResource(hGlobalRes);
	DWORD size = SizeofResource(NULL, hRes);

	if (size > 0)
	{
		nnconfig_xml.assign((char*)lp, size);
		bRet = true;
	}

	FreeResource(hGlobalRes);
	return bRet;
}


int BaseNNet::SetOptions(NNetOptions options)
{
	nn_options = options;
	return 0;
}

NN_TYPE BaseNNet::nntype_from_options()
{
	NN_TYPE nn_type = NN_TYPE_UNKNOWN;
	std::string* ptr_str_nn_type = nn_options.find("NN::nn_type");
	if (ptr_str_nn_type != NULL)
	{
		int64_t val_nn_type = -1;
		if (ConvertToInt((char*)ptr_str_nn_type->c_str(),
			(char*)ptr_str_nn_type->c_str() + ptr_str_nn_type->length(), val_nn_type))
			nn_type = (NN_TYPE)val_nn_type;
	}

	return nn_type;
}

int64_t	BaseNNet::int64_from_options(const char* key, int64_t def)
{
	int64_t val = def;
	std::string* ptr_str_numclass = nn_options.find(key);
	if (ptr_str_numclass != NULL)
	{
		ConvertToInt((char*)ptr_str_numclass->c_str(),
			(char*)ptr_str_numclass->c_str() + ptr_str_numclass->length(), val);
	}

	return val;
}

int BaseNNet::Init(const char* szNNName)
{
	int iRet = 0;
	std::string strXmlNNConfig;
	if (GetNNConfig(strXmlNNConfig) == false)
	{
		printf("Failed to load Neutral Network Configuration.\n");
		return -1;
	}

	if (xmlDoc.Parse(strXmlNNConfig.c_str()) != tinyxml2::XML_SUCCESS)
	{
		printf("Failed to load 'nnconfig.xml'.\n");
		return -1;
	}

	if (xmlDoc.RootElement() == NULL || xmlDoc.RootElement()->NoChildren() || XP_STRICMP(xmlDoc.RootElement()->Name(), "nns") != 0)
	{
		printf("It is an invalid 'nnconfig.xml' file.\n");
		return -1;
	}

	auto child = xmlDoc.RootElement()->FirstChildElement("nn");
	while (child != NULL)
	{
		if (XP_STRICMP(child->Attribute("name"), szNNName) == 0)
			break;
		child = child->NextSiblingElement("nn");
	}

	if (child == NULL)
	{
		printf("Failed to find the neutral network '%s'.\n", szNNName);
		return -1;
	}

	auto nnCategary = child->Attribute("cat");
	if (nnCategary != NULL)
		nn_cat = nnCategary;

	// Load the modules one by one
	auto Elementmodules = child->FirstChildElement("modules");
	if (Elementmodules == NULL || Elementmodules->NoChildren())
	{
		printf("No modules to be loaded.\n");
		return -1;
	}

	auto ElementModule = Elementmodules->FirstChildElement("module");
	while (ElementModule != NULL)
	{
		if (LoadModule(ElementModule) != 0)
		{
			iRet = -1;
			printf("Failed to load the module '%s'.\n", ElementModule->Attribute("name", "(null)"));
			goto done;
		}
		ElementModule = ElementModule->NextSiblingElement("module");
	}

	// load the forward list
	flow_start = child->FirstChildElement("forward");

	nn_model_name = szNNName;

	iRet = 0;

done:
	return iRet;
}

int BaseNNet::Uninit()
{
	for (auto& iter = nn_modules.begin(); iter != nn_modules.end(); iter++)
	{
		unregister_module(iter->first);
		iter->second = nullptr;
	}

	nn_modules.clear();

	return 0;
}

void BaseNNet::Print()
{
	std::cout << "Neutral Network '" << nn_model_name << "' layout:\n";

	std::cout << *this << '\n';

	// print the total numbers of parameters and weight layers
	int64_t total_parameters = 0, weight_layers = 0;
	auto netparams = parameters();
	for (size_t i = 0; i < netparams.size(); i++)
	{
		int64_t param_item_count = netparams[i].sizes().size() > 0 ? 1 : 0;
		for (size_t j = 0; j < netparams[i].sizes().size(); j++)
			param_item_count *= netparams[i].size(j);
		total_parameters += param_item_count;
	}

	std::cout << "\nparameters layout:\n\ttotal parameters: " << total_parameters << ", weight layers: " << netparams.size() / 2 << '\n';

	// print each parameter layout
	for (auto const& p : named_parameters())
		std::cout << "\t" << p.key() << ":\n\t\t" << p.value().sizes() << '\n';

	return;
}

bool BaseNNet::_forward(tinyxml2::XMLElement* first_sibling, const torch::Tensor& input, torch::Tensor& out)
{
	namespace F = torch::nn::functional;

	// Tensor is not processed
	if (first_sibling == NULL)
		return false;

	bool bTensorHasProcessed = false;
	tinyxml2::XMLElement* sibling = first_sibling;

	do
	{
		if (XP_STRICMP(sibling->Name(), "parallel") == 0)
		{
			// backup the current input
			auto x = bTensorHasProcessed ? out : input;
			bool first_branch = true;

			tinyxml2::XMLElement* branch = sibling->FirstChildElement("branch");
			
			do
			{
				if (branch == NULL ||
					branch->NoChildren())
					continue;

				if (first_branch == true)
				{
					if (_forward(branch->FirstChildElement(), x, out))
					{
						//std::cout << out.sizes() << '\n';
						bTensorHasProcessed = true;
						first_branch = false;
					}
					else
					{
						printf("Failed to process the branch: parallel/branch, line number: %d.\n", branch->GetLineNum());
					}
				}
				else
				{
					if (XP_STRICMP(sibling->Attribute("mergeop"), "plus") == 0)
					{
						torch::Tensor tensor_ret;
						if (_forward(branch->FirstChildElement(), x, tensor_ret))
						{
							//std::cout << tensor_ret.sizes() << '\n';
							out += tensor_ret;
							if (branch->BoolAttribute("debug"))
							{
								std::cout << out << std::endl;
								abort();
							}
						}
						else
						{
							printf("Failed to process the branch: parallel/branch, line number: %d.\n", branch->GetLineNum());
						}
					}
					else
						printf("Unsupported mergeop '%s'\n", sibling->Attribute("mergeop"));
				}

			} while (branch = branch->NextSiblingElement("branch"));
		}
		else if (XP_STRICMP(sibling->Name(), "f") == 0)
		{
			auto f = sibling;
			const char* szModuleName = f->Attribute("module");
			if (szModuleName != NULL)
			{
				auto m = nn_modules.find(szModuleName);
				if (m != nn_modules.end())
				{
					std::string& module_type = nn_module_types[szModuleName];
					if (module_type == "conv2d")
					{
						auto spConv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m->second);
						out = spConv2d->forward(bTensorHasProcessed ? out : input);
						if (f->BoolAttribute("debug"))
						{
							std::cout << szModuleName << out << std::endl;
							abort();
						}
						bTensorHasProcessed = true;
						continue;
					}
					else if (module_type == "linear")
					{
						auto spLinear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m->second);
						out = spLinear->forward(bTensorHasProcessed ? out : input);
						if (f->BoolAttribute("debug"))
						{
							std::cout << szModuleName << out << std::endl;
							abort();
						}
						bTensorHasProcessed = true;
						continue;
					}
					else if (module_type == "batchnorm2d")
					{
						auto spBatchNorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m->second);
						out = spBatchNorm2d->forward(bTensorHasProcessed ? out : input);
						if (f->BoolAttribute("debug"))
						{
							std::cout << szModuleName << out << std::endl;
							abort();
						}
						bTensorHasProcessed = true;
						continue;
					}
					else if (module_type == "adaptiveavgpool2d")
					{
						auto spAdaptiveAvgPool2d = std::dynamic_pointer_cast<torch::nn::AdaptiveAvgPool2dImpl>(m->second);
						out = spAdaptiveAvgPool2d->forward(bTensorHasProcessed ? out : input);
						if (f->BoolAttribute("debug"))
						{
							std::cout << szModuleName << out << std::endl;
							abort();
						}
						//std::cout << out.sizes() << '\n';
						bTensorHasProcessed = true;
						continue;
					}
					else
						printf("Unsupported module type: '%s'.\n", module_type.c_str());
				}
				else
					printf("Failed to find the module '%s'.\n", szModuleName);
			}

			const char* szFunctional = f->Attribute("functional");
			if (szFunctional != NULL)
			{
				if (XP_STRICMP(szFunctional, "relu") == 0)
				{
					bool inplace = f->BoolAttribute("inplace", false);
					out = F::relu(bTensorHasProcessed ? out : input, F::ReLUFuncOptions(inplace));
					if (f->BoolAttribute("debug"))
					{
						std::cout << out << '\n';
						abort();
					}
					bTensorHasProcessed = true;
					continue;
				}
				else if (XP_STRICMP(szFunctional, "max_pool2d") == 0)
				{
					int64_t kernel_size = f->Int64Attribute("kernel_size", 2);
					int64_t stride = f->Int64Attribute("stride", kernel_size);
					int64_t padding = f->Int64Attribute("padding", 0);
					out = F::max_pool2d(bTensorHasProcessed ? out : input, F::MaxPool2dFuncOptions(kernel_size).stride(stride).padding(padding));
					if (f->BoolAttribute("debug"))
					{
						std::cout << out << '\n';
						abort();
					}
					bTensorHasProcessed = true;
					continue;
				}
				else if (XP_STRICMP(szFunctional, "dropout") == 0)
				{
					double p = f->DoubleAttribute("p", 0.5);
					bool inplace = f->BoolAttribute("inplace", false);
					out = F::dropout(bTensorHasProcessed ? out : input, F::DropoutFuncOptions().p(p).inplace(inplace));
					bTensorHasProcessed = true;
					continue;
				}
				else if (XP_STRICMP(szFunctional, "adaptive_avg_pool2d") == 0)
				{
					// At present, only support square kernel size
					int64_t outputsize = f->Int64Attribute("outputsize", -1LL);
					if (outputsize <= 0)
						printf("Invalid output size is specified for adaptive_avg_pool2d.\n");
					else
					{
						//std::cout << "avgpool input:\n" << out << '\n';
						out = F::adaptive_avg_pool2d(bTensorHasProcessed ? out : input, F::AdaptiveAvgPool2dFuncOptions({ outputsize, outputsize }));
						//std::cout << "avgpool output:\n" << out << '\n';
						bTensorHasProcessed = true;
					}

					continue;
				}
				else
					printf("Unsupported functional: '%s'\n", szFunctional);
			}

			const char* szView = f->Attribute("view");
			if (szView != NULL)
			{
				if (XP_STRICMP(szView, "flat") == 0)
				{
					if (bTensorHasProcessed)
						out = out.view({ out.size(0), -1 });
					else
						out = input.view({ input.size(0), -1 });
					//std::cout << out.sizes() << '\n';
					continue;
				}
			}
			else
				printf("Unsupported functional {line number: %d}.\n", f->GetLineNum());
		}
		else if (XP_STRICMP(sibling->Name(), "identity") == 0)
		{
			if (!bTensorHasProcessed)
			{
				out = input;
				bTensorHasProcessed = true;
			}
		}
		else
		{
			printf("Unsupported tag '%s' {line number: %d}\n", sibling->Name(), sibling->GetLineNum());
		}
	} while (sibling = sibling->NextSiblingElement());

	return bTensorHasProcessed;
}

torch::Tensor BaseNNet::forward(torch::Tensor& input)
{
	namespace F = torch::nn::functional;

	if (flow_start == NULL || flow_start->NoChildren() || flow_start->FirstChildElement("f") == NULL)
	{
		for (auto& m : modules(false))
		{
			if (m->name() == "torch::nn::Conv2dImpl")
			{
				auto spConv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m);
				input = spConv2d->forward(input);
				continue;
			}
			else if (m->name() == "torch::nn::BatchNorm2dImpl")
			{
				auto spBatchNorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m);
				input = spBatchNorm2d->forward(input);
				continue;
			}
			else if (m->name() == "torch::nn::LinearImpl")
			{
				if (input.sizes().size() != 2)
					input = input.view({ input.size(0), -1 });

				auto spLinear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m);
				input = spLinear->forward(input);
				continue;
			}
			else if (m->name() == "torch::nn::AdaptiveAvgPool2dImpl")
			{
				auto spAdaptiveAvgPool2d = std::dynamic_pointer_cast<torch::nn::AdaptiveAvgPool2dImpl>(m);
				input = spAdaptiveAvgPool2d->forward(input);
				continue;
			}
			else
			{
				printf("Unsupported module '%s'.\n", m->name().c_str());
			}
		}

		return input;
	}

	// travel
	torch::Tensor out;

	if (_forward(flow_start->FirstChildElement(), input, out))
		return out;

	throw std::exception("no tensor is available");
}

int BaseNNet::LoadModule(tinyxml2::XMLElement* moduleElement)
{
	int iRet = 0;
	if (moduleElement == NULL)
		return -1;

	const char* szModuleType = moduleElement->Attribute("type");
	if (szModuleType == NULL)
	{
		printf("Please specify the module type.\n");
		return -1;
	}

	const char* szModuleName = moduleElement->Attribute("name");
	if (szModuleName == NULL)
	{
		printf("Please specify the module name.\n");
		return -1;
	}

	if (XP_STRICMP(szModuleType, "conv2d") == 0)
	{
		// extract the attributes of conv2d
		int64_t in_channels = moduleElement->Int64Attribute("in_channels", -1LL); assert(in_channels > 0);
		int64_t out_channels = moduleElement->Int64Attribute("out_channels", -1LL); assert(out_channels > 0);
		int64_t kernel_size = moduleElement->Int64Attribute("kernel_size", -1LL); assert(kernel_size > 0);
		int64_t stride = moduleElement->Int64Attribute("stride", 1LL); assert(stride > 0);
		int64_t padding = moduleElement->Int64Attribute("padding", 0LL);
		bool bias = moduleElement->BoolAttribute("bias", true);

		if (nn_cat == "ResNet")
		{
			// check whether it is the first full connect layer
			bool bHaveExistedConvLayer = false;
			for (auto& t : nn_module_types) {
				if (t.second == "conv2d")
				{
					bHaveExistedConvLayer = true;
					break;
				}
			}

			if (!bHaveExistedConvLayer)
			{
				auto use_32x32_input = nn_options.find("NN::use_32x32_input");
				if (use_32x32_input != nullptr)
				{
					int64_t b_use_32x32_input = 0;
					if (ConvertToInt((char*)use_32x32_input->c_str(), (char*)use_32x32_input->c_str() + use_32x32_input->length(), b_use_32x32_input))
					{
						printf("Change the neutral network to use small image size: %s.\n", b_use_32x32_input ? "yes" : "no");
						if (b_use_32x32_input)
						{
							kernel_size = 3;
							padding = 1;
							stride = 1;
						}
					}
				}
			}
		}


		std::shared_ptr<torch::nn::Module> spConv2D = 
			std::make_shared<torch::nn::Conv2dImpl>(
				torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(padding).stride(stride).bias(bias));
		nn_modules[szModuleName] = spConv2D;
		nn_module_types[szModuleName] = szModuleType;
		register_module(szModuleName, spConv2D);
	}
	else if (XP_STRICMP(szModuleType, "linear") == 0)
	{
		int64_t in_features = moduleElement->Int64Attribute("in_features", -1LL); assert(in_features > 0);
		int64_t out_features = moduleElement->Int64Attribute("out_features", -1LL); assert(out_features > 0);

		if (moduleElement->Parent()->LastChildElement() == moduleElement)
		{
			auto num_of_classes = nn_options.find("NN::final_out_classes");
			if (num_of_classes != nullptr)
			{
				if (ConvertToInt((char*)num_of_classes->c_str(), (char*)num_of_classes->c_str() + num_of_classes->length(), out_features))
				{
					printf("Change the neutral network output class number to %lld.\n", out_features);
				}
			}
		}

		if (nn_cat == "VGG")
		{
			// check whether it is the first full connect layer
			bool bHaveExistedFCLayer = false;
			for (auto& t : nn_module_types) {
				if (t.second == "linear")
				{
					bHaveExistedFCLayer = true;
					break;
				}
			}

			if (!bHaveExistedFCLayer)
			{
				auto use_32x32_input = nn_options.find("NN::use_32x32_input");
				if (use_32x32_input != nullptr)
				{
					int64_t b_use_32x32_input = 0;
					if (ConvertToInt((char*)use_32x32_input->c_str(), (char*)use_32x32_input->c_str() + use_32x32_input->length(), b_use_32x32_input))
					{
						printf("Change the neutral network to use small image size: %s.\n", b_use_32x32_input?"yes":"no");
						if (in_features == 512 * 7 * 7 && b_use_32x32_input)
							in_features = 512;
					}
				}
			}
		}

		std::shared_ptr<torch::nn::Module> spLinear =
			std::make_shared<torch::nn::LinearImpl>(in_features, out_features);
		nn_modules[szModuleName] = spLinear;
		nn_module_types[szModuleName] = szModuleType;
		register_module(szModuleName, spLinear);
	}
	else if (XP_STRICMP(szModuleType, "batchnorm2d") == 0)
	{
		int64_t num_features = moduleElement->Int64Attribute("num_features", -1); assert(num_features > 0);

		std::shared_ptr<torch::nn::Module> spBatchNorm2D =
			std::make_shared<torch::nn::BatchNorm2dImpl>(num_features);
		nn_modules[szModuleName] = spBatchNorm2D;
		nn_module_types[szModuleName] = szModuleType;
		register_module(szModuleName, spBatchNorm2D);
	}
	else if (XP_STRICMP(szModuleType, "adaptiveavgpool2d") == 0)
	{
		int64_t outputsize = moduleElement->Int64Attribute("outputsize", -1); assert(outputsize > 0);

		std::shared_ptr<torch::nn::Module> spAdaptiveAvgPool2D =
			std::make_shared<torch::nn::AdaptiveAvgPool2dImpl>(outputsize);
		nn_modules[szModuleName] = spAdaptiveAvgPool2D;
		nn_module_types[szModuleName] = szModuleType;
		register_module(szModuleName, spAdaptiveAvgPool2D);
	}
	else
	{
		printf("Failed to load the module with unsupported type: '%s'.\n", szModuleType);
		iRet = -1;
	}

	return iRet;
}