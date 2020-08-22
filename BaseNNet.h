#pragma once

#include <torch/torch.h>
#include "tinyxml2.h"
#include <map>
#include <vector>

using namespace torch::nn;

using NNetOptions = torch::OrderedDict<std::string, std::string>;

enum NN_TYPE
{
	NN_TYPE_UNKNOWN = -1,
	NN_TYPE_LENET = 0,
	NN_TYPE_VGGA = 100,
	NN_TYPE_VGGA_LRN,
	NN_TYPE_VGGB,
	NN_TYPE_VGGC,
	NN_TYPE_VGGD,
	NN_TYPE_VGGE,
	NN_TYPE_RESNET18 = 200,
	NN_TYPE_RESNET34,
	NN_TYPE_RESNET50,
	NN_TYPE_RESNET101,
	NN_TYPE_RESNET152,
};

#define MAX_LABEL_NAME		2048

class BaseNNet: public Module
{
public:
	int SetOptions(NNetOptions options);
	int Init(const char* szNNName);
	int Uninit();

	virtual int train(const char* szTrainSetRootPath, 
					  const char* szTrainSetStateFilePath,
					  int batch_size = 1, 
					  int num_epoch = 1,
					  float learning_rate = -1.0f,
					  unsigned int showloss_per_num_of_batches = 10) = 0;
	virtual void verify(const char* szTrainSetRootPath, const char* szTrainSetStateFilePath) = 0;
	virtual void classify(const char* szImageFile) = 0;
	virtual void Print();

	//
	// net load, save and unload
	//
	virtual int loadnet(const char* szTrainSetStateFilePath) = 0;
	/*!	@brief Save the current neutral network state to the specified file
		@remarks if the specified file path is NULL, then construct the neutral network from the options */
	virtual int savenet(const char* szTrainSetStateFilePath) = 0;
	virtual int unloadnet() = 0;

	virtual torch::Tensor forward(torch::Tensor& input);
	virtual const std::string& nnname() { return nn_model_name; }

protected:
	int LoadModule(tinyxml2::XMLElement* moduleElement);
	bool _forward(tinyxml2::XMLElement* first_sibling, const torch::Tensor& input, torch::Tensor& out);

	NN_TYPE nntype_from_options();
	int64_t	int64_from_options(const char* key);

protected:
	std::map<std::string, std::shared_ptr<torch::nn::Module>> nn_modules;
	std::map<std::string, std::string > nn_module_types;
	tinyxml2::XMLElement* flow_start = NULL;
	tinyxml2::XMLDocument xmlDoc;
	std::string nn_model_name;
	NNetOptions nn_options;
	std::string nn_cat;
};

