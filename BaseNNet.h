#pragma once

#include <torch/torch.h>
#include "tinyxml2.h"
#include <map>
#include <vector>
#include "LearningRateMgr.h"
#include "ImageProcess.h"

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

enum OPTIM_TYPE
{
	OPTIM_UNKNOWN = -1,
	OPTIM_SGD = 0,
	OPTIM_Adam,
	OPTIM_AdamW,
	OPTIM_LBFGS,
	OPTIM_RMSprop,
	OPTIM_Adagrad,
};

#define OPTIM_NAME(o)	((o) == OPTIM_SGD?"SGD":(\
						 (o) == OPTIM_Adam?"Adam":(\
						 (o) == OPTIM_AdamW?"AdamW":(\
						 (o) == OPTIM_LBFGS?"LBFGS":(\
						 (o) == OPTIM_RMSprop?"RMSprop":(\
						 (o) == OPTIM_Adagrad?"Adagrad":"Unknown"))))))

#define MAX_LABEL_NAME		2048

class BaseNNet: public Module
{
public:
	int SetOptions(NNetOptions options);
	int Init(const char* szNNName);
	int Uninit();

	virtual int train(const char* szTrainSetRootPath, 
					  IMGSET_TYPE img_type,
					  const char* szTrainSetStateFilePath,
					  LearningRateMgr* pLRMgr,
					  int batch_size = 1, 
					  int num_epoch = 1,
					  unsigned int showloss_per_num_of_batches = 10,
					  double weight_decay = NAN,
					  double momentum = NAN,
					  OPTIM_TYPE optim_type = OPTIM_SGD) = 0;
	virtual void verify(const char* szTrainSetRootPath) = 0;
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
	int64_t	int64_from_options(const char* key, int64_t def = -1);

protected:
	std::map<std::string, std::shared_ptr<torch::nn::Module>> nn_modules;
	std::map<std::string, std::string > nn_module_types;
	tinyxml2::XMLElement* flow_start = NULL;
	tinyxml2::XMLDocument xmlDoc;
	std::string nn_model_name;
	NNetOptions nn_options;
	std::string nn_cat;
};

