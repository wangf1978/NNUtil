#include "CmdLineParser.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

std::vector<std::string> split(const std::string& s, char delimiter)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

const std::map<std::string, NN_CMD, CaseInsensitiveComparator> CmdLineParser::mapCmds =
{
	{"help",		NN_CMD_HELP},
	{"state",		NN_CMD_STATE},
	{"train",		NN_CMD_TRAIN},
	{"verify",		NN_CMD_VERIFY},
	{"classify",	NN_CMD_CLASSIFY},
	{"show",		NN_CMD_SHOW},
	{"test",		NN_CMD_TEST}
};

const std::map<std::string, NN_TYPE, CaseInsensitiveComparator> CmdLineParser::mapNNTypes =
{
	{"LENET",		NN_TYPE_LENET},
	{"VGGA",		NN_TYPE_VGGA},
	{"VGGA_LRN",	NN_TYPE_VGGA_LRN},
	{"VGGB",		NN_TYPE_VGGB},
	{"VGGC",		NN_TYPE_VGGC},
	{"VGGD",		NN_TYPE_VGGD},
	{"VGGE",		NN_TYPE_VGGE},
	{"RESNET18",	NN_TYPE_RESNET18},
	{"RESNET34",	NN_TYPE_RESNET34},
	{"RESNET50",	NN_TYPE_RESNET50},
	{"RESNET101",	NN_TYPE_RESNET101},
	{"RESNET152",	NN_TYPE_RESNET152},
};

const std::map<std::string, OPTIM_TYPE, CaseInsensitiveComparator> CmdLineParser::mapOptimTypes
{
	{"SGD",			OPTIM_SGD },
	{"Adam",		OPTIM_Adam },
	{"AdamW",		OPTIM_AdamW },
	{"LBFGS",		OPTIM_LBFGS },
	{"RMSprop",		OPTIM_RMSprop },
	{"Adagrad",		OPTIM_Adagrad },
};

const std::map<std::string, IMGSET_TYPE, CaseInsensitiveComparator> CmdLineParser::mapImgsetTypes
{
	{"folder",		IMGSET_FOLDER },
	{"MNIST",		IMGSET_MNIST},
	{"CIFAR10",		IMGSET_CIFAR_10},
	{"CIFAR100",	IMGSET_CIFAR_100},
};

const std::map<std::string, LRD_MODE, CaseInsensitiveComparator> CmdLineParser::mapLRDModes
{
	{"exponent",	LRD_MODE_EXPONENTIAL_DECAY},
	{"natural_exp",	LRD_MODE_NATURAL_EXP_DECAY},
	{"polynomial",	LRD_MODE_POLYNOMIAL_DECAY},
	{"inversetime",	LRD_MODE_INVERSE_TIME_DECAY},
	{"cosine",		LRD_MODE_COSINE_DECAY},
	{"lcosine",		LRD_MODE_LINEAR_COSINE_DECAY},
};

COMMAND_OPTION CmdLineParser::options[] = {
	{"verbose",			"v",		ODT_INTEGER,		"1",		&CmdLineParser::m_cmdLineParser.verbose,				false,	false},
	{"quiet",			"y",		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.silence,				false,	false},
	{"type",			"t",		ODT_NNTYPE,			"VGGD",		&CmdLineParser::m_cmdLineParser.nn_type,				false,	false},
	{"root",			"r",		ODT_STRING,			NULL,		&CmdLineParser::m_cmdLineParser.image_set_root_path,	false,	false},
	{"imgset",			"i",		ODT_IMGSETTYPE,		"folder",	&CmdLineParser::m_cmdLineParser.imgset_type,			false,	false},
};

COMMAND_OPTION CmdLineParser::cmd_flags[] = {
	{"batchsize",		"b",		ODT_INTEGER,		"1",		&CmdLineParser::m_cmdLineParser.batchsize,				false,	false},
	{"epochnum",		"e",		ODT_INTEGER,		"1",		&CmdLineParser::m_cmdLineParser.epochnum,				false,	false},
	{"batchnorm",		NULL,		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.enable_batch_norm,		false,	true},
	{"bn",				NULL,		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.enable_batch_norm,		false,	true},
	{"numclass",		"n",		ODT_INTEGER,		NULL,		&CmdLineParser::m_cmdLineParser.num_classes,			false,	false},
	{"smallsize",		"s",		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.use_32x32_input,		false,	true},
	{"showloss",		NULL,		ODT_INTEGER,		NULL,		&CmdLineParser::m_cmdLineParser.showloss_per_num_of_batches,
																															false,	false},
	{"clean",			NULL,		ODT_BOOLEAN,		"true",		&CmdLineParser::m_cmdLineParser.clean_pretrain_net,		false,	true},
	{"weight_decay",	"w",		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.weight_decay,			false,  false},
	{"momentum",		"m",		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.momentum,				false,  false},
	{"optim",			"o",		ODT_OPTIMTYPE,		NULL,		&CmdLineParser::m_cmdLineParser.optim_type,				false,  false},
	//
	// Console the learning rate
	{"learningrate",	"l",		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.learningrate,			false,	false},
	{"lr",				"l",		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.learningrate,			false,	false},
	{"lrdm",			NULL,		ODT_LRDM,			NULL,		&CmdLineParser::m_cmdLineParser.learningrate_decay_mode,false,	false},
	{"lr_decay_steps",	NULL,		ODT_INTEGER,		NULL,		&CmdLineParser::m_cmdLineParser.lr_decay_steps,			false,	false},
	{"lr_decay_rate",	NULL,		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.lr_decay_rate,			false,	false},
	{"lr_staircase",	NULL,		ODT_BOOLEAN,		NULL,		&CmdLineParser::m_cmdLineParser.lr_staircase,			false,	false},
	{"lr_power",		NULL,		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.lr_power,				false,	false},
	{"lr_end",			NULL,		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.lr_end,					false,	false},
	{"lr_cycle",		NULL,		ODT_BOOLEAN,		NULL,		&CmdLineParser::m_cmdLineParser.lr_cycle,				false,	false},
	{"lr_alpha",		NULL,		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.lr_alpha,				false,	false},
	{"lr_beta",			NULL,		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.lr_beta,				false,	false},
	{"lr_num_periods",	NULL,		ODT_FLOAT,			NULL,		&CmdLineParser::m_cmdLineParser.lr_num_periods,			false,	false},
};

CmdLineParser CmdLineParser::m_cmdLineParser;

CmdLineParser& CmdLineParser::GetCmdLineParser()
{
	return m_cmdLineParser;
}

CmdLineParser::CmdLineParser()
{
}

CmdLineParser::~CmdLineParser()
{
}

void CmdLineParser::parse_options(std::vector<int>& args, COMMAND_OPTION* options_table, size_t table_size, std::vector<int>& unparsed_arg_indexes)
{
	int current_option_idx = -1;
	std::vector<std::string> parse_errors;
	auto iter = args.cbegin();
	while (iter != args.cend())
	{
		// check long pattern
		int cur_arg_idx = *iter++;
		const char* szItem = argv[cur_arg_idx];
		if (szItem[0] == '-' && strlen(szItem) > 1)
		{
			current_option_idx = -1;
			if (szItem[1] == '-')
			{
				// long format
				for (size_t i = 0; i < table_size; i++)
					if (XP_STRICMP(szItem + 2, options_table[i].option_name) == 0) {
						current_option_idx = (int)i;
						break;
					}
			}
			else if (szItem[1] != '\0' && szItem[2] == '\0')
			{
				for (size_t i = 0; i < table_size; i++)
					if (options_table[i].short_tag != NULL && strchr(options_table[i].short_tag, szItem[1]) != NULL) {
						current_option_idx = (int)i;
						break;
					}
			}

			if (current_option_idx < 0)
			{
				std::string err_msg = "Unrecognized option '";
				err_msg.append(szItem);
				err_msg.append("'");
				parse_errors.push_back(err_msg);
			}
			else
			{
				switch (options_table[current_option_idx].data_type)
				{
				case ODT_BOOLEAN:
					*((bool*)options_table[current_option_idx].value_ref) = options_table[current_option_idx].default_value_str != NULL &&
						XP_STRICMP(options_table[current_option_idx].default_value_str, "true") == 0 ? true : false;
					break;
				case ODT_INTEGER:
				{
					int64_t u64Val = 0;
					if (options_table[current_option_idx].default_value_str != NULL)
						ConvertToInt((char*)options_table[current_option_idx].default_value_str,
						(char*)options_table[current_option_idx].default_value_str + strlen(options_table[current_option_idx].default_value_str),
							u64Val);

					*((int32_t*)options_table[current_option_idx].value_ref) = (int32_t)u64Val;
					break;
				}
				case ODT_FLOAT:
				{
					double flVal = NAN;
					if (options_table[current_option_idx].default_value_str != NULL)
						flVal = atof(options_table[current_option_idx].default_value_str);
					*((double*)options_table[current_option_idx].value_ref) = flVal;
					break;
				}
				case ODT_STRING:
				{
					std::string *strVal = (std::string*)options_table[current_option_idx].value_ref;
					if (strVal && options_table[current_option_idx].default_value_str != NULL)
						strVal->assign(options_table[current_option_idx].default_value_str);
					break;
				}
				case ODT_NNTYPE:
				{
					NN_TYPE nn_type = NN_TYPE_UNKNOWN;
					if (options_table[current_option_idx].default_value_str != NULL)
					{
						auto iter = mapNNTypes.find(options_table[current_option_idx].default_value_str);
						if (iter != mapNNTypes.cend())
							nn_type = iter->second;
					}

					*((NN_TYPE*)options_table[current_option_idx].value_ref) = nn_type;
					break;
				}
				case ODT_OPTIMTYPE:
				{
					OPTIM_TYPE optim_type = OPTIM_UNKNOWN;
					if (options_table[current_option_idx].default_value_str != NULL)
					{
						auto iter = mapOptimTypes.find(options_table[current_option_idx].default_value_str);
						if (iter != mapOptimTypes.cend())
							optim_type = iter->second;
					}

					*((OPTIM_TYPE*)options_table[current_option_idx].value_ref) = optim_type;
					break;
				}
				case ODT_IMGSETTYPE:
				{
					IMGSET_TYPE imgset_type = IMGSET_UNKNOWN;
					if (options_table[current_option_idx].default_value_str != NULL)
					{
						auto iter = mapImgsetTypes.find(options_table[current_option_idx].default_value_str);
						if (iter != mapImgsetTypes.cend())
							imgset_type = iter->second;
					}
					*((IMGSET_TYPE*)options_table[current_option_idx].value_ref) = imgset_type;
					break;
				}
				case ODT_LRDM:
				{
					LRD_MODE lrd_mode = LRD_MODE_UNKNOWN;
					if (options_table[current_option_idx].default_value_str != NULL)
					{
						auto iter = mapLRDModes.find(options_table[current_option_idx].default_value_str);
						if (iter != mapLRDModes.cend())
							lrd_mode = iter->second;
					}
					*((LRD_MODE*)options_table[current_option_idx].value_ref) = lrd_mode;
					break;
				}
				case ODT_LIST:
					// Not implemented yet
					break;
				}
			}

			if (options_table[current_option_idx].switcher)
				current_option_idx = -1;

			continue;
		}

		if (current_option_idx >= 0)
		{
			switch (options_table[current_option_idx].data_type)
			{
			case ODT_BOOLEAN:
				*((bool*)options_table[current_option_idx].value_ref) = XP_STRICMP(szItem, "true") == 0 ? true : false;
				current_option_idx = -1;	// already consume the parameter
				break;
			case ODT_INTEGER:
			{
				int64_t u64Val = 0;
				ConvertToInt((char*)szItem, (char*)szItem + strlen(szItem), u64Val);
				*((int32_t*)options_table[current_option_idx].value_ref) = (int32_t)u64Val;
				current_option_idx = -1;	// already consume the parameter
				break;
			}
			case ODT_FLOAT:
			{
				double flVal = atof(szItem);
				*((double*)options_table[current_option_idx].value_ref) = flVal;
				current_option_idx = -1;	// already consume the parameter
				break;
			}
			case ODT_STRING:
			{
				std::string *strVal = (std::string*)options_table[current_option_idx].value_ref;
				strVal->assign(szItem);
				current_option_idx = -1;	// already consume the parameter
				break;
			}
			case ODT_NNTYPE:
			{
				auto iter = mapNNTypes.find(szItem);
				if (iter != mapNNTypes.cend())
					*((NN_TYPE*)options_table[current_option_idx].value_ref) = iter->second;
				break;
			}
			case ODT_OPTIMTYPE:
			{
				auto iter = mapOptimTypes.find(szItem);
				if (iter != mapOptimTypes.cend())
					*((OPTIM_TYPE*)options_table[current_option_idx].value_ref) = iter->second;
				break;
			}
			case ODT_IMGSETTYPE:
			{
				auto iter = mapImgsetTypes.find(szItem);
				if (iter != mapImgsetTypes.cend())
					*((IMGSET_TYPE*)options_table[current_option_idx].value_ref) = iter->second;
				break;
			}
			case ODT_LRDM:
			{
				auto iter = mapLRDModes.find(szItem);
				if (iter != mapLRDModes.cend())
					*((LRD_MODE*)options_table[current_option_idx].value_ref) = iter->second;
				break;
			}
			case ODT_LIST:
				// Not implemented yet
				break;
			}
		}
		else
		{
			unparsed_arg_indexes.push_back(cur_arg_idx);
		}
	}
}

void CmdLineParser::parse_cmdargs(std::vector<int32_t>& args)
{
	std::vector<std::string> parse_errors;

	if (args.empty())
		return;

	if (cmd == NN_CMD_HELP)
	{

	}
	else if (cmd == NN_CMD_TRAIN)
	{

	}
	else if (cmd == NN_CMD_VERIFY)
	{

	}
	else if (cmd == NN_CMD_CLASSIFY)
	{

	}
	else if (cmd == NN_CMD_TEST)
	{

	}
}

bool CmdLineParser::ProcessCommandLineArgs(int argc, const char* argv[])
{
	std::vector<int> cmdoptions;
	std::vector<int> cmdargs;
	std::vector<int>* active_args = &cmdoptions;
	std::vector<int> unparsed_arg_indexes;

	m_cmdLineParser.argc = argc;
	m_cmdLineParser.argv = argv;

	for (int i = 1; i < argc; i++)
	{
		auto iter = mapCmds.find(argv[i]);
		if (iter != mapCmds.cend())
		{
			// hit the command
			m_cmdLineParser.cmd = iter->second;
			active_args = &cmdargs;
			continue;
		}

		active_args->push_back(i);
	}

	unparsed_arg_indexes.clear();
	m_cmdLineParser.parse_options(cmdoptions, options, sizeof(options) / sizeof(options[0]), unparsed_arg_indexes);

	unparsed_arg_indexes.clear();
	m_cmdLineParser.parse_options(cmdargs, cmd_flags, sizeof(cmd_flags) / sizeof(cmd_flags[0]), unparsed_arg_indexes);

	if (unparsed_arg_indexes.size() > 0)
	{
		if (m_cmdLineParser.cmd == NN_CMD_TRAIN)
		{
			m_cmdLineParser.train_net_state_path = argv[unparsed_arg_indexes[0]];
		}
		else if (m_cmdLineParser.cmd == NN_CMD_VERIFY)
		{
			m_cmdLineParser.train_net_state_path = argv[unparsed_arg_indexes[0]];
		}
		else if (m_cmdLineParser.cmd == NN_CMD_CLASSIFY)
		{
			m_cmdLineParser.train_net_state_path = argv[unparsed_arg_indexes[0]];
			m_cmdLineParser.image_path = argv[unparsed_arg_indexes[1]];
		}
		else if (m_cmdLineParser.cmd == NN_CMD_STATE)
		{
			m_cmdLineParser.train_net_state_path = argv[unparsed_arg_indexes[0]];
		}
		else if (m_cmdLineParser.cmd == NN_CMD_TEST)
		{

		}
	}

	if (m_cmdLineParser.verbose > 0)
	{
		printf("unparsed arguments:\n");
		for (auto u : unparsed_arg_indexes)
			printf("\t%s\n", argv[u]);
	}

	m_cmdLineParser.parse_cmdargs(unparsed_arg_indexes);

	return true;
}

void CmdLineParser::Print()
{
	printf("verbose: %d\n", verbose);
	printf("silence: %s\n", silence ? "yes" : "no");
	auto iterCmd = mapCmds.cbegin();
	for (; iterCmd != mapCmds.cend(); iterCmd++)
	{
		if (iterCmd->second == cmd)
		{
			printf("command: %s\n", iterCmd->first.c_str());
			break;
		}
	}

	if (iterCmd == mapCmds.cend())
		printf("command: Unknown\n");

	if (cmd == NN_CMD_TRAIN || cmd == NN_CMD_VERIFY || cmd == NN_CMD_STATE && image_set_root_path.size() > 0)
	{
		printf("image set root path: %s\n", image_set_root_path.c_str());
		for (auto iterImgSetType = mapImgsetTypes.cbegin(); iterImgSetType != mapImgsetTypes.cend(); iterImgSetType++)
		{
			if (iterImgSetType->second == imgset_type)
			{
				printf("image set type: %s\n", iterImgSetType->first.c_str());
				break;
			}
		}
	}

	if (cmd == NN_CMD_TRAIN)
		printf("output train result: %s\n", train_net_state_path.c_str());
	else if (cmd == NN_CMD_VERIFY)
		printf("pre-trained net state: %s\n", train_net_state_path.c_str());

	if (cmd == NN_CMD_CLASSIFY)
	{
		printf("The file path of image to be clarified: %s\n", image_path.c_str());
		printf("The pre-trained net state: %s\n", train_net_state_path.c_str());
	}

	if (cmd == NN_CMD_TRAIN)
	{
		printf("batch size: %d\n", batchsize);
		printf("train epoch rounds: %d\n", epochnum);
		printf("learning rate: %f\n", learningrate);
		printf("enable batchnorm: %s\n", enable_batch_norm ? "yes" : "no");
		printf("num of output class: %d\n", num_classes);
		printf("the neutral network image input size: %s\n", use_32x32_input ? "32x32" : "224x224");
		printf("show loss per number of batches: %d\n", showloss_per_num_of_batches);
		printf("try to clean the previous train result: %s\n", clean_pretrain_net?"yes":"no");
		for (auto iterOptim = mapOptimTypes.cbegin(); iterOptim != mapOptimTypes.cend(); iterOptim++)
		{
			if (iterOptim->second == optim_type)
			{
				printf("optimizer: %s\n", iterOptim->first.c_str());
				break;
			}
		}
		printf("weight decay: %f\n", weight_decay);
	}
}