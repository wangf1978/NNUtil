#pragma once

#include <vector>

using STEP_LR = std::tuple<int64_t, double>;		// learning rate is started from which epoch

enum LRD_MODE
{
	LRD_MODE_UNKNOWN = -1,
	LRD_MODE_PIECEWISE_CONSTANT = 0,		// �ֶγ������ֶγ�ֵ������
	LRD_MODE_EXPONENTIAL_DECAY,
	LRD_MODE_NATURAL_EXP_DECAY,
	LRD_MODE_POLYNOMIAL_DECAY,
	LRD_MODE_COSINE_DECAY,
	LRD_MODE_LINEAR_COSINE_DECAY,
	LRD_MODE_NOISY_LINEAR_COSINE_DECAY,
	LRD_MODE_COSINE_DECAY_RESTARTS,
	LRD_MODE_INVERSE_TIME_DECAY,
};

class LearningRateMgr
{
public:
	LearningRateMgr(double init_lr)
		: learning_rate_mode(LRD_MODE_UNKNOWN)
		, initial_learning_rate(init_lr)
		, learning_rate(init_lr)
	{
	}

	void UsePieceWiseConstant(std::vector<STEP_LR>& steplrs);
	void UseExponentDecay(int64_t decaySteps, double decayRate, bool bStairUse = false);
	void UseNaturalExpDecay(int64_t decaySteps, double decayRate, bool bStairUse = false);
	void UsePolyNomialDecay(int64_t decaySteps, double endLearningRate = 0.0001, double power = 1.0, bool bCycle = false);
	void UseCosineDecay(int64_t decaySteps, double alpha = 0.0);
	void UseLinearCosineDecay(int64_t decaySteps, double numPeriods=0.5, double a = 0.0, double b=0.001);
	void UseNoisyLinearCosineDecay(int64_t decaySteps, double initial_variance = 1.0, double variance_decay = 0.55, 
		double numPeriods = 0.5, double a = 0.0, double b = 0.001);
	void UseInverseTimeDecay(int64_t decaySteps, double decayRate, bool bStairUse=false);

	void OnTrainEpochFinish();
	void OnTrainStepFinish(double step_lossrate);

	double GetLearningRate();
	void ResetLearningRate();

protected:
	void UpdateLearningRate();

protected:
	double		learning_rate;
	LRD_MODE	learning_rate_mode;			// ѧϰ�ʿ���ģʽ
	double		initial_learning_rate;		// ��ʼѧϰ�� 
	int64_t		decay_steps;				// ˥����������������ֵ������˥������ 
	double		power;						// ����ʽ˥������
	double		decay_rate;					// ˥���� 
	double		end_learning_rate;			// ��͵�����ѧϰ�� 
	bool		cycle;						// ѧϰ���½����Ƿ��������� 
	double		alpha;						// ��Сѧϰ�� 
	double		beta;						// 
	double		num_periods;				// ˥�����Ҳ��ֵ������� 
	double		initial_variance;			// �����ĳ�ʼ���� 
	double		variance_decay;				// ˥�������ķ���
	std::vector<STEP_LR> 
				step_lrs;					// ѧϰ�ʿ�ʼ��step, ����
											// {{0, 0.1}, {10, 0.001}}, from step#0, the learning rate is 0.1, and from step 10, learning rate is 0.001
	int64_t		global_epoch = 0;			// ����ѵ������
	int64_t		global_step = 0;			// ����˥�������ȫ�ֲ������Ǹ��������𲽼���˥��ָ�� 
	bool		staircase = false;
};

