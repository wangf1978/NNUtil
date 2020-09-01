#include "LearningRateMgr.h"
#include <math.h>
#include <cmath>
#include <torch/torch.h>

void LearningRateMgr::OnTrainEpochFinish()
{
	global_epoch++;
}

void LearningRateMgr::OnTrainStepFinish(double step_lossrate)
{
	global_step++;
	UpdateLearningRate();
}

void LearningRateMgr::UsePieceWiseConstant(std::vector<STEP_LR>& steplrs)
{
	step_lrs = steplrs;
	learning_rate_mode = LRD_MODE_PIECEWISE_CONSTANT;
}

void LearningRateMgr::UseExponentDecay(int64_t decaySteps, double decayRate, bool bStairUse)
{
	decay_steps = decaySteps;
	decay_rate = decayRate;
	staircase = bStairUse;
	learning_rate_mode = LRD_MODE_EXPONENTIAL_DECAY;
}

void LearningRateMgr::UseNaturalExpDecay(int64_t decaySteps, double decayRate, bool bStairUse)
{
	decay_steps = decaySteps;
	decay_rate = decayRate;
	staircase = bStairUse;
	learning_rate_mode = LRD_MODE_NATURAL_EXP_DECAY;
}

void LearningRateMgr::UsePolyNomialDecay(int64_t decaySteps, double endLearningRate, double pow, bool bCycle)
{
	decay_steps = decaySteps;
	end_learning_rate = endLearningRate;
	power = pow;
	cycle = bCycle;
	learning_rate_mode = LRD_MODE_POLYNOMIAL_DECAY;
}

void LearningRateMgr::UseCosineDecay(int64_t decaySteps, double a)
{
	decay_steps = decaySteps;
	alpha = a;
	learning_rate_mode = LRD_MODE_COSINE_DECAY;
}

void LearningRateMgr::UseLinearCosineDecay(int64_t decaySteps, double numPeriods, double a, double b)
{
	decay_steps = decaySteps;
	num_periods = numPeriods;
	alpha = a;
	beta = b;
	learning_rate_mode = LRD_MODE_LINEAR_COSINE_DECAY;
}

void LearningRateMgr::UseNoisyLinearCosineDecay(int64_t decaySteps, 
	double initialVariance, double varianceDecay, double numPeriods, double a, double b)
{
	decay_steps = decaySteps;
	initial_variance = initialVariance;
	variance_decay = varianceDecay;
	num_periods = numPeriods;
	alpha = a;
	beta = b;
	learning_rate_mode = LRD_MODE_NOISY_LINEAR_COSINE_DECAY;
}

void LearningRateMgr::UseInverseTimeDecay(int64_t decaySteps, double decayRate, bool bStairUse)
{
	decay_steps = decaySteps;
	decay_rate = decayRate;
	staircase = bStairUse;
	learning_rate_mode = LRD_MODE_INVERSE_TIME_DECAY;
}

double LearningRateMgr::GetLearningRate()
{
	return learning_rate;
}

void LearningRateMgr::ResetLearningRate()
{
	learning_rate = initial_learning_rate;
}

void LearningRateMgr::UpdateLearningRate()
{
	switch (learning_rate_mode)
	{
	case LRD_MODE_PIECEWISE_CONSTANT:
		{
			double lr = initial_learning_rate;
			for (auto& step_lr : step_lrs)
			{
				if (global_step < std::get<0>(step_lr))
					break;
				else
					lr = std::get<1>(step_lr);
			}
			learning_rate = lr;
		}
		break;
	case LRD_MODE_EXPONENTIAL_DECAY:
		{
			double p = (double)global_step / decay_steps;
			if (staircase)
				p = floor(p);
			learning_rate = initial_learning_rate * pow(decay_rate, p);
		}
		break;
	case LRD_MODE_NATURAL_EXP_DECAY:
		{
			double p = (double)global_step / decay_steps;
			if (staircase)
				p = floor(p);

			learning_rate = initial_learning_rate * exp(-p * decay_rate);
		}
		break;
	case LRD_MODE_POLYNOMIAL_DECAY:
		if (cycle)
		{
			int64_t dsteps = decay_steps * ceil((double)global_step / decay_steps);
			learning_rate = (initial_learning_rate - learning_rate)*pow((1 - (double)global_step / decay_steps), power) + end_learning_rate;
		}
		else
		{
			int64_t step = global_step <= decay_steps ? global_step : decay_steps;
			learning_rate = (initial_learning_rate - learning_rate)*pow((1 - (double)step / decay_steps), power) + end_learning_rate;
		}
		break;
	case LRD_MODE_COSINE_DECAY:
		{
			int64_t gstep = global_step < decay_steps ? global_step : decay_steps;
			double cosine_decay = 0.5*(1 + cos(3.14159265358979323846*gstep / decay_steps));
			double decayed = (1 - alpha)*cosine_decay + alpha;
			learning_rate = initial_learning_rate * decayed;
		}
		break;
	case LRD_MODE_INVERSE_TIME_DECAY:
		{
			double p = (double)global_step / decay_steps;
			if (staircase)
				p = floor(p);
			learning_rate = initial_learning_rate / (1.0 + decay_rate * p);
		}
		break;
	case LRD_MODE_LINEAR_COSINE_DECAY:
		{
			int64_t gstep = global_step < decay_steps ? global_step : decay_steps;
			double linear_decay = (decay_steps - global_step) / decay_steps;
			double cosine_decay = 0.5*(1 + cos(3.14159265358979323846*2.0*num_periods* gstep / decay_steps));
			double decayed = (alpha + linear_decay)*cosine_decay + beta;
			learning_rate = initial_learning_rate * decayed;
		}
		break;
	case LRD_MODE_NOISY_LINEAR_COSINE_DECAY:
		{
			int64_t gstep = global_step < decay_steps ? global_step : decay_steps;
			double linear_decay = (double)(decay_steps - gstep) / decay_steps;
			double cosine_decay = 0.5 * (1 + cos(3.14159265358979323846 * 2 * num_periods * gstep / decay_steps));
			double variance = initial_variance / pow((1.0 + gstep), variance_decay);
			double std = sqrt(variance);
			auto randnorm = torch::tensor(0.);
			randnorm.normal_(0.0, std);
			double eps_t = randnorm.item().toDouble();
			double decayed = (alpha + linear_decay + eps_t)*cosine_decay + beta;
			learning_rate = initial_learning_rate * decayed;
		}
		break;
	}
}
