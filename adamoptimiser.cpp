#include "adamoptimiser.h"
#include <cmath>
#include <vector>

AdamOptimizer::AdamOptimizer(const std::vector<Parameter>& parameters, float beta1, float beta2, float epsilon)
    : m_parameters(parameters),
    m_beta1(beta1),
    m_beta2(beta2),
    m_epsilon(epsilon),
    m_timeStep(0),
    m_firstMoments(parameters.size(), 0.0F),
    m_secondMoments(parameters.size(), 0.0F)
{
}

const std::vector<AdamOptimizer::Parameter>& AdamOptimizer::step(const std::vector<float> &gradients)
{
    ++m_timeStep;
    for(size_t i = 0; i < m_parameters.size(); ++i) {
        m_firstMoments[i] = m_beta1 * m_firstMoments[i] + (1.0F - m_beta1) * gradients[i];
        m_secondMoments[i] = m_beta2 * m_secondMoments[i] + (1.0F - m_beta2) * gradients[i] * gradients[i];
        const auto firstMomentCorrected = m_firstMoments[i] / (1.0F - std::pow(m_beta1, m_timeStep));
        const auto secondMomentCorrected = m_secondMoments[i] / (1.0F - std::pow(m_beta2, m_timeStep));
        m_parameters[i].value -= m_parameters[i].learning_rate * firstMomentCorrected / (std::sqrt(secondMomentCorrected) + m_epsilon);
    }

    return m_parameters;
}
