using System;
using FeedForwardNeuralNetwork.Data;
using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.Trainers
{
    public interface INeuralNetworkTrainer
    {
        Func<double, double> Sigmoid { get; }
        void Train(IDataSample sample, Matrix[] weights, Matrix[] biases);
    }
}
