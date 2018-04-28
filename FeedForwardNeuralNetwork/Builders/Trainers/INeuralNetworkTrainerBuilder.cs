using System;
using FeedForwardNeuralNetwork.Trainers;

namespace FeedForwardNeuralNetwork.Builders.Trainers
{
    public interface INeuralNetworkTrainerBuilder
    {
        INeuralNetworkTrainerBuilder ActivationFunction(Func<double, double> function);
        INeuralNetworkTrainerBuilder ActivationFunction(Func<double, double> function, Func<double, double> derivative);
        INeuralNetworkTrainerBuilder LearningRate(double rate);
        INeuralNetworkTrainer Build();
    }
}
