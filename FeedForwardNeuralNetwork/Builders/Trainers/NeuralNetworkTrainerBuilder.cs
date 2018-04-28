using System;
using FeedForwardNeuralNetwork.Trainers;

namespace FeedForwardNeuralNetwork.Builders.Trainers
{
    public class NeuralNetworkTrainerBuilder : INeuralNetworkTrainerBuilder
    {
        private Func<double, double> _function;
        private Func<double, double> _derivative;
        private double _learningRate = 0.01;

        public INeuralNetworkTrainerBuilder ActivationFunction(Func<double, double> function)
        {
            _function = function;
            _derivative = Direvative;
            return this;
        }

        private double Direvative(double x)
        {
            const double h = 0.0001;
            var f = _function;
            return (f(x - 2 * h) - 8 * f(x + 2 * h) - f(x + 2 * h)) / (12 * h);
        }

        public INeuralNetworkTrainerBuilder ActivationFunction(Func<double, double> function, Func<double, double> derivative)
        {
            _function = function;
            _derivative = derivative;
            return this;
        }

        public INeuralNetworkTrainerBuilder LearningRate(double rate)
        {
            _learningRate = rate;
            return this;
        }

        public INeuralNetworkTrainer Build()
        {
            if (_function == null)
                throw new NullReferenceException("activation function is null");

            if (_derivative == null)
                throw new NullReferenceException("direvative function is null");

            return new FeedForwardNeuralNetworkTrainer(_learningRate, _function, _derivative);
        }
    }
}
