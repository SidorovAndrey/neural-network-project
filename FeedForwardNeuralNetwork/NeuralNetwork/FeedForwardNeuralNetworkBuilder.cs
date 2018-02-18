using System;
using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public class FeedForwardNeuralNetworkBuilder : INeuralNetworkBuilder
    {
        private int _intputNeuronsCount = 1;
        private int _outputNeuronsCount = 1;

        private int[] _hiddenLayersNeuronsCounts = new int[1];
        private double _learningRate = 0.01;

        private Func<double, double> _function;
        private Func<double, double> _direvative;

        public FeedForwardNeuralNetworkBuilder()
        {
            for (var i = 0; i < _hiddenLayersNeuronsCounts.Length; i++)
                _hiddenLayersNeuronsCounts[i] = 1;
        }

        public FeedForwardNeuralNetworkBuilder HiddenLayers(int count)
        {
            _hiddenLayersNeuronsCounts = new int[count];
            return this;
        }

        public FeedForwardNeuralNetworkBuilder InputLayerNeurons(int count)
        {
            _intputNeuronsCount = count;
            return this;
        }

        public FeedForwardNeuralNetworkBuilder ActivationFunction(Func<double, double> function, Func<double, double> direvative)
        {
            _function = function;
            _direvative = direvative;
        }

        public FeedForwardNeuralNetworkBuilder HiddenLayerNeuronsCount(int layer, int count)
        {
            if (layer > _hiddenLayersNeuronsCounts.Length)
                throw new ArgumentOutOfRangeException(nameof(layer),
                    $"Specified less layers cound in {nameof(HiddenLayers)} method");

            _hiddenLayersNeuronsCounts[layer - 1] = count;
            return this;
        }

        public FeedForwardNeuralNetworkBuilder OutputLayerNeurons(int count)
        {
            _outputNeuronsCount = count;
            return this;
        }

        public FeedForwardNeuralNetworkBuilder LearningRate(double rate)
        {
            _learningRate = rate;
            return this;
        }

        public INeuralNetwork Build()
        {
            var weights = GetWeights();
            var biases = GetBiases(weights);
            return new FeedForwardNeuralNetwork(weights, biases, _learningRate, _function, _direvative);
        }

        private Matrix[] GetBiases(Matrix[] weights)
        {
            var result = new Matrix[weights.Length];
            for (var i = 0; i < result.Length; i++)
                result[i] = MatrixFactory.CreateRandomVector(weights[i].Rows);

            return result;
        }

        private Matrix[] GetWeights()
        {
            var weightsCount = _hiddenLayersNeuronsCounts.Length + 1;
            var weights = new Matrix[weightsCount];

            weights[0] = MatrixFactory.CreateRandomMatrix(_hiddenLayersNeuronsCounts[0], _intputNeuronsCount);
            if (_hiddenLayersNeuronsCounts.Length > 1)
            {
                for (var i = 1; i < _hiddenLayersNeuronsCounts.Length; i++)
                    weights[i] = MatrixFactory.CreateRandomMatrix(_hiddenLayersNeuronsCounts[i], _hiddenLayersNeuronsCounts[i - 1]);
            }

            weights[weightsCount - 1] = MatrixFactory.CreateRandomMatrix(_outputNeuronsCount,
                _hiddenLayersNeuronsCounts[_hiddenLayersNeuronsCounts.Length - 1]);

            return weights;
        }
    }
}