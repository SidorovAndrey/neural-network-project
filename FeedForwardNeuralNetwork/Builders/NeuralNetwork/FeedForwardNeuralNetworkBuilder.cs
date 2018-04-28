using System;
using FeedForwardNeuralNetwork.Math;
using FeedForwardNeuralNetwork.NeuralNetwork;
using FeedForwardNeuralNetwork.Trainers;

namespace FeedForwardNeuralNetwork.Builders.NeuralNetwork
{
    public class FeedForwardNeuralNetworkBuilder : IFeedForwardNeuralNetworkBuilder
    {
        private int _intputNeuronsCount = 1;
        private int _outputNeuronsCount = 1;

        private int[] _hiddenLayersNeuronsCounts = new int[1];

        private INeuralNetworkTrainer _trainer;

        public FeedForwardNeuralNetworkBuilder()
        {
            for (var i = 0; i < _hiddenLayersNeuronsCounts.Length; i++)
                _hiddenLayersNeuronsCounts[i] = 1;
        }

        public IFeedForwardNeuralNetworkBuilder HiddenLayers(int count)
        {
            _hiddenLayersNeuronsCounts = new int[count];
            return this;
        }

        public IFeedForwardNeuralNetworkBuilder InputLayersNeuronsCount(int count)
        {
            _intputNeuronsCount = count;
            return this;
        }

        public IFeedForwardNeuralNetworkBuilder HiddenLayerNeuronsCount(int layer, int count)
        {
            if (layer > _hiddenLayersNeuronsCounts.Length)
                throw new ArgumentOutOfRangeException(nameof(layer),
                    $"Specified less layers cound in {nameof(HiddenLayers)} method");

            _hiddenLayersNeuronsCounts[layer - 1] = count;
            return this;
        }

        public IFeedForwardNeuralNetworkBuilder OutputLayersNeuronsCount(int count)
        {
            _outputNeuronsCount = count;
            return this;
        }

        public IFeedForwardNeuralNetworkBuilder Trainer(INeuralNetworkTrainer trainer)
        {
            _trainer = trainer;
            return this;
        }

        public INeuralNetwork Build()
        {
            if (_trainer == null)
                throw new NullReferenceException("Trainer can't be null");

            var weights = GetWeights();
            var biases = GetBiases(weights);
            return new FeedForwardNeuralNetwork.NeuralNetwork.FeedForwardNeuralNetwork(weights, biases, _trainer);
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