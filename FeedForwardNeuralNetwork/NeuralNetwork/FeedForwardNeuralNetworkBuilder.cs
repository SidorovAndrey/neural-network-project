using System;
using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public class FeedForwardNeuralNetworkBuilder : INeuralNetworkBuilder
    {
        private int _intputNeuronsCount = 1;
        private int _outputNeuronsCount = 1;

        private int[] _hiddenLayersNeuronsCounts = new int[1];

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

        public INeuralNetwork Build()
        {
            var layers = GetLayers();
            var weights = GetWeights();
            return new FeedForwardNeuralNetwork(layers, weights);
        }

        private Matrix[] GetLayers()
        {
            var layers = new Matrix[_hiddenLayersNeuronsCounts.Length + 2];
            layers[0] = MatrixFactory.CreateVector(_intputNeuronsCount);
            for (var i = 1; i < layers.Length - 1; i++)
                layers[i] = MatrixFactory.CreateVector(_hiddenLayersNeuronsCounts[i - 1]);

            layers[layers.Length - 1] = MatrixFactory.CreateVector(_outputNeuronsCount);
            return layers;
        }

        private Matrix[] GetWeights()
        {
            var weights = new Matrix[_hiddenLayersNeuronsCounts.Length + 1];
            weights[0] = MatrixFactory.CreateRandomMatrix(_hiddenLayersNeuronsCounts[0], _intputNeuronsCount);
            for (var i = 1; i < _hiddenLayersNeuronsCounts.Length; i++)
                weights[i] = MatrixFactory.CreateRandomMatrix(_hiddenLayersNeuronsCounts[i], _hiddenLayersNeuronsCounts[i - 1]);

            return weights;
        }
    }
}