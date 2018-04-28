using System.Collections.Generic;
using System.Linq;
using FeedForwardNeuralNetwork.Data;
using FeedForwardNeuralNetwork.Math;
using FeedForwardNeuralNetwork.Trainers;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public class FeedForwardNeuralNetwork : INeuralNetwork
    {
        private readonly Matrix[] _weights;
        private readonly Matrix[] _biases;
        private readonly INeuralNetworkTrainer _trainer;

        public FeedForwardNeuralNetwork(
            Matrix[] weights,
            Matrix[] biases,
            INeuralNetworkTrainer trainer)
        {
            _weights = weights;
            _biases = biases;
            _trainer = trainer;
        }

        public void Train(IEnumerable<IDataSample> samples, int count = 1)
        {
            var dataSamples = samples as IDataSample[] ?? samples.ToArray();
            for (var i = 0; i < count; i++)
            {
                foreach (var sample in dataSamples)
                {
                    _trainer.Train(sample, _weights, _biases);
                }
            }
        }

        public Matrix Predict(Matrix data)
        {
            var currentLayer = data;
            for (int i = 0; i < _weights.Length; i++)
            {
                var nextLayer = MatrixMath.Product(_weights[i], currentLayer);
                nextLayer = MatrixMath.Add(nextLayer, _biases[i]);
                nextLayer.ApplyToEach(_trainer.Sigmoid);

                currentLayer = nextLayer;
            }

            return currentLayer;
        }
    }
}