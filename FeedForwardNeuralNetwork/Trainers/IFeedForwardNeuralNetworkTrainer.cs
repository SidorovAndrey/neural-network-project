using System;
using System.Collections.Generic;
using FeedForwardNeuralNetwork.Data;
using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.Trainers
{
    public class FeedForwardNeuralNetworkTrainer : INeuralNetworkTrainer
    {
        private readonly double _learningRate;
        private readonly Func<double, double> _sigmoid;
        private readonly Func<double, double> _direvativeSigmoid;

        public FeedForwardNeuralNetworkTrainer(double learningRate, Func<double, double> sigmoid, Func<double, double> direvativeSigmoid)
        {
            _learningRate = learningRate;
            _sigmoid = sigmoid;
            _direvativeSigmoid = direvativeSigmoid;
        }

        public Func<double, double> Sigmoid => _sigmoid;

        public void Train(IDataSample sample, Matrix[] weights, Matrix[] biases)
        {
            var layersResults = GetLayersResults(sample.X, weights, biases);
            var errors = GetErrors(layersResults, sample.Y, weights);
            UpdateWeights(layersResults, errors, sample.X, weights, biases);
        }

        private Matrix[] GetLayersResults(Matrix x, Matrix[] weights, Matrix[] biases)
        {
            var previousLayer = x;

            var layersResults = new Matrix[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                var result = MatrixMath.Product(weights[i], previousLayer);
                result = MatrixMath.Add(result, biases[i]);
                result.ApplyToEach(_sigmoid);

                layersResults[i] = result;

                previousLayer = result;
            }

            return layersResults;
        }

        private Matrix[] GetErrors(Matrix[] layersResults, Matrix y, Matrix[] weights)
        {
            var errors = new Matrix[weights.Length];
            errors[errors.Length - 1] = MatrixMath.Substract(y, layersResults[layersResults.Length - 1]);

            for (int i = errors.Length - 2; i >= 0; i--)
            {
                errors[i] = MatrixMath.Product(weights[i + 1].Transpose(), errors[i + 1]);
            }

            return errors;
        }

        private void UpdateWeights(Matrix[] layersResults, Matrix[] errors, Matrix input, Matrix[] weights, Matrix[] biases)
        {
            var gradients = new Matrix[errors.Length];
            var deltas = new Matrix[errors.Length];

            List<Matrix> layers = new List<Matrix>();
            layers.Add(input);
            layers.AddRange(layersResults);

            for (int i = errors.Length - 1; i >= 0; i--)
            {
                layersResults[i].ApplyToEach(_direvativeSigmoid);
                gradients[i] = MatrixMath.HadamarProduct(layersResults[i], errors[i]);
                gradients[i] = MatrixMath.Product(gradients[i], _learningRate);

                deltas[i] = MatrixMath.Product(gradients[i], layers[i].Transpose());
            }

            for (int i = gradients.Length - 1; i >= 0; i--)
            {
                weights[i] = MatrixMath.Add(weights[i], deltas[i]);
                biases[i] = MatrixMath.Add(biases[i], gradients[i]);
            }
        }
    }
}
