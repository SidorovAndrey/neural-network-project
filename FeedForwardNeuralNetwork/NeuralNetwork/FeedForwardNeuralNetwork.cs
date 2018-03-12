using System;
using System.Collections.Generic;
using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public class FeedForwardNeuralNetwork : INeuralNetwork
    {
        private readonly Matrix[] _weights;
        private readonly Matrix[] _biases;
        private readonly double _learningRate;
        private readonly Func<double, double> _sigmoid;
        private readonly Func<double, double> _direvativeSigmoid;

        public FeedForwardNeuralNetwork(
            Matrix[] weights,
            Matrix[] biases,
            double learningRate,
            Func<double, double> sigmoid,
            Func<double, double> direvativeSigmoid)
        {
            _weights = weights;
            _biases = biases;
            _learningRate = learningRate;
            _sigmoid = sigmoid;
            _direvativeSigmoid = direvativeSigmoid;
        }

        public void Train(Matrix x, Matrix y)
        {
            // var input = x;

            // var hidden = MatrixMath.Product(_weights[0], input);
            // hidden = MatrixMath.Add(hidden, _biases[0]);
            // hidden.ApplyToEach(_sigmoid);

            // var output = MatrixMath.Product(_weights[1], hidden);
            // output = MatrixMath.Add(output, _biases[1]);
            // output.ApplyToEach(_sigmoid);

             var layersResults = GetLayersResults(x);

            // // calculate for w[1]
            // var outputErrors = MatrixMath.Substract(y, output);

            // output.ApplyToEach(_direvativeSigmoid);
            // var outputGradient = MatrixMath.HadamarProduct(output, outputErrors);
            // outputGradient = MatrixMath.Product(outputGradient, _learningRate);

            // var outputWeightsDelta = MatrixMath.Product(outputGradient, hidden.Transpose());

            // // calculate for w[0]
            // var hiddenErrors = MatrixMath.Product(_weights[1].Transpose(), outputErrors);

            // hidden.ApplyToEach(_direvativeSigmoid);
            // var hiddenGradient = MatrixMath.HadamarProduct(hidden, hiddenErrors);
            // hiddenGradient = MatrixMath.Product(hiddenGradient, _learningRate);

            // var hiddenWeightsDelta = MatrixMath.Product(hiddenGradient, input.Transpose());


             var errors = GetErrors(layersResults, y);

            // // update weigths with deltas
            // _weights[1] = MatrixMath.Add(_weights[1], outputWeightsDelta);
            // _biases[1] = MatrixMath.Add(_biases[1], outputGradient);

            // _weights[0] = MatrixMath.Add(_weights[0], hiddenWeightsDelta);
            // _biases[0] = MatrixMath.Add(_biases[0], hiddenGradient);

            UpdateWeights(layersResults, errors, x);
        }

        private Matrix[] GetLayersResults(Matrix x) 
        {
            var previousLayer = x;

            var layersResults = new Matrix[_weights.Length];
            for (int i = 0; i < _weights.Length; i++)
            {
                var result = MatrixMath.Product(_weights[i], previousLayer);
                result = MatrixMath.Add(result, _biases[i]);
                result.ApplyToEach(_sigmoid);

                layersResults[i] = result;

                previousLayer = result;
            }

            return layersResults;
        }

        private Matrix[] GetErrors(Matrix[] layersResults, Matrix y)
        {
            var errors = new Matrix[_weights.Length];
            errors[errors.Length - 1] = MatrixMath.Substract(y, layersResults[layersResults.Length - 1]);

            for (int i = errors.Length - 2; i >= 0; i--)
            {
                errors[i] = MatrixMath.Product(_weights[i + 1].Transpose(), errors[i + 1]);
            }

            return errors;
        }

        private void UpdateWeights(Matrix[] layersResults, Matrix[] errors, Matrix input)
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

            for(int i = gradients.Length - 1; i >= 0; i--)
            {
                _weights[i] = MatrixMath.Add(_weights[i], deltas[i]);
                _biases[i] = MatrixMath.Add(_biases[i], gradients[i]);
            }
        }

        public Matrix Predict(Matrix data)
        {
            var currentLayer = data;
            for (int i = 0; i < _weights.Length; i++)
            {
                var nextLayer = MatrixMath.Product(_weights[i], currentLayer);
                nextLayer = MatrixMath.Add(nextLayer, _biases[i]);
                nextLayer.ApplyToEach(_sigmoid);

                currentLayer = nextLayer;
            }

            return currentLayer;
        }
    }
}