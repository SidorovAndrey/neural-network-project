using FeedForwardNeuralNetwork.Data;
using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public class FeedForwardNeuralNetwork : INeuralNetwork
    {
        public Matrix[] Layers { get; }

        public Matrix[] Weights { get; }

        public FeedForwardNeuralNetwork(Matrix[] layers, Matrix[] weights)
        {
            Layers = layers;
            Weights = weights;
        }

        public void Train(TrainingData data)
        {
            throw new System.NotImplementedException();
        }

        public double Predict(InputData data)
        {
            throw new System.NotImplementedException();
        }
    }
}