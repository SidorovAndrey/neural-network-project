using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public interface INeuralNetwork
    {
        void Train(Matrix x, Matrix y);
        Matrix Predict(Matrix data);
    }
}