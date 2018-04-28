using System.Collections.Generic;
using FeedForwardNeuralNetwork.Data;
using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public interface INeuralNetwork
    {
        void Train(IEnumerable<IDataSample> samples, int count);
        Matrix Predict(Matrix data);
    }
}