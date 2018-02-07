using System;
using FeedForwardNeuralNetwork.NeuralNetwork;

namespace FeedForwardNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var nnb = new FeedForwardNeuralNetworkBuilder();

            var nn = nnb
                .InputLayerNeurons(2)
                .HiddenLayers(1)
                .HiddenLayerNeuronsCount(1, 2)
                .OutputLayerNeurons(1)
                .Build();
        }
    }
}
