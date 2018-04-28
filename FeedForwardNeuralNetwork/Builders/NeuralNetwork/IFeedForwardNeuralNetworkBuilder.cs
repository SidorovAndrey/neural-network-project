using FeedForwardNeuralNetwork.NeuralNetwork;
using FeedForwardNeuralNetwork.Trainers;

namespace FeedForwardNeuralNetwork.Builders.NeuralNetwork
{
    public interface IFeedForwardNeuralNetworkBuilder
    {
        IFeedForwardNeuralNetworkBuilder HiddenLayers(int count);
        IFeedForwardNeuralNetworkBuilder InputLayersNeuronsCount(int count);
        IFeedForwardNeuralNetworkBuilder HiddenLayerNeuronsCount(int layerNumber, int count);
        IFeedForwardNeuralNetworkBuilder OutputLayersNeuronsCount(int count);
        IFeedForwardNeuralNetworkBuilder Trainer(INeuralNetworkTrainer trainer);
        INeuralNetwork Build();
    }
}
