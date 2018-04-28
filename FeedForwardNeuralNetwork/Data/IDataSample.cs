using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.Data
{
    public interface IDataSample
    {
        Matrix X { get; set; }
        Matrix Y { get; set; }
    }
}
