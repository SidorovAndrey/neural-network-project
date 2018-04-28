using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.Data
{
    public class DataSample : IDataSample
    {
        public Matrix X { get; set; }
        public Matrix Y { get; set; }
    }
}
