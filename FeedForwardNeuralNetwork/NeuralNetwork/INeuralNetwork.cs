using FeedForwardNeuralNetwork.Data;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public interface INeuralNetwork
    {
        void Train(TrainingData data);
        double Predict(InputData data);
    }
}