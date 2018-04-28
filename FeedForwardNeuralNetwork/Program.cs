using System;
using FeedForwardNeuralNetwork.Builders.NeuralNetwork;
using FeedForwardNeuralNetwork.Builders.Trainers;
using FeedForwardNeuralNetwork.Data;
using FeedForwardNeuralNetwork.Math;
using FeedForwardNeuralNetwork.NeuralNetwork;

namespace FeedForwardNeuralNetwork
{
    class Program
    {
        private static readonly IDataSample[] Data =
        {
            new DataSample
            {
                X = MatrixFactory.FromArray(new double[] { 0, 1 }),
                Y = MatrixFactory.FromArray(new double[] { 1 })
            },
            new DataSample
            {
                X = MatrixFactory.FromArray(new double[] { 1, 0 }),
                Y = MatrixFactory.FromArray(new double[] { 1 })
            },
            new DataSample
            {
                X = MatrixFactory.FromArray(new double[] { 0, 0 }),
                Y = MatrixFactory.FromArray(new double[] { 0 })
            },
            new DataSample
            {
                X = MatrixFactory.FromArray(new double[] { 1, 1 }),
                Y = MatrixFactory.FromArray(new double[] { 0 })
            },
        };

        static void Main(string[] args)
        {
            var nnTrainerBuilder = new NeuralNetworkTrainerBuilder();
            var trainer = nnTrainerBuilder
                .ActivationFunction(ActivationFunctions.Signum, ActivationFunctions.DerevativeSignum)
                .LearningRate(0.1)
                .Build();

            var nnb = new FeedForwardNeuralNetworkBuilder();

            var nn = nnb
                .InputLayersNeuronsCount(2)
                .HiddenLayers(1)
                .HiddenLayerNeuronsCount(1, 2)
                .OutputLayersNeuronsCount(1)
                .Trainer(trainer)
                .Build();

            Train(nn);

            var result1 = nn.Predict(MatrixFactory.FromArray(new double[] { 0, 0 }));
            Console.WriteLine($"0, 0 => {result1[0, 0]}");

            var result2 = nn.Predict(MatrixFactory.FromArray(new double[] { 0, 1 }));
            Console.WriteLine($"0, 1 => {result2[0, 0]}");

            var result3 = nn.Predict(MatrixFactory.FromArray(new double[] { 1, 0 }));
            Console.WriteLine($"1, 0 => {result3[0, 0]}");

            var result4 = nn.Predict(MatrixFactory.FromArray(new double[] { 1, 1 }));
            Console.WriteLine($"1, 1 => {result4[0, 0]}");
        }

        private static void Train(INeuralNetwork network)
        {
            network.Train(Data, 10000);
        }
    }
}
