using System;
using FeedForwardNeuralNetwork.Math;
using FeedForwardNeuralNetwork.NeuralNetwork;

namespace FeedForwardNeuralNetwork
{
    class Program
    {
        private static Random _random = new Random();

        private static TrainData[] _data = {
            new TrainData
            {
                X = MatrixFactory.FromArray(new double[] { 0, 1 }),
                Y = MatrixFactory.FromArray(new double[] { 1 })
            },
            new TrainData
            {
                X = MatrixFactory.FromArray(new double[] { 1, 0 }),
                Y = MatrixFactory.FromArray(new double[] { 1 })
            },
            new TrainData
            {
                X = MatrixFactory.FromArray(new double[] { 0, 0 }),
                Y = MatrixFactory.FromArray(new double[] { 0 })
            },
            new TrainData
            {
                X = MatrixFactory.FromArray(new double[] { 1, 1 }),
                Y = MatrixFactory.FromArray(new double[] { 0 })
            }
        };

        static void Main(string[] args)
        {
            var nnb = new FeedForwardNeuralNetworkBuilder();

            var nn = nnb
                .InputLayerNeurons(2)
                .HiddenLayers(1)
                .HiddenLayerNeuronsCount(1, 2)
                .OutputLayerNeurons(1)
                .LearningRate(0.1)
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

            Console.ReadLine();
        }

        private static void Train(INeuralNetwork network)
        {
            for (var i = 0; i < 10000; i++)
            {
                for (int j = 0; j < _data.Length; j++)
                {
                    network.Train(_data[j].X, _data[j].Y);
                }
                //var data = GetRandomData();
                //network.Train(data.X, data.Y);
            }
        }

        private static TrainData GetRandomData()
        {
            var index = _random.Next(0, _data.Length);
            return _data[index];
        }
    }
}
