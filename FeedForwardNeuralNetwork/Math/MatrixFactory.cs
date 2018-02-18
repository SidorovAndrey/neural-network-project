using System;

namespace FeedForwardNeuralNetwork.Math
{
    public static class MatrixFactory
    {
        private static readonly Random _random = new Random();

        public static Matrix FromArray(double[] array) 
        {
            var core = new double[array.Length, 1];
            for (var i = 0; i < core.Length; i++) 
                core[i, 0] = array[i];

            return new Matrix(core);
        }

        public static Matrix FromArray(double[,] array) 
        {
            return new Matrix(array);
        }

        public static Matrix CreateVector(int size)
        {
            var core = new double[size];
            return FromArray(core);
        }

        public static Matrix CreateMatrix(int rows, int columns)
        {
            var core = new double[rows, columns];
            return FromArray(core);
        }

        public static Matrix CreateRandomVector(int size)
        {
            var core = new double[size];
            for (var i = 0; i < core.Length; i++)
                core[i] = GetRandom();

            return FromArray(core);
        }

        public static Matrix CreateRandomMatrix(int rows, int columns)
        {
            var core = new double[rows, columns];
            for (var i = 0; i < core.GetLength(0); i++)
                for (var j = 0; j < core.GetLength(1); j++)
                    core[i, j] = GetRandom();

            return FromArray(core);
        }

        private static double GetRandom() 
        {
            return _random.NextDouble() * 2 - 1;
        }
    }
}