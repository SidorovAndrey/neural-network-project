using System;
using System.Collections;
using System.Collections.Generic;

namespace FeedForwardNeuralNetwork.Math 
{
    public class Matrix : IEnumerable<double>
    {
        private double[,] _core;

        public Matrix(double[,] core) 
        {
            _core = core;
        }

        public double this[int row, int column]
        {
            get { return _core[row, column]; }
            set { _core[row, column] = value; }
        }

        public int Rows { get { return _core.GetLength(0); } }

        public int Columns { get { return _core.GetLength(1); } }

        public void Apply(Func<double, double> callback)
        {
            for (int i = 0; i < _core.GetLength(0); i++)
                for (int j = 0; j < _core.GetLength(1); j++)
                    _core[i, j] = callback(_core[i, j]);
        }

        public Matrix Clone()
        {
            var copy = (double[,])_core.Clone();
            return new Matrix(copy);
        }

        public IEnumerator<double> GetEnumerator()
        {
            for (int i = 0; i < _core.GetLength(0); i++)
                for (int j = 0; j < _core.GetLength(1); j++)
                    yield return _core[i, j];
        }


        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}