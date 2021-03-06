using System;

namespace FeedForwardNeuralNetwork.Math
{
    public static class MatrixMath 
    {
        public static Matrix Add(Matrix matrix, double value)
        {
            var result = matrix.Clone();
            result.ApplyToEach(x => x + value);
            return result;
        }

        public static Matrix Add(Matrix first, Matrix second)
        {
            if (first.Rows != second.Rows || first.Columns != second.Columns)
                throw new ArgumentException("Matrices has different size");

            var result = MatrixFactory.CreateMatrix(first.Rows, first.Columns);
            for (var i = 0; i < first.Rows; i++)
                for (var j = 0; j < first.Columns; j++)
                    result[i, j] = first[i, j] + second[i, j];

            return result;
        }

        public static Matrix Substract(Matrix matrix, double value)
        {
            var result = matrix.Clone();
            result.ApplyToEach(x => x - value);
            return result;
        }

        public static Matrix Substract(Matrix first, Matrix second)
        {
            if (first.Rows != second.Rows || first.Columns != second.Columns)
                throw new ArgumentException("Matrices has different size");

            var result = MatrixFactory.CreateMatrix(first.Rows, first.Columns);
            for (var i = 0; i < first.Rows; i++)
                for (var j = 0; j < first.Columns; j++)
                    result[i, j] = first[i, j] - second[i, j];

            return result;
        }

        public static Matrix Product(Matrix matrix, double value)
        {
            var result = matrix.Clone();
            result.ApplyToEach(x => x * value);
            return result;
        }

        public static Matrix Product(Matrix first, Matrix second)
        {
            if (first.Columns != second.Rows)
                throw new ArgumentException("Matrices does not satisfying multiplication constraint");

            var result = MatrixFactory.CreateMatrix(first.Rows, second.Columns);
            for (int i = 0; i < result.Rows; i++)
                for (int j = 0; j < result.Columns; j++)
                {
                    var sum = 0d;
                    for (int k = 0; k < first.Columns; k++)
                        sum += first[i, k] * second[k, j];

                    result[i, j] = sum;
                }

            return result;
        }

        public static Matrix HadamarProduct(Matrix first, Matrix second)
        {
            if (first.Rows != second.Rows || first.Columns != second.Columns)
                throw new ArgumentException("Matrices should has equals size");

            var result = MatrixFactory.CreateMatrix(first.Rows, first.Columns);
            for (var i = 0; i < result.Rows; i++)
                for (var j = 0; j < result.Columns; j++)
                    result[i, j] = first[i, j] * second[i, j];

            return result;
        }

        public static Matrix Transpose(this Matrix matrix)
        {
            var result = MatrixFactory.CreateMatrix(matrix.Columns, matrix.Rows);
            for (int i = 0; i < result.Rows; i++)
                for (int j = 0; j < result.Columns; j++)
                    result[i, j] = matrix[j, i];

            return result;
        }
    }
}