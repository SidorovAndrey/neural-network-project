using FeedForwardNeuralNetwork.Math;

namespace FeedForwardNeuralNetwork.NeuralNetwork
{
    public class FeedForwardNeuralNetwork : INeuralNetwork
    {
        private readonly Matrix[] _weights;
        private readonly Matrix[] _biases;
        private readonly double _learningRate;
        private readonly Func<double, double> _signum;
        private readonly Func<double, double> _direvativeSignum;

        public FeedForwardNeuralNetwork(
            Matrix[] weights,
            Matrix[] biases,
            double learningRate,
            Func<double, double> signum,
            Func<double, double> direvativeSignum)
        {
            _weights = weights;
            _biases = biases;
            _learningRate = learningRate;
            _signum = signum;
            _direvativeSignum = direvativeSignum;
        }

        public void Train(Matrix x, Matrix y)
        {
            var input = x;

            var hidden = MatrixMath.Product(_weights[0], input);
            hidden = MatrixMath.Add(hidden, _biases[0]);
            hidden.ApplyToEach(_signum);

            var output = MatrixMath.Product(_weights[1], hidden);
            output = MatrixMath.Add(output, _biases[1]);
            output.ApplyToEach(_signum);

            // TODO: figure out all the math stuff here
            // TODO: generalize algorithm for any layers count

            // calculate for w[1]
            var outputErrors = MatrixMath.Substract(y, output);

            output.ApplyToEach(_direvativeSignum);
            var outputGradient = MatrixMath.HadamarProduct(output, outputErrors);
            outputGradient = MatrixMath.Product(outputGradient, _learningRate);

            var outputWeightsDelta = MatrixMath.Product(outputGradient, hidden.Transpose());

            // calculate for w[0]
            var hiddenErrors = MatrixMath.Product(_weights[1].Transpose(), outputErrors);

            hidden.ApplyToEach(_direvativeSignum);
            var hiddenGradient = MatrixMath.HadamarProduct(hidden, hiddenErrors);
            hiddenGradient = MatrixMath.Product(hiddenGradient, _learningRate);

            var hiddenWeightsDelta = MatrixMath.Product(hiddenGradient, input.Transpose());

            _weights[1] = MatrixMath.Add(_weights[1], outputWeightsDelta);
            _biases[1] = MatrixMath.Add(_biases[1], outputGradient);

            _weights[0] = MatrixMath.Add(_weights[0], hiddenWeightsDelta);
            _biases[0] = MatrixMath.Add(_biases[0], hiddenGradient);
        }

        public Matrix Predict(Matrix data)
        {
            var input = data;

            var hidden = MatrixMath.Product(_weights[0], input);
            hidden = MatrixMath.Add(hidden, _biases[0]);
            hidden.ApplyToEach(Sigm);

            var output = MatrixMath.Product(_weights[1], hidden);
            output = MatrixMath.Add(output, _biases[1]);
            output.ApplyToEach(Sigm);

            return output;
        }
    }
}