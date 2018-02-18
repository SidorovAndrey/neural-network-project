namespace FeedForwardNeuralNetwork.Math
{
    public static class ActivationFunctions
    {
        public static double Signum(double x)
        {
            return 1 / (1 + System.Math.Exp(-x));
        }

        public static double DerevativeSignum(double x)
        {
            return x * (1 - x);
        }
    }
}