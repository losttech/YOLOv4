namespace tensorflow.keras {
    using LostTech.Gradient;
    using LostTech.Gradient.ManualWrappers;
    static class Activations {
        public static Tensor Softplus(IGraphNodeBase input, double threshold = 20) {
            Tensor softlog(IGraphNodeBase input) => tf.log1p(tf.exp(input));

            return tf.@case(
                new (object, object)[]{
                    (tf.greater(input, threshold), tf.identity_fn),
                    (tf.less(input, -threshold), tf.exp_fn),
                },
                @default: PythonFunctionContainer.Of<IGraphNodeBase, Tensor>(softlog));
        }

        public static Tensor Mish(IGraphNodeBase input)
            => input * tf.tanh(tf.log1p(tf.exp(input)));

        public static PythonFunctionContainer Mish_fn { get; }
            = PythonFunctionContainer.Of<IGraphNodeBase, Tensor>(Mish);
    }
}
