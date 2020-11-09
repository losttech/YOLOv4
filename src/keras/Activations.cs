namespace tensorflow.keras {
    using LostTech.Gradient;
    using LostTech.Gradient.ManualWrappers;
    static class Activations {
        public static Tensor Mish(IGraphNodeBase input)
            // https://github.com/hunglc007/tensorflow-yolov4-tflite/commit/a61f81f9118df9cec4d53736648174f6fb113e5f#diff-69d62c22a92472901b83e55ac7c153317c649564d4ae9945dcaed27d37295867R41
            => input * tf.tanh(tf.nn.softplus(input));

        public static PythonFunctionContainer Mish_fn { get; }
            = PythonFunctionContainer.Of<IGraphNodeBase, Tensor>(Mish);
    }
}
