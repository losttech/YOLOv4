namespace tensorflow.keras {
    using System;

    using LostTech.Gradient.ManualWrappers;

    using tensorflow.image;
    using tensorflow.keras.layers;
    static class Blocks {
        public static Tensor Conv(IGraphNodeBase input, int[] filtersShape,
                                  Func<Tensor, Tensor>? activation,
                                  bool downsample = false,
                                  bool batchNorm = true
        ) {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (filtersShape is null) throw new ArgumentNullException(nameof(filtersShape));

            int strides = 1;
            IGraphNodeBase convolutionInput = input;
            string padding = "same";

            if (downsample) {
                convolutionInput = ZeroPadding2D.NewDyn(padding: ((1, 0), (1, 0))).__call__(input);
                padding = "valid";
                strides = 2;
            }

            var conv = new Conv2D(filters: filtersShape[^1], kernel_size: filtersShape[0],
                                  strides: strides, padding: padding,
                                  use_bias: !batchNorm,
                                  kernel_regularizer: tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer: new random_normal_initializer(stddev: 0.01),
                                  bias_initializer: new constant_initializer(0.0))
                .__call__(convolutionInput);

            if (batchNorm)
                conv = new FreezableBatchNormalization().__call__(conv);

            return activation is null ? conv : activation(conv);
        }

        static Tensor TunedLeakyRelu(Tensor input) => tf.nn.leaky_relu(input, alpha: 0.1);

        public static Tensor Conv(IGraphNodeBase input, int[] filtersShape, bool downsample = false, bool batchNorm = true)
            => Conv(input, filtersShape,
                    downsample: downsample, batchNorm: batchNorm,
                    activation: TunedLeakyRelu);

        public static Tensor Residual(Tensor input, int inputChannel, int filter1, int filter2,
                                      Func<Tensor, Tensor>? activation) {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (inputChannel <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannel));
            if (filter1 <= 0) throw new ArgumentOutOfRangeException(nameof(filter1));
            if (filter2 <= 0) throw new ArgumentOutOfRangeException(nameof(filter2));

            var shortcut = input;
            var conv = Conv(input, filtersShape: new[] { 1, 1, inputChannel, filter1 }, activation: activation);
            conv = Conv(conv, filtersShape: new[] { 3, 3, filter1, filter2 }, activation: activation);
            return shortcut + conv;
        }

        public static Tensor Residual(Tensor input, int inputChannel, int filter1, int filter2)
            => Residual(input, inputChannel: inputChannel,
                        filter1: filter1, filter2: filter2,
                        activation: TunedLeakyRelu);

        public static Tensor Upsample(Tensor input)
            => tf.image.resize(input, new [] { input.shape[1] * 2, input.shape[2] * 2 }, method: ResizeMethod.NEAREST_NEIGHBOR);
    }
}
