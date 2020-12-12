namespace tensorflow.keras.models {
    using System;
    using System.Collections.Generic;

    using LostTech.Gradient.ManualWrappers;

    using static Tools;
    public class CrossStagePartialDarknet53: Model {
        (Tensor, Tensor, Tensor) CallImpl(IGraphNodeBase input, object? mask) {
            if (mask != null)
                throw new NotImplementedException("mask");

            return Apply(input);
        }

        public static (Tensor, Tensor, Tensor) Apply(IGraphNodeBase input) {
            static Tensor Conv(IGraphNodeBase input, int[] filtersShape, bool downsample = false, bool batchNorm = true)
                => Blocks.Conv(input, filtersShape, downsample: downsample, batchNorm: batchNorm,
                               activation: Activations.Mish);
            static Tensor Residual(Tensor input, int inputChannel, int filter1, int filter2)
                => Blocks.Residual(input, inputChannel, filter1, filter2,
                                   activation: Activations.Mish);

            var result = Conv(input, new[] { 3, 3, 3, 32 });
            result = Conv(result, new[] { 3, 3, 32, 64 }, downsample: true);

            var route = result;
            route = Conv(route, new[] { 1, 1, 64, 64 });
            result = Conv(result, new[] { 1, 1, 64, 64 });

            foreach (var _ in Repeat(1))
                result = Residual(result, 64, 32, 64);
            result = Conv(result, new[] { 1, 1, 64, 64 });

            result = tf.concat(new[] { result, route }, axis: -1);
            result = Conv(result, new[] { 1, 1, 128, 64 });
            result = Conv(result, new[] { 3, 3, 64, 128 }, downsample: true);

            route = result;
            route = Conv(route, new[] { 1, 1, 128, 64 });
            result = Conv(result, new[] { 1, 1, 128, 64 });

            foreach (var _ in Repeat(2))
                result = Residual(result, 64, 64, 64);
            result = Conv(result, new[] { 1, 1, 64, 64 });

            result = tf.concat(new[] { result, route }, axis: -1);

            result = Conv(result, new[] { 1, 1, 128, 128 });
            result = Conv(result, new[] { 3, 3, 128, 256 }, downsample: true);

            route = result;
            route = Conv(route, new[] { 1, 1, 256, 128 });
            result = Conv(result, new[] { 1, 1, 256, 128 });

            foreach (var _ in Repeat(8))
                result = Residual(result, 128, 128, 128);
            result = Conv(result, new[] { 1, 1, 128, 128 });

            result = tf.concat(new[] { result, route }, axis: -1);

            result = Conv(result, new[] { 1, 1, 256, 256 });
            var skip1 = result;
            result = Conv(result, new[] { 3, 3, 256, 512 }, downsample: true);

            route = result;
            route = Conv(route, new[] { 1, 1, 512, 256 });
            result = Conv(result, new[] { 1, 1, 512, 256 });

            foreach (var _ in Repeat(8))
                result = Residual(result, 256, 256, 256);
            result = Conv(result, new[] { 1, 1, 256, 256 });

            result = tf.concat(new[] { result, route }, axis: -1);

            result = Conv(result, new[] { 1, 1, 512, 512 });
            var skip2 = result;
            result = Conv(result, new[] { 3, 3, 512, 1024 }, downsample: true);

            route = result;
            route = Conv(route, new[] { 1, 1, 1024, 512 });
            result = Conv(result, new[] { 1, 1, 1024, 512 });

            foreach (var _ in Repeat(4))
                result = Residual(result, 512, 512, 512);
            result = Conv(result, new[] { 1, 1, 512, 512 });

            result = tf.concat(new[] { result, route }, axis: -1);

            result = Conv(result, new[] { 1, 1, 1024, 1024 });
            result = Blocks.Conv(result, new[] { 1, 1, 1024, 512 });
            result = Blocks.Conv(result, new[] { 3, 3, 512, 1024 });
            result = Blocks.Conv(result, new[] { 1, 1, 1024, 512 });

            result = tf.concat(new[] {
                tf.nn.max_pool(result, ksize: 13, padding: "SAME", strides: 1),
                tf.nn.max_pool(result, ksize: 9, padding: "SAME", strides: 1),
                tf.nn.max_pool(result, ksize: 5, padding: "SAME", strides: 1),
                result,
            }, axis: -1);

            result = Blocks.Conv(result, new[] { 1, 1, 2048, 512 });
            result = Blocks.Conv(result, new[] { 3, 3, 512, 1024 });
            result = Blocks.Conv(result, new[] { 1, 1, 1024, 512 });

            return (skip1, skip2, result);
        }

        public new(Tensor, Tensor, Tensor) call(IGraphNodeBase inputs, IGraphNodeBase training, IGraphNodeBase mask)
            => this.CallImpl(inputs, mask);

        public new(Tensor, Tensor, Tensor) call(IGraphNodeBase inputs, bool training, IGraphNodeBase? mask = null)
            => this.CallImpl(inputs, mask);

        public new(Tensor, Tensor, Tensor) call(IGraphNodeBase inputs, IGraphNodeBase? training = null, IEnumerable<IGraphNodeBase>? mask = null)
            => this.CallImpl(inputs, mask);
    }
}
