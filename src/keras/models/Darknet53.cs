namespace tensorflow.keras.models {
    using System;
    using System.Collections.Generic;

    using LostTech.Gradient.ManualWrappers;

    using static Tools;
    public class Darknet53 : Model {
        (Tensor, Tensor, Tensor) CallImpl(IGraphNodeBase input, object? mask) {
            if (mask != null)
                throw new NotImplementedException("mask");

            var result = (Tensor)input;

            result = Blocks.Conv(result, new[] { 3, 3, 3, 32 });
            result = Blocks.Conv(result, new[] { 3, 3, 32, 64 }, downsample: true);

            foreach (var _ in Repeat(1))
                result = Blocks.Residual(result, 64, 32, 64);

            result = Blocks.Conv(result, new[] { 3, 3, 64, 128 }, downsample: true);

            foreach (var _ in Repeat(2))
                result = Blocks.Residual(result, 128, 64, 128);

            result = Blocks.Conv(result, new[] { 3, 3, 128, 256 }, downsample: true);

            foreach (var _ in Repeat(8))
                result = Blocks.Residual(result, 256, 128, 256);

            var skip1 = result;
            result = Blocks.Conv(result, new[] { 3, 3, 256, 512 }, downsample: true);

            foreach (var _ in Repeat(8))
                result = Blocks.Residual(result, 512, 256, 512);

            var skip2 = result;
            result = Blocks.Conv(result, new[] { 3, 3, 512, 1024 }, downsample: true);

            foreach (var _ in Repeat(4))
                result = Blocks.Residual(result, 1024, 512, 1024);

            return (skip1, skip2, result);
        }

        public new (Tensor, Tensor, Tensor) call(IGraphNodeBase inputs, IGraphNodeBase training, IGraphNodeBase mask)
            => this.CallImpl(inputs, mask);

        public new(Tensor, Tensor, Tensor) call(IGraphNodeBase inputs, bool training, IGraphNodeBase? mask = null)
            => this.CallImpl(inputs, mask);

        public new(Tensor, Tensor, Tensor) call(IGraphNodeBase inputs, IGraphNodeBase? training = null, IEnumerable<IGraphNodeBase>? mask = null)
            => this.CallImpl(inputs, mask);
    }
}
