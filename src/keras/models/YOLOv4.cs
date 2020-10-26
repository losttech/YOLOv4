namespace tensorflow.keras.models {
    using System;

    using LostTech.Gradient.ManualWrappers;

    using tensorflow.keras.layers;

    public class YOLOv4 : Model {
        static readonly int[] anchors = new[]{12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401};
        public static ReadOnlySpan<int> Anchors => anchors;
        static readonly int[] strides = { 8, 16, 32 };
        public static ReadOnlySpan<int> Strides => strides;
        static readonly float[] xyScale = { 1.2f, 1.1f, 1.05f };
        public static ReadOnlySpan<float> XYScale => xyScale;
        public static int AnchorsPerScale => 3;

        public static Output Apply(IGraphNodeBase input, int classCount) {
            if (classCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(classCount));

            static Tensor Conv1_3_1_3_1(Tensor input, int inSize, int outSize) {
                var result = input;
                result = Blocks.Conv(result, new[] { 1, 1, inSize, outSize });
                result = Blocks.Conv(result, new[] { 3, 3, outSize, inSize });
                result = Blocks.Conv(result, new[] { 1, 1, inSize, outSize });
                result = Blocks.Conv(result, new[] { 3, 3, outSize, inSize });
                result = Blocks.Conv(result, new[] { 1, 1, inSize, outSize });
                return result;
            }

            var (skip1, skip2, backboneOut) = CrossStagePartialDarknet53.Apply(input);
            var conv = backboneOut;

            var route = conv;
            conv = Blocks.Conv(conv, new[] { 1, 1, 512, 256 });
            conv = Blocks.Upsample(conv);

            skip2 = Blocks.Conv(skip2, new[] { 1, 1, 512, 256 });
            conv = tf.concat(new[] { skip2, conv }, axis: -1);

            conv = Conv1_3_1_3_1(conv, 512, 256);

            skip2 = conv;
            conv = Blocks.Conv(conv, new[] { 1, 1, 256, 128 });
            conv = Blocks.Upsample(conv);

            skip1 = Blocks.Conv(skip1, new[] { 1, 1, 256, 128 });
            conv = tf.concat(new[] { skip1, conv }, axis: -1);

            conv = Conv1_3_1_3_1(conv, 256, 128);

            skip1 = conv;
            conv = Blocks.Conv(conv, new[] { 3, 3, 128, 256 });
            var conv_sbbox = Blocks.Conv(conv, new[] { 1, 1, 256, 3 * (classCount + 5) },
                                         activation: null, batchNorm: false);

            conv = Blocks.Conv(skip1, new[] { 3, 3, 128, 256 }, downsample: true);
            conv = tf.concat(new[] { conv, skip2 }, axis: -1);

            conv = Conv1_3_1_3_1(conv, 512, 256);

            skip2 = conv;
            conv = Blocks.Conv(conv, new[] { 3, 3, 256, 512 });
            var conv_mbbox = Blocks.Conv(conv, new[] { 1, 1, 512, 3 * (classCount + 5) },
                                         activation: null, batchNorm: false);

            conv = Blocks.Conv(skip2, new[] { 3, 3, 256, 512 }, downsample: true);
            conv = tf.concat(new[] { conv, route }, axis: -1);

            conv = Conv1_3_1_3_1(conv, 1024, 512);

            conv = Blocks.Conv(conv, new[] { 3, 3, 512, 1024 });
            var conv_lbbox = Blocks.Conv(conv, new[] { 1, 1, 1024, 3 * (classCount + 5) },
                                         activation: null, batchNorm: false);

            return new Output { SSBox = conv_sbbox, MBBox = conv_mbbox, LBBox = conv_lbbox };
        }

        public struct Output {
            public Tensor SSBox { get; set; }
            public Tensor MBBox { get; set; }
            public Tensor LBBox { get; set; }
        }
    }
}
