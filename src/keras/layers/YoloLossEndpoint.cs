namespace tensorflow.keras.layers {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using LostTech.Gradient.ManualWrappers;

    using tensorflow.keras.applications;

    public class YoloLossEndpoint : Layer {
        readonly Tensor<float>[] trueLabels;
        readonly Tensor<float>[] trueBoxes;
        readonly int[] strides;
        readonly int classCount;

        public YoloLossEndpoint(Tensor<float>[] trueLabels, Tensor<float>[] trueBoxes,
                                ReadOnlySpan<int> strides,
                                int classCount) {
            this.trueLabels = trueLabels ?? throw new ArgumentNullException(nameof(trueLabels));
            this.trueBoxes = trueBoxes ?? throw new ArgumentNullException(nameof(trueBoxes));
            if (strides.Length == 0) throw new ArgumentException();
            this.classCount = classCount;

            this.strides = strides.ToArray();
        }

        public Tensor call(IEnumerable<IGraphNodeBase> trainableOutputs) {
            var output = trainableOutputs.ToArray();
            var loss = YOLO.Loss.Zero;
            for(int scaleIndex = 0; scaleIndex < this.strides.Length; scaleIndex++) {
                IGraphNodeBase conv = output[scaleIndex * 2];
                IGraphNodeBase pred = output[scaleIndex * 2 + 1];

                loss += YOLO.ComputeLoss((Tensor)pred, (Tensor)conv,
                                targetLabels: this.trueLabels[scaleIndex],
                                targetBBoxes: this.trueBoxes[scaleIndex],
                                strideSize: this.strides[scaleIndex],
                                classCount: this.classCount,
                                intersectionOverUnionLossThreshold: YOLO.DefaultIntersectionOverUnionLossThreshold);
            }

            this.add_loss(loss.Conf);
            this.add_loss(loss.GIUO);
            this.add_loss(loss.Prob);
            return loss.Conf + loss.GIUO + loss.Prob;
        }
    }
}
