namespace tensorflow.keras.applications {
    using System;
    using System.Collections.Generic;

    using LostTech.Gradient;

    public static partial class YOLO {
        static (Tensor xywh, Tensor conf, Tensor prob) DecodeCommon(
                            Tensor convOut, int outputSize, int classCount,
                            ReadOnlySpan<int> strides, Tensor<int> anchors,
                            int scaleIndex, ReadOnlySpan<float> xyScale) {
            var varScope = new variable_scope("scale" + scaleIndex.ToString(System.Globalization.CultureInfo.InvariantCulture));
            using var _ = varScope.StartUsing();
            Tensor batchSize = tf.shape(convOut)[0];

            convOut = tf.reshape_dyn(convOut, new object[] { batchSize, outputSize, outputSize, 3, 5 + classCount });
            Tensor[] raws = tf.split(convOut, new[] { 2, 2, 1, classCount }, axis: -1);
            var (convRawDxDy, convRawDwDh, convRawConf, convRawProb) = raws;

            var meshgrid = tf.meshgrid(tf.range_dyn(outputSize), tf.range_dyn(outputSize));
            meshgrid = tf.expand_dims(tf.stack(meshgrid, axis: -1), axis: 2); // [gx, gy, 1, 2]
            Tensor xyGrid = tf.tile_dyn(
                tf.expand_dims(meshgrid, axis: 0),
                new object[] { tf.shape(convOut)[0], 1, 1, 3, 1 });

            xyGrid = tf.cast(xyGrid, tf.float32);

            var predictedXY = ((tf.sigmoid(convRawDxDy) * xyScale[scaleIndex]) - 0.5 * (xyScale[scaleIndex] - 1) + xyGrid) * strides[scaleIndex];
            var predictedWH = tf.exp(convRawDwDh) * tf.cast(anchors[scaleIndex], tf.float32);
            var predictedXYWH = tf.concat(new[] { predictedXY, predictedWH }, axis: -1);

            var predictedConf = tf.sigmoid(convRawConf);
            var predictedProb = tf.sigmoid(convRawProb);

            return (predictedXYWH, conf: predictedConf, prob: predictedProb);
        }

        static (Tensor boxes, Tensor conf) FilterBoxes(Tensor xywh, Tensor scores,
                                                       float scoreThreshold,
                                                       Tensor inputShape) {
            Tensor scoresMax = tf.reduce_max(scores, axis: new[] { -1 });
            Tensor mask = scoresMax >= scoreThreshold;
            Tensor classBoxes = tf.boolean_mask(xywh, mask);
            Tensor conf = tf.boolean_mask(scores, mask);

            Tensor count = tf.shape(scores)[0];

            classBoxes = tf.reshape_dyn(classBoxes, new object[] { count, -1, tf.shape(classBoxes)[^1] });
            conf = tf.reshape_dyn(conf, new object[] { count, -1, tf.shape(conf)[^1] });

            Tensor[] boxXY_WH = tf.split(classBoxes, new[] { 2, 2 }, axis: -1);
            var (boxXY, boxWH) = boxXY_WH;

            inputShape = tf.cast(inputShape, tf.float32);

            var boxYX = boxXY[tf.rest_of_the_axes, TensorDimensionSlice.Reverse];
            var boxHW = boxWH[tf.rest_of_the_axes, TensorDimensionSlice.Reverse];

            var boxMins = (boxYX - (boxHW / 2f)) / inputShape;
            var boxMaxes = (boxYX + (boxHW / 2f)) / inputShape;

            var boxes = tf.concat(new[] {
                boxMins[tf.rest_of_the_axes, 0..1], //y_min
                boxMins[tf.rest_of_the_axes, 1..2], //x_min
                boxMaxes[tf.rest_of_the_axes, 0..1], //y_max
                boxMaxes[tf.rest_of_the_axes, 1..2], //x_max
            }, axis: -1);

            return (boxes, conf);
        }
    }
}