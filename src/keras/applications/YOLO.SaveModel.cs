namespace tensorflow.keras.applications {
    using System;
    using System.Collections.Generic;

    using LostTech.Gradient;
    using LostTech.Gradient.BuiltIns;

    using tensorflow.keras.models;

    partial class YOLO {
        public static Model CreateSaveable(int inputSize, int classCount,
                                          ReadOnlySpan<int> strides, Tensor<int> anchors,
                                          ReadOnlySpan<float> xyScale,
                                          float scoreThreshold) {
            Tensor input = tf.keras.Input(new TensorShape(inputSize, inputSize, 3));
            var featureMaps = YOLOv4.Apply(input, classCount: classCount);
            return CreateSaveable(inputSize: inputSize, input: input, featureMaps,
                                 classCount: classCount,
                                 strides: strides, anchors: anchors, xyScale: xyScale,
                                 scoreThreshold: scoreThreshold);
        }

        public static Model CreateSaveable(int inputSize, Tensor input, YOLOv4.Output featureMaps,
                                          int classCount,
                                          ReadOnlySpan<int> strides, Tensor<int> anchors,
                                          ReadOnlySpan<float> xyScale, float scoreThreshold) {
            var suppression = SelectBoxes(featureMaps, inputSize: inputSize, classCount: classCount,
                                          strides: strides, anchors: anchors,
                                          xyScale: xyScale,
                                          scoreThreshold: scoreThreshold);
            return new Model(new { inputs = input, outputs = new PythonList<Tensor> {
                suppression.Boxes, suppression.Scores, suppression.Classes, suppression.Detections,
            }}.AsKwArgs());
        }

        public static Tensor ProcessPrediction(int inputSize, YOLOv4.Output modelOutput, int classCount, ReadOnlySpan<int> strides, Tensor<int> anchors, ReadOnlySpan<float> xyScale, float scoreThreshold) {
            var bboxTensors = new List<Tensor>();
            var probTensors = new List<Tensor>();
            foreach (var (scaleIndex, featureMap) in Tools.Enumerate(modelOutput.SSBox, modelOutput.MBBox, modelOutput.LBBox)) {
                var outputTensors = Decode(featureMap,
                    outputSize: inputSize / strides[scaleIndex],
                    classCount: classCount,
                    strides: strides,
                    anchors: anchors,
                    scaleIndex: scaleIndex,
                    xyScale: xyScale);
                bboxTensors.Add(outputTensors.xywh);
                probTensors.Add(outputTensors.prob);
            }
            var bbox = tf.concat(bboxTensors.ToArray(), axis: 1);
            var prob = tf.concat(probTensors.ToArray(), axis: 1);

            var (boxes, conf) = FilterBoxes(bbox, prob,
                                            scoreThreshold: scoreThreshold,
                                            inputShape: tf.constant(new[] { inputSize, inputSize }));

            return tf.concat(new[] { boxes, conf }, axis: -1);
        }

        static (Tensor xywh, Tensor prob) Decode(
                                  Tensor convOut, int classCount, int outputSize,
                                  ReadOnlySpan<int> strides, Tensor<int> anchors,
                                  int scaleIndex, ReadOnlySpan<float> xyScale) {
            var pred = DecodeCommon(convOut,
                                    classCount: classCount, outputSize: outputSize,
                                    strides: strides, anchors: anchors,
                                    scaleIndex: scaleIndex,
                                    xyScale: xyScale);

            Tensor batchSize = tf.shape(convOut)[0];
            pred.prob = pred.conf * pred.prob;
            pred.prob = tf.reshape_dyn(pred.prob, new object[] { batchSize, -1, classCount });
            pred.xywh = tf.reshape_dyn(pred.xywh, new object[] { batchSize, -1, 4 });
            return (pred.xywh, pred.prob);
        }
    }
}