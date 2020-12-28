namespace tensorflow.keras.applications {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using LostTech.Gradient;
    using LostTech.Gradient.BuiltIns;

    using numpy;

    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.PixelFormats;

    using tensorflow.data;
    using tensorflow.image;
    using tensorflow.keras.models;
    static partial class YOLO {
        static Tensor DecodeEval(Tensor convOut, int classCount) {
            Tensor batchSize = tf.shape(convOut)[0];
            Tensor outputSize = tf.shape(convOut)[1];

            convOut = tf.reshape_dyn(convOut, new object[] { batchSize, outputSize, outputSize, 3, 5 + classCount });
            Tensor[] raws = tf.split(convOut, new[] { 4, 1, classCount }, axis: -1);
            var (convRawXYWH, convRawConf, convRawProb) = raws;

            Tensor predConf = tf.sigmoid(convRawConf);
            Tensor predProb = tf.sigmoid(convRawProb);

            return tf.concat(new[] { convRawXYWH, predConf, predProb }, axis: -1);
        }

        public static Model CreateV4EvalOnly(int inputSize, int classCount) {
            if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
            if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

            Tensor input = tf.keras.Input(new TensorShape(inputSize, inputSize, 3));
            var featureMaps = YOLOv4.Apply(input, classCount: classCount);

            var bboxTensors = new PythonList<Tensor>();
            foreach (var featureMap in new[] { featureMaps.SSBox, featureMaps.MBBox, featureMaps.LBBox }) {
                var bbox = DecodeEval(featureMap, classCount: classCount);
                bboxTensors.Add(bbox);
            }
            return new Model(new { inputs = input, outputs = bboxTensors }.AsKwArgs());
        }

        public static ObjectDetectionResult[] Detect(dynamic detector, Size supportedSize, Image<Rgb24> image) {
            if (detector is null) throw new ArgumentNullException(nameof(detector));
            if (image is null) throw new ArgumentNullException(nameof(image));

            var input = ImageTools.YoloPreprocess(new ObjectDetectionDataset.ClrEntry {
                Image = image.Clone(),
            }, supportedSize);
            var images = input.Image[np.newaxis, np.rest_of_the_axes].AsArray();

            IDictionary<string, Tensor> prediction = detector(tf.constant(images));
            _ArrayLike Get(string name) => prediction["tf_op_layer_" + name].numpy();
            ndarray<float> boxs = Get(nameof(SelectedBoxesOutput.Boxes)).AsArray<float>();
            ndarray<float> scores = Get(nameof(SelectedBoxesOutput.Scores)).AsArray<float>();
            ndarray<long> classes = Get(nameof(SelectedBoxesOutput.Classes)).AsArray<long>();
            ndarray<int> detections = Get(nameof(SelectedBoxesOutput.Detections)).AsArray<int>();

            return ObjectDetectionResult.FromCombinedNonMaxSuppressionBatch(
                boxs, scores, classes, detections[0].AsScalar());
        }

        public static ObjectDetectionResult[] DetectRaw(Model rawDetector,
                                                        Size supportedSize, int classCount,
                                                        Image<Rgb24> image,
                                                        ReadOnlySpan<int> strides, Tensor<int> anchors,
                                                        ReadOnlySpan<float> xyScale,
                                                        float scoreThreshold = 0.2f) {
            if (rawDetector is null) throw new ArgumentNullException(nameof(rawDetector));
            if (image is null) throw new ArgumentNullException(nameof(image));

            var input = ImageTools.YoloPreprocess(new ObjectDetectionDataset.ClrEntry {
                Image = image.Clone(),
            }, supportedSize);
            var images = input.Image[np.newaxis, np.rest_of_the_axes].AsArray();

            IList<Tensor> prediction = rawDetector.__call__(images);
            Debug.Assert(prediction.Count == 3);
            var output = new YOLOv4.Output {
                SSBox = prediction[0],
                MBBox = prediction[1],
                LBBox = prediction[2],
            };
            var suppression = SelectBoxes(output, inputSize: supportedSize.Width, classCount: classCount,
                                          strides: strides, anchors: anchors,
                                          xyScale: xyScale,
                                          scoreThreshold: scoreThreshold);

            ndarray<float> boxs = suppression.Boxes.numpy();
            ndarray<float> scores = suppression.Scores.numpy();
            ndarray<long> classes = suppression.Classes.numpy();
            ndarray<int> detections = suppression.Detections.numpy();

            return ObjectDetectionResult.FromCombinedNonMaxSuppressionBatch(
                boxs, scores, classes, detections[0].AsScalar());
        }

        public static SelectedBoxesOutput
                      SelectBoxes(YOLOv4.Output featureMaps, int inputSize, int classCount,
                                  ReadOnlySpan<int> strides, Tensor<int> anchors,
                                  ReadOnlySpan<float> xyScale,
                                  float scoreThreshold = 0.2f) {
            var pred = ProcessPrediction(inputSize: inputSize, featureMaps,
                                         classCount: classCount,
                                         strides: strides,
                                         anchors: anchors,
                                         xyScale: xyScale,
                                         scoreThreshold: scoreThreshold);

            var boxes = pred[.., .., 0..4];
            var conf = pred[.., .., 4..];

            var batchSize = tf.shape(boxes)[0];

            var suppression = tf.image.combined_non_max_suppression(
                boxes: tf.reshape_dyn(boxes, new object[] { batchSize, -1, 1, 4 }),
                scores: tf.reshape_dyn(conf, new object[] { batchSize, -1, tf.shape(conf)[^1] }),
                max_output_size_per_class: tf.constant(50),
                max_total_size: tf.constant(50),
                iou_threshold: 0.45f,
                score_threshold: 0.20f
            );
            return new SelectedBoxesOutput {
                Boxes = tf.identity(suppression[0], name: nameof(SelectedBoxesOutput.Boxes)),
                Scores = tf.identity(suppression[1], name: nameof(SelectedBoxesOutput.Scores)),
                Classes = tf.cast<long>(suppression[2], name: nameof(SelectedBoxesOutput.Classes)),
                Detections = tf.identity(suppression[3], name: nameof(SelectedBoxesOutput.Detections)),
            };
        }

        static ndarray<float> PostProcessBBBox(IEnumerable<ndarray<float>> predictions,
                                     ndarray<int> anchors,
                                     ReadOnlySpan<int> strides,
                                     ReadOnlySpan<float> xyScale) {
            foreach(var (scaleIndex, pred) in Tools.Enumerate(predictions.ToArray())) {
                var convShape = pred.shape;
                int outputSize = convShape.Item2;
                var convRawDxDy = pred[.., .., .., .., 0..2];
                var convRawDwDh = pred[.., .., .., .., 2..4];

                dynamic numpy = PythonModuleContainer.Get<np>();
                var sizeRange = Enumerable.Range(0, outputSize).ToNumPyArray();
                PythonList<ndarray<int>> tempGrid = numpy.meshgrid(sizeRange, sizeRange);
                ndarray<int> xyGrid = np.expand_dims(
                    np.stack(tempGrid, axis: -1),
                    axis: 2).AsArray<int>(); // [gx, gy, 1, 2]

                xyGrid = numpy.tile(np.expand_dims(xyGrid, axis: 0), new[] { 1, 1, 1, 3, 1 });
                var xyGridFloat = xyGrid.AsType<float>();

                var predXY = ((tf.sigmoid_dyn(convRawDxDy).numpy() * xyScale[scaleIndex]) - 0.5f * (xyScale[scaleIndex] - 1) + xyGridFloat) * strides[scaleIndex];
                ndarray<double> predWH = numpy.exp(convRawDwDh) * anchors[scaleIndex];

                pred[.., .., .., .., 0..2] = predXY;
                pred[.., .., .., .., 2..4] = predWH.AsType<float>();
            }

            var reshapedPredictions = predictions.Select(
                x => x.reshape(new[] { -1, (int)x.shape.Item5 }).AsArray<float>());
            return np.concatenate(reshapedPredictions, axis: 0);
        }

        static ndarray<float> PostProcessBoxes(ndarray<float> predictions, Size originalSize, int inputSize, float scoreThreshold) {
            var predXYWH = predictions[.., 0..4];
            var predConf = predictions[.., 4];
            var predProb = predictions[.., 5..];

            // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
            var predCoor = np.concatenate(new[]{
                    predXYWH[.., ..2] - predXYWH[..,2..]*0.5f,
                    predXYWH[.., ..2] + predXYWH[..,2..]*0.5f,
                }, axis: -1);

            // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
            var (h, w) = (originalSize.Height, originalSize.Width);
            float resizeRatio = Math.Min(inputSize * 1f / w, inputSize * 1f / h);

            float dw = (inputSize - resizeRatio * w) / 2;
            float dh = (inputSize - resizeRatio * h) / 2;

            for (int i = 0; i < 2; i++) {
                predCoor[.., i * 2] = (predCoor[.., i * 2] - dw) / resizeRatio;
                predCoor[.., i * 2 + 1] = (predCoor[.., i * 2 + 1] - dh) / resizeRatio;
            }

            // (3) clip some boxes those are out of range
            predCoor = np.concatenate(new[] {
                np.maximum(predCoor[.., ..2], np.zeros<float>(2)),
                np.maximum(predCoor[.., 2..], new []{w-1f, h-1f}.ToNumPyArray()),
            }, axis: -1);
            dynamic numpy = PythonModuleContainer.Get<np>();
            ndarray<bool> invalidMask = numpy.logical_or(predCoor[.., 0] > predCoor[.., 2], predCoor[.., 1] > predCoor[.., 3]);
            predCoor[invalidMask] = 0;

            // (4) discard some invalid boxes
            ndarray<float> bboxesScale = numpy.sqrt(numpy.multiply.reduce(
                predCoor[.., 2..4] - predCoor[.., 0..2],
                axis: -1));
            ndarray<bool> scaleMask = numpy.logical_and(0 < bboxesScale, bboxesScale < float.PositiveInfinity);

            // (5) discard some boxes with low scores
            var classes = predProb.argmax(axis: -1).AsArray<long>();
            ndarray<float> scores = predConf * predProb[numpy.arange(predCoor.shape.Item1), classes];
            ndarray<bool>? scoreMask = scores > scoreThreshold;
            ndarray<bool> mask = numpy.logical_and(scaleMask, scoreMask);

            var coords = predCoor[mask].AsArray();
            scores = scores[mask].AsArray();
            classes = classes[mask].AsArray();

            return np.concatenate(new [] {
                coords,
                scores[.., np.newaxis],
                classes[.., np.newaxis].AsArray().AsType<float>(),
            }, axis: -1);
        }

        public struct SelectedBoxesOutput {
            public Tensor<float> Boxes { get; set; }
            public Tensor<float> Scores { get; set; }
            public Tensor<long> Classes { get; set; }
            public Tensor<int> Detections { get; set; }
        }
    }
}
